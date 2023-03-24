# ------------------------------------------------------------------------
# Mofified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
from models.clip.clip import _MODELS, _download, available_models
from models.clip.model import ModifiedResNet


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone, num_channels: int, return_interm_layers: bool, feature_layer: str):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            parameter.requires_grad_(train_backbone)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            # Hard-coded backbone parameters
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            if hasattr(backbone, 'pool_proj'):
                return_layers = {'pool_proj': "0"}
            elif feature_layer == 'layer4':
                return_layers = {'layer4': "layer4"}
            elif feature_layer == 'layer3':
                return_layers = {'layer3': "layer3"}
            elif feature_layer == 'layer34':
                return_layers = {'layer3': "layer3", 'layer4': 'layer4'}
            elif feature_layer == 'layer23':
                return_layers = {'layer2': "layer2", 'layer3': "layer3"}
            elif feature_layer == 'layer234':
                return_layers = {'layer2': "layer2", 'layer3': "layer3", 'layer4': 'layer4'}
            # Hard-coded backbone parameters
            # doesn't seem to be used
            if feature_layer != 'layer3' and feature_layer != 'layer34':
                self.strides = [32]
            else:
                self.strides = [16]
            if feature_layer == 'layer234':
                self.num_channels = [num_channels // 4, num_channels // 2, num_channels]
            elif feature_layer == 'layer34':
                self.num_channels = [num_channels // 2, num_channels]
            elif feature_layer == 'layer3':
                self.num_channels = [num_channels // 2]
            else:
                self.num_channels = [num_channels]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 feature_layer,
                 args):

        dilation = args.dilation, 
        no_clip_init = args.no_clip_init_image
        model_path = args.model_path
        region_prompt_path = args.region_prompt_path

        if "clip" in name and not no_clip_init:
            name = name.replace('clip_', '')
            if model_path:
                pass
            elif name in _MODELS:
                model_path = _download(_MODELS[name], os.path.expanduser("~/.cache/clip"))
            elif os.path.isfile(name):
                model_path = name
            else:
                raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

            with open(model_path, 'rb') as opened_file:
                model = torch.jit.load(opened_file, map_location="cpu")
                state_dict = model.state_dict()
                
            vit = "visual.proj" in state_dict

            if vit:
                vision_width = state_dict["visual.conv1.weight"].shape[0]
                vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
                vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
                grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
                image_resolution = vision_patch_size * grid_size
            else:
                counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
                vision_layers = tuple(counts)
                vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
                vision_patch_size = None
                output_dim, embed_dim = state_dict['visual.attnpool.c_proj.weight'].shape
                vision_heads = vision_width * 32 // 64
                image_resolution = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5) * 32
            
            if name in ['RN50', 'RN50x4']:
                backbone = ModifiedResNet(layers=vision_layers, output_dim=output_dim, heads=vision_heads, input_resolution=image_resolution, 
                                          width=vision_width, bn=FrozenBatchNorm2d, args=args)
                new_state_dict = dict()
                num_channels = embed_dim
                new_state_dict.update({k.replace('visual.', ''): v for k, v in state_dict.items() if k.startswith('visual.')})
                if feature_layer not in ['layer3', 'layer4', 'layer34', 'layer23', 'layer234']:
                    num_channels = state_dict['visual.attnpool.c_proj.weight'].size(0)
                if region_prompt_path:
                    region_prompt = torch.load(region_prompt_path, map_location='cpu')
                    new_state_dict.update(region_prompt)
                backbone.load_state_dict(new_state_dict)
        else:
            if name == 'clip_RN50' and no_clip_init:
                name = 'resnet50'
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
            assert name not in ('resnet18', 'resnet34'), "Number of channels are hard coded, thus do not support res18/34."
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, feature_layer)
        self.attn_pool = backbone.attnpool
        if feature_layer in ['layer3', 'layer34', 'layer23', 'layer234', 'layer4']:
            # assume that we apply layer4 on roi feature
            self.layer4 = backbone.layer4


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        pos = dict()
        for name, x in xs.items():
            # position encoding
            pos[name] = self[1](x).to(x.tensors.dtype)
        return xs, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    return_interm_layers = args.masks or args.multiscale
    if args.backbone_feature == 'layer4':
        feature_layer = 'layer4'
    elif args.anchor_pre_matching:
        feature_layer = 'layer34'
    else:
        feature_layer = 'layer3'
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, feature_layer=feature_layer, args=args)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
