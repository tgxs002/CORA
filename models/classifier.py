# ------------------------------------------------------------------------
# Mofified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
import os
from sqlite3 import adapt
from unicodedata import category
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
from models.clip.clip import _MODELS, _download, available_models, tokenize
from models.clip.model import Transformer
from models.clip.prompts import imagenet_templates
from .misc import MLP
import random

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Classifier(torch.nn.Module):
    def __init__(self, name: str,
                 train_backbone: bool,
                 token_len = 77, no_clip_init=False, use_adapter=False, use_prompt=False, adapter_dim=0):
        super().__init__()
        self.train_backbone = train_backbone
        if not self.train_backbone:
            self.cache = {}
        
        if "clip" in name:
            name = name.replace('clip_', '')
            if name in _MODELS:
                model_path = _download(_MODELS[name], os.path.expanduser("~/.cache/clip"))
            elif os.path.isfile(name):
                model_path = name
            else:
                raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

            with open(model_path, 'rb') as opened_file:
                model = torch.jit.load(opened_file, map_location="cpu")
                state_dict = model.state_dict()

            embed_dim = state_dict["text_projection"].shape[1]
            # self.context_length = state_dict["positional_embedding"].shape[0]
            self.context_length = token_len
            self.vocab_size = state_dict["token_embedding.weight"].shape[0]
            transformer_width = state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
            
            if use_prompt:
                self.context_length += 10
            
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )
            
            self.token_embedding = nn.Embedding(self.vocab_size, transformer_width)
            self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
            self.ln_final = LayerNorm(transformer_width)
            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
            
            # load 
            if not no_clip_init:
                self.transformer.load_state_dict({k.replace('transformer.', ''): v for k, v in state_dict.items() if k.startswith('transformer.')})
                self.token_embedding.load_state_dict({k.replace('token_embedding.', ''): v for k, v in state_dict.items() if 'token_embedding' in k})
                self.ln_final.load_state_dict({k.replace('ln_final.', ''): v for k, v in state_dict.items() if 'ln_final' in k})
                self.positional_embedding.data = state_dict['positional_embedding']
                self.text_projection.data = state_dict['text_projection']
            else:
                nn.init.normal_(self.token_embedding.weight, std=0.02)
                nn.init.normal_(self.positional_embedding, std=0.01)
                
                proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
                attn_std = self.transformer.width ** -0.5
                fc_std = (2 * self.transformer.width) ** -0.5
                for block in self.transformer.resblocks:
                    nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                    nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                    nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                    nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
                    
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            
            if not self.train_backbone:
                for v in self.parameters():
                    v.requires_grad_(False)
                    
            self.use_adapter = use_adapter
            self.use_prompt = use_prompt
            if use_adapter:
                assert not self.train_backbone
                embed_dim = self.text_projection.size(1)
                self.prompt_text_adapter = MLP(embed_dim, adapter_dim, embed_dim, 2, bias=False)
            if use_prompt:
                self.prompt_text_token = nn.Embedding(10, transformer_width)
                self.prompt_text_token.weight.data = self.token_embedding.weight[[random.randint(0, self.token_embedding.weight.size(0)-1) for _ in range(10)]].detach().clone()
        else:
            assert train_backbone
            raise NotImplementedError
            
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_text(self, text):
        x = self.token_embedding(text)
        
        if self.use_prompt:
            x = torch.cat([self.prompt_text_token.weight[None].expand(x.size(0), -1, -1), x], dim=1)
            text = torch.cat([text[:,:10] * 0, text], dim=-1)

        x = x + self.positional_embedding[:self.context_length]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.to(x.dtype)

        return x
    
    def forward_feature(self, category_list):
        if self.use_prompt:
            templates = ['{}']
        elif self.train_backbone or self.use_adapter:
            templates = ['a photo of {}']
        else:
            templates = imagenet_templates
        texts = [template.format(cetegory) for cetegory in category_list for template in templates] #format with class
        texts = tokenize(texts, context_length=self.context_length - 10 if self.use_prompt else self.context_length, truncate=True).to(self.positional_embedding.device)
        class_embeddings = []
        cursor = 0
        step = 3000
        while cursor <= len(texts):
            class_embeddings.append(self.encode_text(texts[cursor:cursor + step]))
            cursor += step
        class_embeddings = torch.cat(class_embeddings)
        if self.use_adapter:
            class_embeddings = class_embeddings + self.prompt_text_adapter(class_embeddings)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.unflatten(0, (len(category_list), len(templates)))
        class_embedding = class_embeddings.mean(dim=1)
        class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        return class_embedding
    
    def forward(self, category_list):
        if not self.train_backbone and not self.use_adapter and not self.use_prompt:
            new_category = [category for category in category_list if category not in self.cache]
            new_class_embedding = self.forward_feature(new_category)
            for category, feat in zip(new_category, new_class_embedding):
                self.cache[category] = feat.to('cpu')
            class_embedding = torch.stack([self.cache[category] for category in category_list]).to(self.positional_embedding.device)
        
        else:
            class_embedding = self.forward_feature(category_list)

        return class_embedding

def build_classifier(args):
    train_backbone = args.lr_language > 0
    classifier = Classifier(args.backbone, train_backbone, token_len=args.text_len, 
                            no_clip_init=args.no_clip_init or args.no_clip_init_text, use_adapter=args.text_adapter, use_prompt=args.text_prompt, adapter_dim=args.adapter_dim)
    return classifier
