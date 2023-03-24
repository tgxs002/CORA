# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import box_ops
from torch.nn.utils.rnn import pad_sequence
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid, get_rank)

from .backbone import build_backbone
from .classifier import build_classifier
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm, dice_loss, sigmoid_focal_loss)
from .dab_transformer import build_transformer as build_dab_transformer
from .dab_transformer import gen_sineembed_for_position
from .misc import _get_clones, MLP
import torch.distributed as dist
import copy

# for offline rpn
from detectron2.config import get_cfg
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone as build_offline_backbone
from detectron2.structures import ImageList
from detectron2.layers import ShapeSpec, batched_nms

from util.box_ops import box_iou, box_cxcywh_to_xyxy, generalized_box_iou

import torchvision
from models.attention import multi_head_attention_forward_trans as MHA_woproj


class FastDETR(nn.Module):
    def __init__(self, args, backbone, transformer, classifier, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         that our model can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.args = args
        self.multiscale = args.multiscale
        self.num_feature_levels = 3 if self.multiscale else 1          # Hard-coded multiscale parameters
        self.num_queries = args.num_queries
        self.aux_loss = args.aux_loss
        self.hidden_dim = args.hidden_dim
        assert self.hidden_dim == transformer.d_model

        self.backbone = backbone
        self.transformer = transformer
        self.classifier = classifier

        # Instead of modeling query_embed as learnable parameters in the shape of (num_queries, d_model),
        # we directly model reference boxes in the shape of (num_queries, 4), in the format of (xc yc w h).
        if not self.args.rpn:
            self.query_embed = nn.Embedding(self.num_queries, 4)           # Reference boxes
        else:
            self.query_embed = None

        # ====================================================================================
        #                                   * Clarification *
        #  -----------------------------------------------------------------------------------
        #  Whether self.input_proj contains nn.GroupNorm should not affect performance much.
        #  nn.GroupNorm() is introduced in some of our experiments by accident.
        #  Our experience shows that it even slightly degrades the performance.
        #  We recommend you to simply delete nn.GroupNorm() in self.input_proj for your own
        #  experiments, but if you wish to reproduce our results, please follow our setups below.
        # ====================================================================================
        if self.multiscale:
            input_proj_list = []
            for _ in range(self.num_feature_levels):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            if self.args.epochs >= 25:
                kernel_size = 1
                padding = 0
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=kernel_size, padding=padding),
                    )])
                if self.args.add_gn:
                    self.input_proj[0].append(nn.GroupNorm(32, self.hidden_dim))
            else:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )])

        # self.class_embed = nn.Linear(self.hidden_dim, num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.args.end2end:
            self.image_proj = nn.Linear(self.hidden_dim, self.classifier.text_projection.data.size(1))
            self.class_bias = nn.Parameter(torch.ones([]) * bias_value)
            self.tau = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.objectness_embed = nn.Linear(self.hidden_dim, 1)
            if not self.args.disable_init:
                nn.init.constant_(self.objectness_embed.bias, bias_value)

        if self.args.rpn:
            self.stage1_box_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
            self.stage1_obj_embed = nn.Linear(self.hidden_dim, 1)
            if not self.args.disable_init:
                nn.init.constant_(self.stage1_obj_embed.bias, bias_value)

        # init bbox_embed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if self.aux_loss:
            self.bbox_embed = _get_clones(self.bbox_embed, args.dec_layers)
            self.transformer.decoder.bbox_embed = self.bbox_embed
            if self.args.rpn:
                self.transformer.stage1_box_embed = self.stage1_box_embed
                self.transformer.stage1_obj_embed = self.stage1_obj_embed
        else:
            self.bbox_embed = _get_clones(self.bbox_embed, 1)

        if self.args.condition_on_text:
            # hard code the number of dimension
            text_dim = self.args.text_dim
            self.text_proj = MLP(text_dim, self.args.condition_bottleneck, self.hidden_dim, 2)

        if self.args.test_attnpool_path:
            self.test_attnpool = [torch.load(self.args.test_attnpool_path)]

    def forward(self, samples: NestedTensor, categories, gt_classes=None, targets=None, split_class=False):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        if self.args.lr_backbone > 0:
            features, pos_embeds = self.backbone(samples)
        else:
            with torch.no_grad():
                features, pos_embeds = self.backbone(samples)

        src, mask = features[self.args.backbone_feature].decompose()
        srcs = [self.input_proj[0](src)]
        masks = [mask]
        assert mask is not None

        if not self.args.rpn:
            query = self.query_embed.weight
            query = query.unsqueeze(1).expand(-1, srcs[0].size(0), -1)
        else:
            query = None

        def get_query_and_mask(query):
            query_features = None
            sa_mask = None
            classes_ = None
            confidences = None
            if self.args.anchor_pre_matching:
                # classify the proposal
                # TODO: try align the feature with training stage
                # if any error here, make sure layer4 is passed
                src_feature = features['layer4']
                sizes = [((1 - m[0].float()).sum(), (1 - m[:,0].float()).sum()) for m in src_feature.decompose()[1]]
                
                if self.args.box_conditioned_pe:
                    box_emb = gen_sineembed_for_position(query.sigmoid())[:,:,256:]
                else:
                    box_emb = None

                with torch.no_grad():
                    roi_features = self._sample_feature(sizes, query.sigmoid().permute(1,0,2), src_feature.tensors, extra_conv=False, box_emb=box_emb)
                    # classify the proposals
                    text_feature = self.classifier(categories)
                
                if split_class:
                    # split class
                    split_mask = torch.rand_like(text_feature[:,0]) > 0.5
                    text_feature1 = text_feature.clone()
                    text_feature2 = text_feature.clone()
                    text_feature1[split_mask] = 0.0
                    text_feature2[~split_mask] = 0.0
                    outputs_class1 = roi_features @ text_feature1.t()
                    outputs_class2 = roi_features @ text_feature2.t()
                    outputs_class = torch.cat([outputs_class1, outputs_class2])
                    ori_bs = len(sizes)
                    sizes = [*sizes, *sizes]
                    query = query.repeat(1, 2, 1)
                    for k in features.keys():
                        features[k].tensors = features[k].tensors.repeat(2, 1, 1, 1)
                        features[k].mask = features[k].mask.repeat(2, 1, 1)
                else:
                    outputs_class = roi_features @ text_feature.t()
                    
                with torch.no_grad():
                    if gt_classes is not None:
                        new_outputs_class = torch.zeros_like(outputs_class)
                        for i, gt_idx in enumerate(gt_classes):
                            new_outputs_class[i, :, gt_idx] = outputs_class[i, :, gt_idx]
                        outputs_class = new_outputs_class
                    outputs_class = torch.cat([outputs_class, torch.ones_like(outputs_class[:,:,:1]) * self.args.bg_threshold], dim=-1)
                    if self.args.softmax_along == 'class':
                        outputs_class = (outputs_class * 100).softmax(dim=-1)
                    elif self.args.softmax_along == 'box':
                        outputs_class = (outputs_class * 100).softmax(dim=-2)
                    elif self.args.softmax_along == 'none':
                        pass
                    if self.args.target_class_factor != 1.0 and not self.training:
                        if outputs_class.size(-1) == 66:
                            # COCO
                            target_index = [4, 5, 11, 12, 15, 16, 21, 23, 27, 29, 32, 34, 45, 47, 54, 58, 63]
                        elif outputs_class.size(-1) == 1204:
                            # LVIS
                            target_index = [12, 13, 16, 19, 20, 29, 30, 37, 38, 39, 41, 48, 50, 51, 62, 68, 70, 77, 81, 84, 92, 104, 105, 112, 116, 118, 122, 125, 129, 130, 135, 139, 141, 143, 146, 150, 154, 158, 160, 163, 166, 171, 178, 181, 195, 201, 208, 209, 213, 214, 221, 222, 230, 232, 233, 235, 236, 237, 239, 243, 244, 246, 249, 250, 256, 257, 261, 264, 265, 268, 269, 274, 280, 281, 286, 290, 291, 293, 294, 299, 300, 301, 303, 306, 309, 312, 315, 316, 320, 322, 325, 330, 332, 347, 348, 351, 352, 353, 354, 356, 361, 363, 364, 365, 367, 373, 375, 380, 381, 387, 388, 396, 397, 399, 404, 406, 409, 412, 413, 415, 419, 425, 426, 427, 430, 431, 434, 438, 445, 448, 455, 457, 466, 477, 478, 479, 480, 481, 485, 487, 490, 491, 502, 505, 507, 508, 512, 515, 517, 526, 531, 534, 537, 540, 541, 542, 544, 550, 556, 559, 560, 566, 567, 570, 571, 573, 574, 576, 579, 581, 582, 584, 593, 596, 598, 601, 602, 605, 609, 615, 617, 618, 619, 624, 631, 633, 634, 637, 639, 645, 647, 650, 656, 661, 662, 663, 664, 670, 671, 673, 677, 685, 687, 689, 690, 692, 701, 709, 711, 713, 721, 726, 728, 729, 732, 742, 751, 753, 754, 757, 758, 763, 768, 771, 777, 778, 782, 783, 784, 786, 787, 791, 795, 802, 804, 807, 808, 809, 811, 814, 819, 821, 822, 823, 828, 830, 848, 849, 850, 851, 852, 854, 855, 857, 858, 861, 863, 868, 872, 882, 885, 886, 889, 890, 891, 893, 901, 904, 907, 912, 913, 916, 917, 919, 924, 930, 936, 937, 938, 940, 941, 943, 944, 951, 955, 957, 968, 971, 973, 974, 982, 984, 986, 989, 990, 991, 993, 997, 1002, 1004, 1009, 1011, 1014, 1015, 1027, 1028, 1029, 1030, 1031, 1046, 1047, 1048, 1052, 1053, 1056, 1057, 1074, 1079, 1083, 1115, 1117, 1118, 1123, 1125, 1128, 1134, 1143, 1144, 1145, 1147, 1149, 1156, 1157, 1158, 1164, 1166, 1192]
                        else:
                            assert False, "the dataset may not be supported"
                        outputs_class[:,:,target_index] = outputs_class[:,:,target_index] * self.args.target_class_factor
                    if self.args.bg_threshold == -1.:
                        outputs_class = outputs_class[:,:,:-1]
                        num_classes = len(categories)
                    else:
                        text_feature = torch.cat([text_feature, torch.zeros_like(text_feature[:1])], dim=0)
                        num_classes = len(categories) + 1
                    if self.args.global_topk:
                        confidences, selected_indices = outputs_class.flatten(1,2).topk(k=1000, dim=-1)
                        indices = selected_indices // num_classes
                        classes_ = selected_indices % num_classes
                        indices = indices.permute(1,0).unsqueeze(-1).expand(indices.size(1), indices.size(0), 4)
                    else:
                        if self.args.resample_factor != 1.0:
                            assert False
                            ori_dist = outputs_class.sum(1)
                            if self.args.filter_classes:
                                filtered_dist = torch.zeros_like(ori_dist)
                                for i, cls in enumerate(outputs_class.max(dim=-1)[1]):
                                    filtered_dist[i, torch.unique(cls)] = ori_dist[i, torch.unique(cls)]
                                ori_dist = filtered_dist
                            # OT requires extra precision
                            adjusted_dist = (ori_dist ** self.args.resample_factor).to(torch.float64)
                            class_dist = adjusted_dist / adjusted_dist.sum(-1, keepdim=True)
                            box_dist = torch.zeros_like(outputs_class[:,:,0]).to(torch.float64) + (1 / outputs_class.size(1))
                            classes = []
                            for b, c, cost in zip(box_dist, class_dist, -outputs_class):
                                classes.append(ot.emd(b, c, cost))
                            classes = torch.stack(classes).max(-1)[1]
                        else:
                            if self.args.topk_matching <= 0:
                                classes_ = outputs_class.max(-1)[1]
                            else:
                                classes_ = outputs_class.topk(dim=-1, k=self.args.topk_matching)[1]
                            classes_, indices = classes_.sort(-1)
                            indices = indices.permute(1,0).unsqueeze(-1).expand(indices.size(1), indices.size(0), 4)
                query = torch.gather(query, 0, indices)
                sa_mask = None
                if self.args.condition_on_text:
                    projected_text = self.text_proj(text_feature)
                    if classes_.dim() == 3:
                        used_classes_ = classes_[:,:,0]
                    else:
                        used_classes_ = classes_
                    query_features = F.one_hot(used_classes_, num_classes=text_feature.size(0)).to(text_feature.dtype) @ projected_text
            return classes_, query_features, query, confidences

        if not self.args.rpn:
            classes_, query_features, query, confidences = get_query_and_mask(query)
        else:
            query_features = None
            classes_ = None
            confidences = None

        used_pos_embed = [pos_embeds[self.args.backbone_feature]]

        hs, reference, memory, classes_temp, out_dict, confidences_ = self.transformer(srcs, masks, query, used_pos_embed, tgt_mask=None, src_query=query_features, cls_func=get_query_and_mask)
        if classes_temp is not None and classes_ is None:
            classes_ = classes_temp
        if confidences_ is not None and confidences is None:
            confidences = confidences_

        if hs.size(0) != reference.size(0):
            hs = hs[-reference.size(0):]

        outputs_coords = []
        outputs_class = []
        for lvl in range(reference.shape[0]):
            reference_before_sigmoid = inverse_sigmoid(reference[lvl])
            bbox_offset = self.bbox_embed[lvl](hs[lvl])
            outputs_coord = (reference_before_sigmoid + bbox_offset).sigmoid()
            outputs_coords.append(outputs_coord)
            if self.args.end2end:
                image_feat = self.image_proj(hs[lvl])
                image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
                text_feature = self.classifier(categories)
                similarity = image_feat @ text_feature.t()
                logits = similarity * self.tau.exp() + self.class_bias
                outputs_class.append(logits)

        outputs_coords = torch.stack(outputs_coords)

        if self.args.end2end:
            outputs_class = torch.stack(outputs_class)
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coords[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coords)
            return out
        else:
            objectness_score = self.objectness_embed(hs)

        roi_feats = []
        if not self.training:
            src_feature = features['layer4' if self.training else 'layer3']
            sizes = [((1 - m[0].float()).sum(), (1 - m[:,0].float()).sum()) for m in src_feature.decompose()[1]]

            boxes = outputs_coords
            
            if self.aux_loss and self.training:
                sample_box = boxes
            else:
                sample_box = boxes[-1:]

            for coord in sample_box:
                if self.args.box_conditioned_pe or hasattr(self, 'test_attnpool'):
                    box_emb = gen_sineembed_for_position(coord.permute(1,0,2))[:,:,256:]
                else:
                    box_emb = None
                roi_feats.append(self._sample_feature(sizes, coord, src_feature.tensors, extra_conv=not self.training, box_emb=box_emb))

            roi_features = roi_feats[-1]
            roi_feats = torch.stack(roi_feats)

        out = {'pred_logits': objectness_score[-1], 'pred_boxes': outputs_coords[-1]}
        if query is not None:
            out['proposal'] = query.sigmoid()

        if self.args.masks:
            out['last_hs'] = hs[-1]
            out['memory'] = memory
            out['attn_masks'] = masks
        
        if split_class:
            out['split_class'] = True
            
                
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(objectness_score, outputs_coords)

        if self.args.remove_misclassified:
            out['proposal_classes'] = classes_
            if self.aux_loss:
                for aux in out['aux_outputs']:
                    aux['proposal_classes'] = classes_

        if self.args.semantic_cost > 0 or self.args.matching_threshold >= 0:
            out['text_feature'] = self.classifier(categories)
            if self.aux_loss:
                for aux in out['aux_outputs']:
                    aux['text_feature'] = self.classifier(categories)
        
        # add the rpn prediction
        if out_dict is not None:
            out['rpn_output'] = out_dict

        if self.args.iou_rescore:
            iou_objectness = []
            for coord, target in zip(outputs_coords[-1], targets):
                iou_objectness.append(box_iou(box_ops.box_cxcywh_to_xyxy(coord), box_ops.box_cxcywh_to_xyxy(target['boxes']))[0].max(dim=-1)[0])
            iou_objectness = torch.stack(iou_objectness).unsqueeze(-1)
            iou_objectness = inverse_sigmoid(iou_objectness)
            objectness_score[-1] = iou_objectness

        if not self.training:

            if self.args.aggresive_eval:
                if self.args.iou_relabel_eval:
                    ious = [generalized_box_iou(box_cxcywh_to_xyxy((out['pred_boxes'][i])), box_cxcywh_to_xyxy(target['boxes'])) for i, target in enumerate(targets)]
                    gt_idx = [iou.argmax(dim=-1) for iou in ious]
                    gt_labels = [torch.gather(target['labels'], dim=0, index=gt_idx_) for target, gt_idx_ in zip(targets, gt_idx)]
                    gt_labels = torch.stack(gt_labels)
                    classes_ = gt_labels
                scores = F.one_hot(classes_, num_classes=len(categories))
                if confidences is not None:
                    scores = scores * confidences.unsqueeze(-1)
                scores = scores * objectness_score[-1].sigmoid()
                out['pred_logits'] = inverse_sigmoid(scores)
            else:
                # the text feature
                text_feature = self.classifier(categories)
                outputs_class = roi_features @ text_feature.t()
                outputs_class = torch.cat([outputs_class, torch.zeros_like(outputs_class[:,:,:1])], dim=-1)
                outputs_class = (outputs_class * self.args.eval_tau).softmax(dim=-1) * (objectness_score[-1].sigmoid() ** self.args.objectness_alpha)
                outputs_class = outputs_class[:,:,:-1]
                outputs_class = inverse_sigmoid(outputs_class)
                out['pred_logits'] = outputs_class

        if self.args.use_nms and not self.args.no_nms:
            out['use_nms'] = 0

        return out

    @torch.no_grad()
    def _sample_feature(self, sizes, pred_boxes, features, extra_conv, unflatten=True, box_emb=None):
        rpn_boxes = [box_ops.box_cxcywh_to_xyxy(pred) for pred in pred_boxes]
        for i in range(len(rpn_boxes)):
            rpn_boxes[i][:,[0,2]] = rpn_boxes[i][:,[0,2]] * sizes[i][0]
            rpn_boxes[i][:,[1,3]] = rpn_boxes[i][:,[1,3]] * sizes[i][1]

        if self.args.backbone == 'clip_RN50x4':
            reso = 18
        elif self.args.backbone == 'clip_RN50':
            reso = 14

        if self.args.no_efficient_pooling or extra_conv:
            roi_features = torchvision.ops.roi_align(
                features,
                rpn_boxes,
                output_size=(reso, reso) if extra_conv else (reso // 2, reso // 2),
                spatial_scale=1.0,
                aligned=True)  # (bs * num_queries, c, 14, 14)

            if extra_conv:
                roi_features = self.backbone[0].layer4(roi_features)
            if hasattr(self, 'test_attnpool'):
                self.test_attnpool[0].to(roi_features.device)
                roi_features = self.test_attnpool[0](roi_features, box_emb)
            else:
                roi_features = self.backbone[0].attn_pool(roi_features, box_emb)
        else:
            with torch.cuda.amp.autocast(enabled=False):
                features = features.permute(0,2,3,1)
                attn_pool = self.backbone[0].attn_pool
                q_feat = attn_pool.q_proj(features)
                k_feat = attn_pool.k_proj(features)
                v_feat = attn_pool.v_proj(features)
                hacked = False
                if box_emb is not None:
                    if not self.args.use_efficient_pe_proj:
                        positional_emb = attn_pool.positional_embedding(box_emb)
                    else:
                        # efficient hack implementation of attn_pool.positional_embedding
                        hacked = True
                        pe = attn_pool.positional_embedding
                        x = pe.mlp(box_emb).unflatten(-1, (pe.num_embeddings, -1)).permute(2,1,0,3).flatten(1,2)
                        if not hasattr(self, "hack_q_weight"):
                            assert pe.proj_dim > 0
                            self.hack_q_weight = attn_pool.q_proj.weight @ pe.out_proj.weight
                            self.hack_k_weight = attn_pool.k_proj.weight @ pe.out_proj.weight
                            self.hack_v_weight = attn_pool.v_proj.weight @ pe.out_proj.weight
                            self.hack_q_bias = pe.pe_bias @ attn_pool.q_proj.weight.t()
                            self.hack_k_bias = pe.pe_bias @ attn_pool.k_proj.weight.t()
                            self.hack_v_bias = pe.pe_bias @ attn_pool.v_proj.weight.t()
                        q_pe = F.linear(x[:1], self.hack_q_weight) + self.hack_q_bias[:1,None]
                        k_pe = F.linear(x[1:], self.hack_k_weight) + self.hack_k_bias[1:,None]
                        v_pe = F.linear(x[1:], self.hack_v_weight) + self.hack_v_bias[1:,None]
                else:
                    positional_emb = attn_pool.positional_embedding
                
                if not hacked:
                    q_pe = F.linear(positional_emb[:1], attn_pool.q_proj.weight)
                    k_pe = F.linear(positional_emb[1:], attn_pool.k_proj.weight)
                    v_pe = F.linear(positional_emb[1:], attn_pool.v_proj.weight)
                if q_pe.dim() == 3:
                    assert q_pe.size(0) == 1
                    q_pe = q_pe[0]
                # actually this is the correct code. I keep a bug here to trade accuracy for efficiency
                # k_pe = F.linear(attn_pool.positional_embedding, attn_pool.k_proj.weight)
                # v_pe = F.linear(attn_pool.positional_embedding, attn_pool.v_proj.weight)
                q, k, v = q_feat.permute(0,3,1,2), k_feat.permute(0,3,1,2), v_feat.permute(0,3,1,2)
                q = torchvision.ops.roi_align(
                    q,
                    rpn_boxes,
                    output_size=(reso // 2, reso // 2),
                    spatial_scale=1.0,
                    aligned=True)
                k = torchvision.ops.roi_align(
                    k,
                    rpn_boxes,
                    output_size=(reso // 2, reso // 2),
                    spatial_scale=1.0,
                    aligned=True)
                v = torchvision.ops.roi_align(
                    v,
                    rpn_boxes,
                    output_size=(reso // 2, reso // 2),
                    spatial_scale=1.0,
                    aligned=True)
                
                q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
                q = q.mean(-1) # NC
                q = q + q_pe # NC
                if k_pe.dim() == 3:
                    k = k + k_pe.permute(1,2,0)
                    v = v + v_pe.permute(1,2,0)
                else:
                    k = k + k_pe.permute(1,0).unsqueeze(0).contiguous() # NC(HW)
                    v = v + v_pe.permute(1,0).unsqueeze(0).contiguous() # NC(HW)
                q = q.unsqueeze(-1)
                roi_features = MHA_woproj(q, k, v, k.size(-2), attn_pool.num_heads, in_proj_weight=None, in_proj_bias=None,
                    bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=attn_pool.c_proj.weight,
                    out_proj_bias=attn_pool.c_proj.bias, training=False, out_dim=k.size(-2), need_weights=False)[0][0]
                roi_features = roi_features.float()
            
        roi_features = roi_features / roi_features.norm(dim=-1, keepdim=True)
        if unflatten:
            roi_features = roi_features.unflatten(0, (features.size(0), -1))
        return roi_features

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coords):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coords[:-1])]


class SetCriterion(nn.Module):
    """ 
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, rpn_cls_cost):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        # self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.rpn_cls_cost = rpn_cls_cost

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], src_logits.size(-1), dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2, reduce=False)
        weight_mask = torch.ones_like(loss_ce)
        for i, ig in enumerate(outputs['ignore']):
            weight_mask[i, ig] = 0.0
        loss_ce = loss_ce * weight_mask
        loss_ce = loss_ce.mean(1).sum() / num_boxes * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        objectness = outputs['pred_logits']
        device = objectness.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (objectness.argmax(-1) != objectness.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if 'ignore' in outputs_without_aux:
            outputs['ignore'] = outputs_without_aux['ignore']

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = sum([index[0].numel() for index in indices])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'split_class' not in outputs:
            losses['matched_gt'] = torch.zeros_like(losses['loss_bbox']) + num_boxes / sum(len(t["labels"]) for t in targets)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets[:aux_outputs['pred_logits'].size(0)])
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets[:aux_outputs['pred_logits'].size(0)], indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'rpn_output' in outputs:
            # hack implementation of the different rpn cost
            ori_cls_cost = self.matcher.cost_class
            self.matcher.cost_class = self.rpn_cls_cost
            indices = self.matcher(outputs['rpn_output'], targets[:outputs['rpn_output']['pred_logits'].size(0)])
            self.matcher.cost_class = ori_cls_cost
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, outputs['rpn_output'], targets[:outputs['rpn_output']['pred_logits'].size(0)], indices, num_boxes, **kwargs)
                l_dict = {k + f'_rpn': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, args):
        super().__init__()
        if args.dataset_file == 'lvis':
            self.max_det = 300
        else:
            self.max_det = 100

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        if 'use_nms' in outputs:
            score_threshold = 0.001
            nms_thres = 0.5
            boxes__ = box_ops.box_cxcywh_to_xyxy(out_bbox)
            scores = []
            labels = []
            boxes = []
            out_logits = out_logits.sigmoid()
            for class_logit, coords in zip(out_logits, boxes__):
                valid_mask = torch.isfinite(coords).all(dim=1) & torch.isfinite(class_logit).all(dim=1)
                if not valid_mask.all():
                    coords = coords[valid_mask]
                    class_logit = class_logit[valid_mask]

                coords = coords.unsqueeze(1)
                filter_mask = class_logit > score_threshold
                filter_inds = filter_mask.nonzero()
                coords = coords[filter_inds[:, 0], 0]
                scores_ = class_logit[filter_mask]
                keep = batched_nms(coords, scores_, filter_inds[:, 1], nms_thres)
                keep = keep[:self.max_det]
                coords, scores_, filter_inds = coords[keep], scores_[keep], filter_inds[keep]
                scores.append(scores_)
                labels.append(filter_inds[:, 1])
                boxes.append(coords)
        else:
            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = [box * fct[None] for box, fct in zip(boxes, scale_fct)]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


def build(args):
    device = torch.device(args.device)
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250

    backbone = build_backbone(args)
    classifier = build_classifier(args)
        
    transformer = build_dab_transformer(args)
    model = FastDETR(args, backbone, transformer, classifier, num_classes=num_classes)
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
    }
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.rpn:
        weight_dict[f'loss_ce_rpn'] = args.cls_loss_coef_rpn
        weight_dict[f'loss_bbox_rpn'] = args.bbox_loss_coef
        weight_dict[f'loss_giou_rpn'] = args.giou_loss_coef

    if args.masks:
        losses += ["masks"]
    
    criterion = SetCriterion(num_classes,
                            matcher=matcher,
                            weight_dict=weight_dict,
                            focal_alpha=args.focal_alpha,
                            losses=losses,
                            rpn_cls_cost=args.set_cost_class_rpn)
    criterion.to(device)
    post_processors = {'bbox': PostProcess(args)}
    if args.masks:
        post_processors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            post_processors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, post_processors
