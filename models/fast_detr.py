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
from .transformer import build_transformer
from .misc import _get_clones, MLP
import torch.distributed as dist


class FastDETR(nn.Module):
    """ This is the SAM-DETR module that performs object detection """
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
        # self.multiscale = args.multiscale
        # self.num_feature_levels = 3 if self.multiscale else 1          # Hard-coded multiscale parameters
        self.num_queries = args.num_queries
        self.aux_loss = args.aux_loss
        self.hidden_dim = args.hidden_dim
        # assert self.hidden_dim == transformer.d_model

        self.backbone = backbone
        self.classifier = classifier
        self.tau = 100 # hard code 
        
        if args.use_proposal:
            # Instead of modeling query_embed as learnable parameters in the shape of (num_queries, d_model),
            # we directly model reference boxes in the shape of (num_queries, 4), in the format of (xc yc w h).
            self.query_embed = nn.Embedding(self.num_queries, 4)           # Reference boxes

            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                )])
            
            self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
            self.objectness = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
            prior_prob = args.prior_prob
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.objectness.layers[2].bias.data = self.objectness.layers[2].bias.data * 0 + bias_value

            # self.class_embed = _get_clones(self.class_embed, args.dec_layers)
            self.bbox_embed = _get_clones(self.bbox_embed, args.dec_layers)
            self.transformer = transformer
            self.transformer.decoder.bbox_embed = self.bbox_embed
        
        # self.text_adapter = None
        # self.image_adapter = None
        # if args.prompt == 'text_adapter':
        #     embed_dim = self.classifier.text_projection.size(1)
        #     self.text_adapter = MLP(embed_dim, embed_dim // 4, embed_dim, 2)
        # elif args.prompt == 'image_adapter':
        #     embed_dim = self.backbone[0].attnpool.positional_embedding.size(1)
        #     self.image_adapter = MLP(embed_dim, embed_dim // 4, embed_dim, 2)

        # ====================================================================================
        #                                   * Clarification *
        #  -----------------------------------------------------------------------------------
        #  Whether self.input_proj contains nn.GroupNorm should not affect performance much.
        #  nn.GroupNorm() is introduced in some of our experiments by accident.
        #  Our experience shows that it even slightly degrades the performance.
        #  We recommend you to simply delete nn.GroupNorm() in self.input_proj for your own
        #  experiments, but if you wish to reproduce our results, please follow our setups below.
        # ====================================================================================


    def forward(self, samples: NestedTensor, categories):
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

        features, pos_embeds = self.backbone(samples)
        text_feature = self.classifier(categories)
        
        outputs = dict(
            features=features,
            text_feature=text_feature,
            tau=self.tau
        )
        
        if self.args.use_proposal:
            srcs = []
            masks = []
            # here only use feature of the last layer
            features = features[-1:]
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
                
            hs, reference = self.transformer(srcs, masks, self.query_embed.weight, pos_embeds)
            
            outputs_coords = []
            outputs_class = []
            text_feature = self.classifier(categories)
            for lvl in range(hs.shape[0]):
                reference_before_sigmoid = inverse_sigmoid(reference[lvl])
                bbox_offset = self.bbox_embed[lvl](hs[lvl])
                outputs_coord = (reference_before_sigmoid + bbox_offset).sigmoid()
                outputs_coords.append(outputs_coord)
                # outputs_class.append(self.class_embed[lvl](hs[lvl]))
                # image_feat = self.feat_projector(hs[lvl])
                # image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
                # similarity = image_feat @ text_feature.t()
                # logits = similarity * self.tau.exp() + self.class_bias
                # outputs_class.append(logits)
            outputs_coords = torch.stack(outputs_coords)
            # outputs_class = torch.stack(outputs_class)
            objectness = self.objectness(hs)
            
            out = {'pred_logits': objectness[-1], 'pred_boxes': outputs_coords[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(objectness, outputs_coords)
            
            outputs.update(out)
        
        return outputs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coords):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coords[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for SAM-DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, iou_modulation=False, iou_with_gt=False):
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
        self.iou_modulation = iou_modulation
        self.iou_with_gt = iou_with_gt

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
        masks = None
        if self.iou_modulation:
            src_boxes = box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'])
            if self.iou_with_gt:
                iou = [box_ops.box_iou(boxes, box_ops.box_cxcywh_to_xyxy(target['boxes']))[0] for boxes, target in zip(src_boxes, targets)]
                masks = []
                for i in range(src_boxes.size(0)):
                    # cat zero to deal with images wo box
                    mask = (torch.cat([iou[i], torch.zeros_like(src_logits[0])], dim=-1).max(dim=-1)[0] > 0.5).float()
                    masks.append(mask)
                masks = torch.stack(masks).unsqueeze(-1)
                num_boxes = (target_classes_onehot * masks).sum()
            else:
                iou = torch.stack([box_ops.box_iou(boxes, boxes)[0] for boxes in src_boxes])
                masks = []
                for i in range(src_boxes.size(0)):
                    gt_idx = idx[1][idx[0] == i]
                    # cat zero to deal with images wo box
                    mask = (torch.cat([iou[i][gt_idx], torch.zeros_like(iou[:1,0])]).max(dim=0)[0] > 0.5).float()
                    masks.append(mask)
                masks = torch.stack(masks).unsqueeze(-1)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2, loss_mask=masks)
        if masks is None:
            loss_ce = loss_ce * src_logits.shape[1]
        else:
            loss_ce = loss_ce * masks.sum(1).mean()
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
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

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    

def contrastive_loss(outputs, targets):
    """ This performs the loss computation.
    Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc
        
            return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

    """

    nnp = [len(target['np_gt']) for target in targets]

    # get all image and span features
    image_feature = outputs['image_feature'].flatten(1,2)
    L, N, C = image_feature.shape
    span_feature = outputs['span_feature']
    image_feature = image_feature.flatten(0,1)
    
    if get_world_size() > 1:
        squeezed = torch.cat([image_feature, span_feature], dim=0)
        
        nfeat = squeezed.size(0)
        nfeats = [None for _ in range(get_world_size())]
        dist.all_gather_object(nfeats, nfeat)
        total_nfeats = sum(nfeats)
        prev_nfeats = sum(nfeats[:get_rank()])
        
        gather_matrix = torch.zeros(total_nfeats, squeezed.size(1), device=squeezed.device, dtype=squeezed.dtype)
        gather_matrix[prev_nfeats:prev_nfeats+nfeat] = squeezed
        dist.all_reduce(gather_matrix)
        all_image_span_features = list(torch.split(gather_matrix, nfeats, dim=0))
        all_image_span_features[get_rank()] = squeezed

        nnps = [None for _ in range(get_world_size())]
        dist.all_gather_object(nnps, nnp)
        
        all_image_features = []
        all_span_features = []
        for feature in all_image_span_features:
            all_image_features.append(feature[:L*N].reshape(L, N, -1))
            all_span_features.append(feature[L*N:])
        outputs['all_span_features'] = all_span_features
        outputs['all_image_features'] = all_image_features
        all_image_features = torch.cat(all_image_features, dim=1)
        all_span_features = torch.cat(all_span_features, dim=0)
    else:
        all_image_features = image_feature.reshape(L, N, -1) # L, bs * N, C
        all_span_features = span_feature # nspans_all, C
        outputs['all_span_features'] = [all_span_features]
        outputs['all_image_features'] = [all_image_features]
        nnps = [nnp]
    
    # only use the last layer's feature
    all_image_features = all_image_features[-1] # bs * N, C
    nnps = [x for y in nnps for x in y]
    all_image_features = all_image_features.view(len(nnps), -1, all_image_features.size(-1)) # bnc
    all_span_features = all_span_features.split(nnps)
    masks = [torch.ones_like(x[:,0]) for x in all_span_features]
    all_span_features = pad_sequence(all_span_features, batch_first=True) # bnc
    masks = pad_sequence(masks, batch_first=True)
    logits = all_image_features.flatten(0,1) @ all_span_features.flatten(0,1).t() * outputs['contra_tau']
    logits = logits.view(*all_image_features.shape[:2], *all_span_features.shape[:2]).permute(0,2,1,3)
    masks = masks.unsqueeze(0)
    logits, assignments = logits.max(dim=2)
    logits = (logits * masks).sum(dim=-1) / masks.sum(dim=-1)
    target = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
    loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2
    losses = dict(
        loss_contrastive=loss
    )

    return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
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
        boxes = boxes * scale_fct[:, None, :]

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
    transformer = None
    if args.use_proposal:
        transformer = build_transformer(args)
    model = FastDETR(args, backbone, transformer, classifier, num_classes=num_classes)
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
    }
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha,
                             losses=losses,
                             iou_modulation=args.iou_modulation,
                             iou_with_gt=args.iou_with_gt)
    criterion.to(device)
    post_processors = {'bbox': PostProcess()}
    if args.masks:
        post_processors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            post_processors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, post_processors
