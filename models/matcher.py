# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import enum
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from detectron2.layers import batched_nms


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, score_threshold=2.0):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.score_threshold = score_threshold
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        obj_key = "pred_logits"
        box_key = "pred_boxes"
        bs, num_queries = outputs[obj_key].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs[obj_key].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs[box_key].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]

        ignore = []
        for score, box, target in zip(outputs['pred_logits'], outputs['pred_boxes'], targets):
            # filter by threshold
            score = score[:,0]
            mask = score.sigmoid() > self.score_threshold
            ignore_idx = mask.nonzero()[:,0]
            score = score[mask]
            box = box[mask]
            keep = batched_nms(box_cxcywh_to_xyxy(box), score, torch.zeros_like(score), 0.5)
            ignore.append(ignore_idx[keep])
            

        if 'proposal_classes' in outputs:
            ori_tgt_ids = torch.cat([v["ori_labels"] for v in targets])
            batch_idx = torch.cat([torch.zeros_like(v["ori_labels"]) + i for i, v in enumerate(targets)])
            batched_ori_tgt_ids = torch.zeros_like(ori_tgt_ids).unsqueeze(0).repeat((len(targets), 1)) - 1
            batched_ori_tgt_ids.scatter_(0, batch_idx.unsqueeze(0), ori_tgt_ids.unsqueeze(0))

            if 'semantic_cost' not in targets[0] and 'matching_threshold' not in targets[0]:
                if 'class_group' in target:
                    class_group = torch.tensor(target['class_group'], device=batched_ori_tgt_ids.device).unsqueeze(0).expand(batched_ori_tgt_ids.size(0), len(target['class_group']))
                    out = torch.gather(class_group, dim=-1, index=outputs['proposal_classes'])
                    invalid = batched_ori_tgt_ids < 0
                    batched_ori_tgt_ids[invalid] = 0
                    tgt = torch.gather(class_group, dim=-1, index=batched_ori_tgt_ids)
                    tgt[invalid] = -1
                    valid_mask = out.unsqueeze(-1) == tgt.unsqueeze(1)
                else:
                    # classified correctly
                    if outputs['proposal_classes'].dim() == 3:
                        batched_ori_tgt_ids = batched_ori_tgt_ids.unsqueeze(1)
                        valid_mask = (outputs['proposal_classes'].unsqueeze(-1) == batched_ori_tgt_ids.unsqueeze(1)).any(-2)
                    else:
                        valid_mask = outputs['proposal_classes'].unsqueeze(-1) == batched_ori_tgt_ids.unsqueeze(1)
                giou = -cost_giou.view(bs, num_queries, -1)
                # only consider the correctly classified subset
                giou[~valid_mask] = -1
                # any one of the correctly classified subset has giou > 0 is a valid box
                valid_box = (giou > 0).any(1)
                # remove the invalid box
                valid_mask[~valid_box.unsqueeze(1).expand_as(valid_mask)] = False
                C[~valid_mask] = 99
            else:
                invalid = batched_ori_tgt_ids < 0
                batched_ori_tgt_ids[invalid] = 0
                nc, dim = outputs['text_feature'].shape
                proposal_feature = torch.gather(outputs['text_feature'].unsqueeze(0).expand(len(targets), nc, dim), dim=1, \
                    index=outputs['proposal_classes'].unsqueeze(-1).expand(len(targets), outputs['proposal_classes'].size(1), dim))
                target_feature = torch.gather(outputs['text_feature'].unsqueeze(0).expand(len(targets), nc, dim), dim=1, \
                    index=batched_ori_tgt_ids.unsqueeze(-1).expand(len(targets), batched_ori_tgt_ids.size(1), dim))
                if 'matching_threshold' in targets[0]:
                    valid_mask = (1 - torch.bmm(proposal_feature, target_feature.permute(0,2,1))) < targets[0]['matching_threshold']
                    giou = -cost_giou.view(bs, num_queries, -1)
                    # only consider the correctly classified subset
                    giou[~valid_mask] = -1
                    # any one of the correctly classified subset has giou > 0 is a valid box
                    valid_box = (giou > 0).any(1)
                    # remove the invalid box
                    valid_mask[~valid_box.unsqueeze(1).expand_as(valid_mask)] = False
                    C[~valid_mask] = 99
                else:
                    semantic_cost = ((1 - torch.bmm(proposal_feature, target_feature.permute(0,2,1))) * targets[0]['semantic_cost']).exp() - 1
                    C = C + semantic_cost
            
        C = C.cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        C = [c[i] for i, c in enumerate(C.split(sizes, -1))]
        new_indices = []
        for i, c in enumerate(C):
            mask = (c[indices[i]] < 99).numpy()
            new_indices.append((indices[i][0][mask], indices[i][1][mask]))
        indices = new_indices

        new_ignore = []
        for ig, ind, box in zip(ignore, indices, outputs['pred_boxes']):
            if ig.numel() == 0 or len(ind[0]) == 0:
                new_ignore.append(ig)
            else:
                mask = box_iou(box_cxcywh_to_xyxy(box[ig]), box_cxcywh_to_xyxy(box[ind[0]]))[0].max(-1)[0] < 0
                new_ignore.append(ig[mask])
        outputs['ignore'] = new_ignore

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, score_threshold=args.score_threshold)
