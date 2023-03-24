# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
from util import box_ops

import util.misc as utils
from datasets.coco_eval import CocoEvaluator, convert_to_xywh
from datasets.panoptic_eval import PanopticEvaluator
from util.visualizer import COCOVisualizer
from torch.nn.functional import interpolate
from copy import deepcopy
import random
import pycocotools.mask as mask_util
import numpy as np

def generate_deterministic_rand(num):
    prev_state = random.getstate()
    random.seed(num)
    rand = random.random()
    random.setstate(prev_state)
    return rand

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0,
                    args=None,
                    model_ema=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if args.debug or utils.get_world_size() == 1:
        print_freq = 10
    else:
        print_freq = 100

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v if isinstance(v, (list, dict)) else v.to(device) for k, v in t.items()} for t in targets]

        categories = data_loader.dataset.category_list

        # add pseudo labels
        pseudo_categories = list(set([a for target in targets if 'pseudo_labels' in target for a in target['pseudo_labels']]))
        for target in targets:
            if 'pseudo_labels' not in target:
                continue
            pseudo_label_ids = [pseudo_categories.index(cat) + len(categories) for cat in target['pseudo_labels']]
            target['labels'] = torch.cat([target['labels'], torch.tensor(pseudo_label_ids, device=target['labels'].device, dtype=target['labels'].dtype)])

        if args.class_oracle:
            gt_classes = [target['labels'] for target in targets]
        else:
            gt_classes = None

        class_group = None
        if args.num_label_sampled > 0:
            assert len(pseudo_categories) == 0
            gt = torch.cat([target['labels'] for target in targets]).unique()
            if gt.numel() >= args.num_label_sampled:
                sampled = gt[torch.randperm(gt.numel(), device=gt.device)][:args.num_label_sampled]
            else:
                all_class = torch.arange(len(categories), device=gt.device)
                neg_class = all_class[~(all_class.unsqueeze(1) == gt.unsqueeze(0)).any(-1)]
                num_sample = args.num_label_sampled - gt.numel()
                sampled = neg_class[torch.randperm(neg_class.numel(), device=gt.device)][:num_sample]
                sampled = torch.cat([gt, sampled])
            used_categories = [categories[i] for i in sampled.tolist()]
            # reorder
            for target in targets:
                label = target['labels']
                sampled_mask = (label.unsqueeze(-1) == sampled.unsqueeze(0)).any(-1)
                target['boxes'] = target['boxes'][sampled_mask]
                label = label[sampled_mask]
                new_label = (label.unsqueeze(-1) == sampled.unsqueeze(0)).int().argmax(-1)
                target['labels'] = new_label
            if hasattr(data_loader.dataset, 'class_group') and data_loader.dataset.class_group is not None:
                assert data_loader.dataset.class_group is None
        else:
            used_categories = categories + pseudo_categories
            if hasattr(data_loader.dataset, 'class_group') and data_loader.dataset.class_group is not None:
                assert args.semantic_cost < 0
                class_group = data_loader.dataset.class_group
                for target in targets:
                    target['class_group'] = class_group
            elif args.semantic_cost > 0:
                for target in targets:
                    target['semantic_cost'] = args.semantic_cost
            if args.matching_threshold >= 0.:
                for target in targets:
                    target['matching_threshold'] = args.matching_threshold
                

        split_class = generate_deterministic_rand(_cnt) < args.split_class_p
        if split_class:
            targets = [*deepcopy(targets), *deepcopy(targets)]
        outputs = model(samples, categories=used_categories, gt_classes=gt_classes, split_class=split_class, targets=targets)

        # hard code for class agnostic training
        if not args.end2end:
            for target in targets:
                target['ori_labels'] = target['labels']
                target['labels'] = target['labels'] - target['labels']
            
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}.\n  Training terminated.".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        del samples
        del targets
        del outputs
        del loss_dict
        del loss_dict_reduced
        del loss_dict_reduced_unscaled
        del weight_dict
        del losses
        del losses_reduced_scaled
        
        _cnt += 1
        if args.debug:
            if _cnt % (15 * 4) == 0:
                print("BREAK!"*5)
                break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    if args.dataset_file == 'lvis':
        from lvis import LVISEval, LVISResults
        cat2label = data_loader.dataset.cat2label
        label2cat = {v: k for k, v in cat2label.items()}
        panoptic_evaluator = None
        coco_evaluator = None
        lvis_results = []
        label_map = args.label_map
        iou_types = ['bbox']
    else:
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        coco_evaluator = CocoEvaluator(base_ds, iou_types, label2cat=data_loader.dataset.label2catid)

        panoptic_evaluator = None
        if 'panoptic' in postprocessors.keys():
            panoptic_evaluator = PanopticEvaluator(
                data_loader.dataset.ann_file,
                data_loader.dataset.ann_folder,
                output_dir=os.path.join(output_dir, "panoptic_eval"),
            )

    if args.debug or utils.get_world_size() == 1:
        print_freq = 10
    else:
        print_freq = 100
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v if isinstance(v, (list, dict)) else v.to(device) for k, v in t.items()} for t in targets]

        if args.class_oracle:
            gt_classes = [target['labels'] for target in targets]
        else:
            gt_classes = None

        if not args.eval_gt:
            outputs = model(samples, categories=data_loader.dataset.category_list, gt_classes=gt_classes, targets=targets)
            # for loss only
            training_target = []
            for target in targets:
                new_target = target.copy()
                new_target['ori_labels'] = target['labels']
                new_target['labels'] = target['labels'] - target['labels']
                training_target.append(new_target)
            loss_dict = criterion(outputs, training_target)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
        else:
            # to make metric logger happy
            metric_logger.update(loss=0, class_error=0)
            results = []
            for target in targets:
                out = {}
                out['scores'] = torch.ones_like(target['labels'])
                out['labels'] = target['labels']
                gt_boxes = box_ops.box_cxcywh_to_xyxy(target['boxes'])
                h, w = target['orig_size']
                scaler = torch.tensor([w, h, w, h], device=gt_boxes.device).unsqueeze(0)
                out['boxes'] = gt_boxes * scaler
                results.append(out)

        if args.dataset_file == 'lvis':
            for target, output in zip(targets, results):
                image_id = target["image_id"].item()

                if "masks" in output.keys():
                    masks = output["masks"].data.cpu().numpy()
                    masks = masks > 0.5
                    rles = [
                        mask_util.encode(
                            np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                        )[0]
                        for mask in masks
                    ]
                    for rle in rles:
                        rle["counts"] = rle["counts"].decode("utf-8")

                boxes = convert_to_xywh(output["boxes"])
                for ind in range(len(output["scores"])):
                    temp = {
                        "image_id": image_id,
                        "score": output["scores"][ind].item(),
                        "category_id": output["labels"][ind].item(),
                        "bbox": boxes[ind].tolist(),
                    }
                    if label_map:
                        temp["category_id"] = label2cat[temp["category_id"]]
                    if "masks" in output.keys():
                        temp["segmentation"] = rles[ind]

                    lvis_results.append(temp)
        else:
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)

        if args.visualize:
            visualizer = COCOVisualizer()
            img_h, img_w = orig_target_sizes[0].unbind(0)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0).unsqueeze(0)
            boxes = [box * fct[None] for box, fct in zip(box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'][0]).unsqueeze(0), scale_fct)][0]
            scores = outputs['pred_logits'].sigmoid().squeeze()
            def visualize(img, targets, results, filter=None, name='debug'):
                if filter is not None:
                    mask = filter(results)
                else:
                    mask = None
                visualizer.visualize(img[0], dict(
                    boxes=results['boxes'][mask] if mask is not None else results['boxes'],
                    size=targets['orig_size'],
                    box_label=[f"{results['scores'][i].item():.2f}" for i, p in (enumerate(mask if mask is not None else results['scores'])) if p],
                    image_id=_cnt,
                ), caption=name, savedir=os.path.join(args.output_dir, "vis0.3_15e"), show_in_console=False)
            new_results = dict(
                scores=scores,
                boxes=boxes,
                ignore=outputs['ignore'],
            )
            def score(results):
                return results['scores'] >= 0.3
            def topk_score(results):
                return results['scores'] >= results['scores'].topk(k=20)[0][-1]
            def topk_cost(results):
                return results['cost'] >= results['cost'].topk(k=20)[0][-1]
            def ignore(results):
                ret = torch.zeros_like(results['scores'], dtype=torch.bool)
                ret[results['ignore']] = True
                return ret
            visualize(interpolate(samples.decompose()[0], orig_target_sizes[0].tolist()), targets[0], new_results, filter=score, name='novel_0.2')
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
            panoptic_evaluator.update(res_pano)
            
        _cnt += 1
        if args.debug:
            if _cnt % (15 * 5) == 0:
                print("BREAK!"*5)
                break

    if args.dataset_file == 'lvis':
        rank = utils.get_rank()
        torch.save(lvis_results, output_dir + f"/pred_{rank}.pth")

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if rank == 0:
            world_size = utils.get_world_size()
            for i in range(1, world_size):
                temp = torch.load(output_dir + f"/pred_{i}.pth")
                lvis_results += temp

            lvis_results = LVISResults(base_ds, lvis_results, max_dets=300)
            for iou_type in iou_types:
                lvis_eval = LVISEval(base_ds, lvis_results, iou_type)
                lvis_eval.run()
                lvis_eval.print_results()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        return stats, lvis_eval
    else:
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()
        if panoptic_evaluator is not None:
            panoptic_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
        panoptic_res = None
        if panoptic_evaluator is not None:
            panoptic_res = panoptic_evaluator.summarize()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if coco_evaluator is not None:
            if 'bbox' in postprocessors.keys():
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in postprocessors.keys():
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        if panoptic_res is not None:
            stats['PQ_all'] = panoptic_res["All"]
            stats['PQ_th'] = panoptic_res["Things"]
            stats['PQ_st'] = panoptic_res["Stuff"]
        
        del samples
        del targets
        del outputs
        del loss_dict
        del loss_dict_reduced
        del loss_dict_reduced_unscaled
        del weight_dict

        torch.cuda.empty_cache()

        return stats, coco_evaluator


@torch.no_grad()
def lvis_evaluate(
    model, criterion, postprocessors, data_loader, base_ds, device, output_dir, label_map, amp, args
):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    lvis_results = []

    cat2label = data_loader.dataset.cat2label
    label2cat = {v: k for k, v in cat2label.items()}

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(enabled=amp, device_type='cuda'):
            outputs = model(samples, categories=data_loader.dataset.category_list)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        if "segm" in postprocessors.keys():
            results, topk_boxes = postprocessors["bbox"](outputs, orig_target_sizes)
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            outputs_masks = outputs["pred_masks"].squeeze(2)

            bs = len(topk_boxes)
            outputs_masks_new = [[] for _ in range(bs)]
            for b in range(bs):
                for index in topk_boxes[b]:
                    outputs_masks_new[b].append(outputs_masks[b : b + 1, index : index + 1, :, :])
            for b in range(bs):
                outputs_masks_new[b] = torch.cat(outputs_masks_new[b], 1)
            outputs["pred_masks"] = torch.cat(outputs_masks_new, 0)

            results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)
        else:
            results = postprocessors["bbox"](outputs, orig_target_sizes)

        for target, output in zip(targets, results):
            image_id = target["image_id"].item()

            if "masks" in output.keys():
                masks = output["masks"].data.cpu().numpy()
                masks = masks > 0.5
                rles = [
                    mask_util.encode(
                        np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                    )[0]
                    for mask in masks
                ]
                for rle in rles:
                    rle["counts"] = rle["counts"].decode("utf-8")

            boxes = convert_to_xywh(output["boxes"])
            for ind in range(len(output["scores"])):
                temp = {
                    "image_id": image_id,
                    "score": output["scores"][ind].item(),
                    "category_id": output["labels"][ind].item(),
                    "bbox": boxes[ind].tolist(),
                }
                if label_map:
                    temp["category_id"] = label2cat[temp["category_id"]]
                if "masks" in output.keys():
                    temp["segmentation"] = rles[ind]

                lvis_results.append(temp)

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    rank = utils.get_rank()
    torch.save(lvis_results, output_dir + f"/pred_{rank}.pth")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        world_size = utils.get_world_size()
        for i in range(1, world_size):
            temp = torch.load(output_dir + f"/pred_{i}.pth")
            lvis_results += temp

        from lvis import LVISEval, LVISResults

        lvis_results = LVISResults(base_ds, lvis_results, max_dets=300)
        for iou_type in iou_types:
            lvis_eval = LVISEval(base_ds, lvis_results, iou_type)
            lvis_eval.run()
            lvis_eval.print_results()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if rank == 0:
        stats.update(lvis_eval.get_results())
    return stats, None