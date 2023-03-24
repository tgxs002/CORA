# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import json
import os
import argparse
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from util.visualizer import COCOVisualizer
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco import make_coco_transforms
from models import build_model
from main import get_args_parser

def main(args):

    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only."
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, post_processors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of params in model: ', n_parameters)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model_ema'])

    DETECTION_THRESHOLD = args.det_thr
    dataset_train = build_dataset(image_set='train', args=args)
    if args.distributed:
        sampler = DistributedSampler(dataset_train, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset_train)

    dataloader_train = DataLoader(dataset_train,
                                 1,
                                 sampler=sampler,
                                 drop_last=False,
                                 collate_fn=lambda x: x[0], # batch size is one
                                 num_workers=args.num_workers)
    model.eval()

    if args.visualize:
        vslzr = COCOVisualizer()
        
        def visualize(img, targets, results, filter, name):
            mask = filter(results)
            vslzr.visualize(img[0], dict(
                boxes=results['boxes'][mask],
                size=targets['orig_size'],
                box_label=[f"{results['box_label'][i]}_{results['scores'][i].item():.3f}" for i, p in enumerate(mask) if p],
                # box_label=[f"{results['box_label'][i]}" for i, p in enumerate(mask) if p],
                image_id=idx,
            ), caption=name, savedir=os.path.join(args.output_dir, "vis"), show_in_console=False)
    else:
        annotations = list()
    
    with torch.no_grad():
        for idx, (img, targets) in tqdm(enumerate(dataloader_train)):
            w, h = img.shape[-2:] if args.visualize else targets['orig_size']
            target_sizes = torch.tensor([[w, h]], device=device)
            img = img.to(device)
            img = img.unsqueeze(0)   # adding batch dimension
            outputs = model(img, categories=targets['class_labels'])
            results = post_processors['bbox'](outputs, target_sizes)[0]
            if args.visualize:
                results['box_label'] = [targets['class_labels'][int(item)] for item in results['labels']]
                def score(results):
                    return results['scores'] >= DETECTION_THRESHOLD
                visualize(img, targets, results, score, 'threshold0.2')
            else:
                selected = results['scores'] >= DETECTION_THRESHOLD
                labels = results['labels'][selected]
                boxes = results['boxes'][selected]
                boxes[:,2:] = boxes[:,2:] - boxes[:,:2]
                scores = results['scores'][selected]
                for label, box, confidence in zip(labels, boxes, scores):
                    annotation = dict(
                        image_id=targets['image_id'].item(),
                        bbox=box.tolist(),
                        class_label=targets['class_labels'][label],
                        score=confidence.item(),
                    )
                    annotations.append(annotation)
        with open(f"pseudo_label_thr{DETECTION_THRESHOLD}_rand{utils.get_rank()}.json", "w") as f:
            json.dump(annotations, f)

    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
    if utils.is_main_process():
        annotations_all = list()
        for i in range(utils.get_world_size()):
            with open(f"pseudo_label_thr{DETECTION_THRESHOLD}_rand{i}.json", "r") as f:
                part_annotations = json.load(f)
            annotations_all.extend(part_annotations)
        with open(f"pseudo_label_thr{DETECTION_THRESHOLD}.json", "w") as f:
            json.dump(annotations_all, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CORA", parents=[get_args_parser()])
    parser.add_argument('--cls_annotation_path', required=True, type=str)
    parser.add_argument('--det_thr', default=0.2, type=float)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
