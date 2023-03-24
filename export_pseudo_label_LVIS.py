# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import json
import os
import io
import argparse
import random
from pathlib import Path
from nltk.corpus import wordnet

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
from PIL import Image
import datasets.transforms as T
from tqdm import tqdm
from torch.utils.data import Dataset


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
    model.eval()

    DETECTION_THRESHOLD = args.det_thr
    dataset = build_dataset(image_set='pseudo_label', args=args)
    print('Loading LVIS meta')
    data = json.load(open(args.lvis_meta_path, 'r'))
    print('Done')

    if args.visualize:
        vslzr = COCOVisualizer()
        
        def visualize(img, orig_size, results, filter, name, idx):
            mask = filter(results)
            vslzr.visualize(img[0], dict(
                boxes=results['boxes'][mask],
                size=orig_size,
                box_label=[f"{results['box_label'][i]}_{results['scores'][i].item():.3f}" for i, p in enumerate(mask) if p],
                # box_label=[f"{results['box_label'][i]}" for i, p in enumerate(mask) if p],
                image_id=idx,
            ), caption=name, savedir=os.path.join(args.output_dir, "vis"), show_in_console=False)
    else:
        annotations = list()

    images = []
    image_counts = dict()
    annotation_counts = dict()
    rare_categories = [k['name'] for k in data['categories'] if k['frequency'] == 'r']

    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader_train = DataLoader(dataset,
                                1,
                                sampler=sampler,
                                drop_last=False,
                                # collate_fn=lambda x: x[0], # batch size is one
                                num_workers=args.num_workers)

    used_cats = [k['name'] for k in data['categories']]
    cat2id = {k['name']: k['id'] for k in data['categories']}
    for i, (img, meta) in tqdm(enumerate(dataloader_train)):


        with torch.no_grad():
            if args.visualize:
                target_sizes = meta['size'].to(device)
            else:
                target_sizes = meta['orig_size'].to(device)
            img = img.to(device)
            outputs = model(img, categories=used_cats)
            _ids = [used_cats.index(cat_name) for cat_name in rare_categories]
            interested_scores = outputs['pred_logits'][:,:,_ids]
            outputs['pred_logits'] = outputs['pred_logits'] - 999
            outputs['pred_logits'][:,:,_ids] = interested_scores
            results = post_processors['bbox'](outputs, target_sizes)[0]
            if args.visualize:
                if results['scores'].numel() == 0:
                    continue
                results['box_label'] = [used_cats[item.item()] for item in results['labels']]
                def score(results):
                    return results['scores'] >= DETECTION_THRESHOLD
                def max_score(results):
                    return results['scores'] == results['scores'].max()
                m = results['scores'].max().item() * 100
                if results['scores'].max().item() < DETECTION_THRESHOLD:
                    continue
                visualize(img, meta['orig_size'][0], results, score, f'{m:.3f}', idx=meta['image_id'].item())
            else:
                selected = results['scores'] >= DETECTION_THRESHOLD
                labels = results['labels'][selected]
                boxes = results['boxes'][selected]
                boxes[:,2:] = boxes[:,2:] - boxes[:,:2]
                scores = results['scores'][selected]
                for label, box, confidence in zip(labels, boxes, scores):
                    annotation = dict(
                        id=-1, # index it after finished labeling
                        image_id=meta['image_id'].item(),
                        bbox=box.tolist(),
                        category_id=cat2id[used_cats[label.item()]],
                        score=confidence.item(),
                    )
                    annotations.append(annotation)
    out = annotations


    with open(f"lvis_rare_image_info_thr{DETECTION_THRESHOLD}_rand{utils.get_rank()}.json", "w") as f:
        json.dump(out, f)

    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
    if utils.is_main_process():
        out_all = out
        for i in range(1, utils.get_world_size()):
            with open(f"lvis_rare_image_info_thr{DETECTION_THRESHOLD}_rand{i}.json", "r") as f:
                part_annotations = json.load(f)
            out_all.extend(part_annotations)
        with open(f"lvis_rare_image_info_thr{DETECTION_THRESHOLD}.json", "w") as f:
            json.dump(out_all, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CORA", parents=[get_args_parser()])

    parser.add_argument('--lvis_meta_path', default='datasets/lvis/lvis_v1_val.json')
    parser.add_argument('--petrel_cfg', default='petreloss.config')
    parser.add_argument('--det_thr', default=0.2, type=float)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
