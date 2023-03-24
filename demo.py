# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import argparse
import random
from pathlib import Path

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

    # def match_keywords(n, name_keywords):
    #     out = False
    #     for b in name_keywords:
    #         if b in n:
    #             out = True
    #             break
    #     return out

    # param_dicts = [
    #     {
    #         "params":
    #             [p for n, p in model_without_ddp.named_parameters()
    #              if "backbone.0" not in n and not match_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
    #         "lr": args.lr,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters()
    #                    if "backbone.0" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters()
    #                    if match_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
    #         "lr": args.lr * args.lr_linear_proj_mult,
    #     }
    # ]

    # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # if args.frozen_weights is not None:
    #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    #     model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model_ema'])
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     args.start_epoch = checkpoint['epoch'] + 1

    DETECTION_THRESHOLD = 0.2
    # inference_dir = "./images/"
    # image_dirs = os.listdir(inference_dir)
    # image_dirs = [filename for filename in image_dirs if filename.endswith(".jpg") and 'det_res' not in filename]
    dataset_val = build_dataset(image_set='val', args=args)
    if hasattr(dataset_val, 'coco'):
        cocojs = dataset_val.coco.dataset
        id2name = {item['id']: item['name'] for item in cocojs['categories']}
    model.eval()
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
        print("here")
    
    with torch.no_grad():
        for idx in range(50):
            img, targets = dataset_val[idx]
            w, h = img.shape[-2:]
            # w, h = targets['orig_size']
            target_sizes = torch.tensor([[w, h]], device=device)
            img = img.to(device)
            img = img.unsqueeze(0)   # adding batch dimension
            outputs = model(img, categories=dataset_val.category_list)
            results = post_processors['bbox'](outputs, target_sizes)[0]
            results['box_label'] = [id2name[dataset_val.label2catid[int(item)]] for item in results['labels']]
            def score(results):
                return results['scores'] >= DETECTION_THRESHOLD
            def target_class(results):
                target_catids = [28, 21, 47, 6, 76, 41, 18, 63, 32, 36, 81, 22, 61, 87, 5, 17, 49]
                ret = []
                for label in results['labels']:
                    ret.append(dataset_val.label2catid[label.item()] in target_catids)
                return ret
            visualize(img, targets, results, score, 'threshold0.2')
            # visualize(img, targets, results, target_class, 'all_target')
            # indexes = results['scores'] >= DETECTION_THRESHOLD
            # scores = results['scores'][indexes]
            # labels = results['labels'][indexes]
            # boxes = results['boxes'][indexes]


            # Visualize the detection results
            # import cv2
            # img_det_result = cv2.imread(os.path.join(inference_dir, image_dir))
            # for i in range(scores.shape[0]):
            #     x1, y1, x2, y2 = round(float(boxes[i, 0])), round(float(boxes[i, 1])), round(float(boxes[i, 2])), round(float(boxes[i, 3]))
            #     img_det_result = cv2.rectangle(img_det_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.imwrite(os.path.join(inference_dir, "det_res_" + image_dir), img_det_result)
            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CORA", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
