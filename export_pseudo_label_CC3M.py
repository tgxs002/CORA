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
try:
    from petrel_client.client import Client
except ImportError as E:
    "petrel_client.client cannot be imported"
    pass
from tqdm import tqdm
from torch.utils.data import Dataset

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff).convert("RGB")


class TCSLoader(object):

    def __init__(self, conf_path):
        self.client = Client(conf_path)

    def __call__(self, fn):
        try:
            img_value_str = self.client.get(fn)
            img = pil_loader(img_value_str)
        except:
            print('Read image failed ({})'.format(fn))
            return None
        else:
            return img

class ceph_cc3m_dataset(Dataset):
    def __init__(self, tcsloader, captions, categories, args):
        self.loader = tcsloader
        self.captions = captions
        self.dataset_path = args.cc3m_path
        self.categories = categories

        if 'clip' in args.backbone:
            MEAN = [0.48145466, 0.4578275, 0.40821073]
            STD = [0.26862954, 0.26130258, 0.27577711]
        else:
            MEAN = [0.485, 0.456, 0.406]
            STD = [0.229, 0.224, 0.225]

        normalize = T.Compose([
            T.ToRGB(),
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])

        self.transforms = T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
        

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        data = json.loads(self.captions[idx])
        image_path = os.path.join(self.dataset_path, data['image'].replace('.zip@', ''))
        image = self.loader(image_path)
        w, h = image.size
        lemmas = ' '.join(data['spacy_lemmas'])
        tags = [k for k in self.categories if ' ' + k + ' ' in ' ' + lemmas + ' ']
        meta = dict(
            file_path=image_path,
            orig_size=torch.tensor([h, w]),
            image_id=idx,
            categories=tags,
        )
        image, _ = self.transforms(image, target={})
        return image, meta

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
    captions = open(args.caption_path, 'r').readlines()
    reader = TCSLoader(args.petrel_cfg)


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
    categories = open(args.noun_path, 'r').readlines()
    categories = [k.split(',')[0] for k in categories]

    dataset = ceph_cc3m_dataset(reader, captions, categories, args)

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

    for i, (img, meta) in tqdm(enumerate(dataloader_train)):
        file_path = meta['file_path'][0]
        h, w = meta['orig_size'][0].tolist()

        current_categories = [k[0] for k in meta['categories']]
        if len(meta['categories']) == 0:
            continue

        with torch.no_grad():
            target_sizes = torch.tensor([img.shape[-2:] if args.visualize else [h, w]], device=device)
            img = img.to(device)
            
            used_cats = random.sample(categories, k=48 if args.num_label_sampled < 0 else args.num_label_sampled)
            used_cats = list(set(used_cats + current_categories))
            outputs = model(img, categories=used_cats)
            _ids = [used_cats.index(cat_name) for cat_name in current_categories]
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
                        category=used_cats[label.item()],
                        score=confidence.item(),
                    )
                    annotations.append(annotation)

        file_name = "/".join(file_path.split('/')[-2:])
        image = {
            'id': meta['image_id'].item(),
            'file_name': file_name,
            'pos_category_ids': list(set([used_cats[item.item()] for item in labels])),
            'width': w,
            'height': h
        }
        images.append(image)

    print('# Images', len(images))
    out = {'images': images, 'annotations': annotations}


    with open(f"cc3m_{args.name}_image_info_thr{DETECTION_THRESHOLD}_rand{utils.get_rank()}.json", "w") as f:
        json.dump(out, f)

    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
    if utils.is_main_process():
        out_all = out
        for i in range(1, utils.get_world_size()):
            with open(f"cc3m_{args.name}_image_info_thr{DETECTION_THRESHOLD}_rand{i}.json", "r") as f:
                part_annotations = json.load(f)
            out_all['annotations'].extend(part_annotations['annotations'])
            out_all['images'].extend(part_annotations['images'])

        for i in range(len(out_all['annotations'])):
            out_all['annotations'][i]['id'] = i
        with open(f"cc3m_{args.name}_image_info_thr{DETECTION_THRESHOLD}.json", "w") as f:
            json.dump(out_all, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CORA", parents=[get_args_parser()])
    parser.add_argument('--cc3m_path', type=str)
    parser.add_argument('--noun_path', default='')
    parser.add_argument('--caption_path', type=str)
    parser.add_argument('--petrel_cfg', default='petreloss.config')
    parser.add_argument('--det_thr', default=0.3, type=float)
    parser.add_argument('--name', default='', type=str)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
