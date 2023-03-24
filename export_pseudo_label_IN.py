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

class ceph_dataset(Dataset):
    def __init__(self, tcsloader, file_list, args):
        self.loader = tcsloader
        self.files = file_list

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
        return len(self.files)

    def __getitem__(self, idx):
        image = self.loader(self.files[idx])
        w, h = image.size
        meta = dict(
            file_path=self.files[idx],
            orig_size=torch.tensor([h, w]),
            image_id=idx,
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
    # dataset_train = build_dataset(image_set='train', args=args)
    print('Loading LVIS meta')
    data = json.load(open(args.lvis_meta_path, 'r'))
    print('Done')
    synset2cat = {x['synset']: x for x in data['categories']}
    # folders = sorted(os.listdir(args.imagenet_path))
    # only the folder names of overlapping classes
    with open(os.path.join(args.meta_path, "imagenet_lvis_wnid.txt"), 'r') as f:
        folders = sorted(f.readlines())
        folders = [k.strip() for k in folders]
    with open(os.path.join(args.meta_path, "all_images.txt"), 'r') as f:
        all_images = [k.strip() for k in f.readlines()]
    assert args.imagenet_path.startswith("s3"), "only support ceph dataloading"
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
    no_rare_categories = [k['name'] for k in data['categories'] if k['frequency'] != 'r']

    full_pathes = ['{}/{}/{}'.format(args.imagenet_path, file[:file.find('_')], file) for file in all_images]
    dataset = ceph_dataset(reader, full_pathes, args)

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
        
        synset = wordnet.synset_from_pos_and_offset('n', int(file_path.split('/')[-2][1:])).name()
        cat = synset2cat[synset]
        cat_id = cat['id']
        cat_name = cat['name']

        with torch.no_grad():
            target_sizes = torch.tensor([img.shape[-2:] if args.visualize else [h, w]], device=device)
            img = img.to(device)
            used_cats = random.sample(no_rare_categories, k=args.num_label_sampled) if args.num_label_sampled > 0 else no_rare_categories
            used_cats = used_cats + ([cat_name] if cat_name not in used_cats else [])
            outputs = model(img, categories=used_cats)
            _id = used_cats.index(cat_name)
            interested_scores = outputs['pred_logits'][:,:,_id]
            outputs['pred_logits'] = outputs['pred_logits'] - 999
            outputs['pred_logits'][:,:,_id] = interested_scores
            results = post_processors['bbox'](outputs, target_sizes)[0]
            if args.visualize:
                if results['scores'].numel() == 0:
                    continue
                results['box_label'] = [cat_name for item in results['labels']]
                def score(results):
                    return results['scores'] >= DETECTION_THRESHOLD
                def max_score(results):
                    return results['scores'] == results['scores'].max()
                m = results['scores'].max().item() * 100
                visualize(img, meta['orig_size'][0], results, score, f'{m:.3f}_{cat_name}', idx=meta['image_id'].item())
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
                        category_id=cat_id, # does not align with lvis
                        score=confidence.item(),
                    )
                    annotations.append(annotation)
                    if cat_id not in annotation_counts:
                        annotation_counts[cat_id] = 0
                    annotation_counts[cat_id] += 1

        file_name = "/".join(file_path.split('/')[-2:])
        image = {
            'id': meta['image_id'].item(),
            'file_name': file_name,
            'pos_category_ids': [cat_id],
            'width': w,
            'height': h
        }
        images.append(image)
        if cat_id not in image_counts:
            image_counts[cat_id] = 0
        image_counts[cat_id] += 1

    print('# Images', len(images))
    for x in data['categories']:
        x['image_count'] = image_counts[x['id']] if x['id'] in image_counts else 0
        x['instance_count'] = annotation_counts[x['id']] if x['id'] in annotation_counts else 0
    out = {'categories': data['categories'], 'images': images, 'annotations': annotations}


    with open(f"imagenet_lvis_image_info_thr{DETECTION_THRESHOLD}_rand{utils.get_rank()}.json", "w") as f:
        json.dump(out, f)

    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
    if utils.is_main_process():
        out_all = out
        for i in range(1, utils.get_world_size()):
            with open(f"imagenet_lvis_image_info_thr{DETECTION_THRESHOLD}_rand{i}.json", "r") as f:
                part_annotations = json.load(f)
            for cat, new_cat in zip(out_all['categories'], part_annotations['categories']):
                cat['image_count'] += new_cat['image_count']
                cat['instance_count'] += new_cat['instance_count']
            out_all['annotations'].extend(part_annotations['annotations'])
            out_all['images'].extend(part_annotations['images'])

        for i in range(len(out_all['annotations'])):
            out_all['annotations'][i]['id'] = i
        with open(f"imagenet_lvis_image_info_thr{DETECTION_THRESHOLD}.json", "w") as f:
            json.dump(out_all, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CORA", parents=[get_args_parser()])
    parser.add_argument('--imagenet_path', default='datasets/imagenet/ImageNet-LVIS')
    parser.add_argument('--meta_path', default='')
    parser.add_argument('--lvis_meta_path', default='datasets/lvis/lvis_v1_val.json')
    parser.add_argument('--petrel_cfg', default='petreloss.config')
    parser.add_argument('--det_thr', default=0.2, type=float)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
