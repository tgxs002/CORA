# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.
"""
import math
from pathlib import Path

import torch
import json
import os
import random
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from util import box_ops

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, use_caption, pseudo_box, pseudo_threshold, no_overlapping_pseudo, \
        base_sample_prob, repeat_factor_sampling, repeat_threshold, remove_base_pseudo_label):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms        
        
        self.all_categories = {k['id']: k['name'] for k in self.coco.dataset['categories']}
        self.category_list = [self.all_categories[k] for k in sorted(self.all_categories.keys())]
        self.category_ids = {v: k for k, v in self.all_categories.items()}
        self.label2catid = {k: self.category_ids[v] for k, v in enumerate(self.category_list)}
        self.catid2label = {v: k for k, v in self.label2catid.items()}
        self.use_caption = use_caption
        self.no_overlapping_pseudo = no_overlapping_pseudo
        self.remove_base_pseudo_label = remove_base_pseudo_label
        self.base_sample_prob = base_sample_prob
        
        if use_caption:
            if "train2017" in str(ann_file):
                caption_file = "captions_train2017.json"
            else:
                caption_file = "captions_val2017.json"
            caption_file = os.path.join(os.path.dirname(str(ann_file)), caption_file)
            span_annotation = f"span/span_{os.path.basename(caption_file)}"
            with open(span_annotation, 'r') as f:
                spans = json.load(f)
            with open(caption_file, 'r') as f_:
                captions = json.load(f_)['annotations']
                self.captions = dict()
                for caption in captions:
                    if caption['image_id'] not in self.captions:
                        self.captions[caption['image_id']] = list()
                    if len(spans[str(caption['id'])]) > 0:
                        self.captions[caption['image_id']].append(dict(
                            caption=caption['caption'],
                            spans=spans[str(caption['id'])]
                        ))

        self.use_pseudo_box = pseudo_box != ""
        if self.use_pseudo_box:
            with open(pseudo_box, 'r') as f:
                pseudo_annotations = json.load(f)
            self.pseudo_annotations = dict()
            for annotation in pseudo_annotations:
                if annotation['image_id'] not in self.pseudo_annotations:
                    self.pseudo_annotations[annotation['image_id']] = []
                if annotation['score'] > pseudo_threshold:
                    self.pseudo_annotations[annotation['image_id']].append(annotation)

        if repeat_factor_sampling:
            # 1. For each category c, compute the fraction of images that contain it: f(c)
            counter = {k: 1e-3 for k in self.category_ids.values()}
            for id in self.ids:
                target = self.coco.imgToAnns[id]
                cats = [t['category_id'] for t in target]
                cats = set(cats)
                for c in cats:
                    counter[c] += 1
            num_images = len(self.ids)
            for k, v in counter.items():
                counter[k] = v / num_images
            # 2. For each category c, compute the category-level repeat factor:
            #    r(c) = max(1, sqrt(t / f(c)))
            category_rep = {
                cat_id: max(1.0, math.sqrt(repeat_threshold / cat_freq))
                for cat_id, cat_freq in counter.items()
            }
            # 3. For each image I, compute the image-level repeat factor:
            #    r(I) = max_{c in I} r(c)
            rep_factors = []
            for id in self.ids:
                target = self.coco.imgToAnns[id]
                cats = [t['category_id'] for t in target]
                cats = set(cats)
                rep_factor = max({category_rep[cat_id] for cat_id in cats}, default=1.0)
                rep_factors.append(rep_factor)
            self.rep_factors = rep_factors
        
        self.prepare = ConvertCocoPolysToMask(return_masks, map=self.catid2label)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]

        if self.base_sample_prob < 1.0:
            new_target = []
            for t in target:
                if random.random() < self.base_sample_prob:
                    new_target.append(t)
            target = new_target
        if self.use_pseudo_box:
            pseudo_annotations = self.pseudo_annotations[image_id] if image_id in self.pseudo_annotations else []
            target.extend(pseudo_annotations)
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        if self.use_pseudo_box:
            # recover the pseudo boxes
            target['pseudo_labels'] = [target['pseudo_label_map'][i] for i in target['labels'].tolist() if i < 0]
            target.pop('pseudo_label_map')
            target['labels'] = target['labels'][target['labels'] >= 0]
            if self.no_overlapping_pseudo or self.remove_base_pseudo_label:
                num_gt_boxes = target['labels'].size(0)
                gt_boxes = target['boxes'][:num_gt_boxes]
                pseudo_boxes = target['boxes'][num_gt_boxes:]
                iou = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(pseudo_boxes), box_ops.box_cxcywh_to_xyxy(gt_boxes))[0]
                valid = None
                if self.no_overlapping_pseudo:
                    valid = (iou < 0.5).all(-1)
                if self.remove_base_pseudo_label:
                    valid2 = [not any([l in label.lower() for l in self.category_list]) for label in target['pseudo_labels']]
                    valid2 = torch.tensor(valid2, device=valid.device)
                    if valid is not None:
                        valid = torch.logical_and(valid, valid2)
                    else:
                        valid = valid2
                valid = valid.nonzero()[:,0]
                target['boxes'] = torch.cat([gt_boxes, pseudo_boxes[valid]], dim=0)
                target['box_ids'] = target['box_ids'][:target['boxes'].size(0)]
                target['pseudo_labels'] = [target['pseudo_labels'][i] for i in valid.tolist()]
            
        if self.use_caption:
            caption_anno = random.choice(self.captions[target['image_id'].item()])
            caption = caption_anno['caption']
            spans_anno = caption_anno['spans']
            spans = []
            NP_ids = []
            current = 0
            for span in spans_anno:
                if current != span[0]:
                    spans.append(caption[current:span[0]])
                NP_ids.append(len(spans))
                spans.append(caption[span[0]:span[1]])
                current = span[1]
            if current != len(caption):
                spans.append(caption[current:])
            assert len(NP_ids) > 0
            target['caption'] = spans
            target['np_gt'] = NP_ids

        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, map=None):
        self.return_masks = return_masks
        self.map = map

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] if 'category_id' in obj else -i - 1 for i, obj in enumerate(anno)]
        pseudo_label_map = {}
        for i, obj in enumerate(anno):
            if 'class_label' in obj:
                pseudo_label_map[-i - 1] = obj['class_label']
        classes = torch.tensor(classes, dtype=torch.int64)
        box_ids = [obj["id"] if 'id' in obj else -1 for obj in anno]
        box_ids = torch.tensor(box_ids, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        box_ids = box_ids[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["box_ids"] = box_ids
        target["pseudo_label_map"] = pseudo_label_map
        if self.map is not None:
            for idx, label in enumerate(target['labels']):
                # filter out pseudo label
                if label >= 0:
                    target['labels'][idx] = self.map[label.item()]
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        # iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # target["area"] = area[keep]
        # target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args):

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
    
    if args.clip_aug:
        crop = T.RandomCrop([224, 224]) if image_set == 'train' else T.CenterCrop([224, 224])
        return T.Compose([
            T.RandomResize([224]),
            crop,
            normalize,
        ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    
    if args.ovd:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017_base.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017_basetarget.json'),
        }
        if args.label_type == 'coco_nouns':
            PATHS['train'] = (root / "train2017", root / "annotations" / f'{mode}_train2017_base_coconouns.json')
        if args.export:
            PATHS['val'] = PATHS['train'] # export the annotated
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' ),
        }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, args), return_masks=args.masks, use_caption=args.use_caption, \
        pseudo_box=args.pseudo_box, pseudo_threshold=args.pseudo_threshold, no_overlapping_pseudo=args.no_overlapping_pseudo, base_sample_prob=args.base_sample_prob if image_set == 'train' else 1.0,
        repeat_factor_sampling=args.repeat_factor_sampling if image_set == 'train' else False, repeat_threshold=args.repeat_threshold, remove_base_pseudo_label=args.remove_base_pseudo_label)
    return dataset
