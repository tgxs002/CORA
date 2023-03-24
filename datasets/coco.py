# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.
"""
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
    def __init__(self, img_folder, ann_file, transforms, return_masks, tag_annotation, pseudo_box):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        
        self.all_categories = {k['id']: k['name'] for k in self.coco.dataset['categories']}
        self.category_list = [self.all_categories[k] for k in sorted(self.all_categories.keys())]
        self.category_ids = {v: k for k, v in self.all_categories.items()}
        self.label2catid = {k: self.category_ids[v] for k, v in enumerate(self.category_list)}
        self.catid2label = {v: k for k, v in self.label2catid.items()}
        self.image_tags = json.load(open(tag_annotation, 'r')) if tag_annotation is not None else None

        self.use_pseudo_box = pseudo_box != ""
        if self.use_pseudo_box:
            with open(pseudo_box, 'r') as f:
                pseudo_annotations = json.load(f)
            self.pseudo_annotations = dict()
            for annotation in pseudo_annotations:
                if annotation['image_id'] not in self.pseudo_annotations:
                    self.pseudo_annotations[annotation['image_id']] = []
                self.pseudo_annotations[annotation['image_id']].append(annotation)
        
        self.prepare = ConvertCocoPolysToMask(return_masks, map=self.catid2label)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]

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
            num_gt_boxes = target['labels'].size(0)
            gt_boxes = target['boxes'][:num_gt_boxes]
            pseudo_boxes = target['boxes'][num_gt_boxes:]
            iou = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(pseudo_boxes), box_ops.box_cxcywh_to_xyxy(gt_boxes))[0]
            valid = (iou < 0.5).all(-1).nonzero()[:,0]
            target['boxes'] = torch.cat([gt_boxes, pseudo_boxes[valid]], dim=0)
            target['pseudo_labels'] = [target['pseudo_labels'][i] for i in valid.tolist()]

        if self.image_tags is not None:
            target['class_labels'] = self.image_tags[str(image_id)]

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

        # classes = [obj["category_id"] for obj in anno]
        classes = [obj["category_id"] if 'category_id' in obj else -i - 1 for i, obj in enumerate(anno)]
        pseudo_label_map = {}
        for i, obj in enumerate(anno):
            if 'class_label' in obj:
                pseudo_label_map[-i - 1] = obj['class_label']
        classes = torch.tensor(classes, dtype=torch.int64)

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
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["pseudo_label_map"] = pseudo_label_map
        if self.map is not None:
            for idx, label in enumerate(target['labels']):
                target['labels'][idx] = self.map[label.item()] if label.item() >= 0 else label.item()
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
        if args.label_version == 'RN50x4base':
            PATHS['train'] = (root / "train2017", root / "annotations" / f'{mode}_train2017_base_RN50x4relabel.json')
        elif args.label_version == 'RN50x4base_prev':
            PATHS['train'] = (root / "train2017", root / "annotations" / f'{mode}_train2017_base_RN50x4relabel_pre.json')
        elif args.label_version == 'RN50x4base_coconames':
            PATHS['train'] = (root / "train2017", root / "annotations" / f'{mode}_train2017_base_RN50x4relabel_coconames.json')
        elif args.label_version == 'RN50base':
            PATHS['train'] = (root / "train2017", root / "annotations" / f'{mode}_train2017_base_RN50relabel.json')
        elif args.label_version == 'custom':
            PATHS['val'] = (root / "custom", root / "annotations" / f'custom.json')
        if args.eval_target:
            PATHS['val'] = (root / "val2017", root / "annotations" / f'{mode}_val2017_target.json')
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' ),
        }

    img_folder, ann_file = PATHS[image_set]
    if hasattr(args, 'cls_annotation_path'):
        transforms = make_coco_transforms("val", args)
        tag_annotation = args.cls_annotation_path
    else:
        transforms = make_coco_transforms(image_set, args)
        tag_annotation = None
    dataset = CocoDetection(img_folder, ann_file, transforms=transforms, return_masks=args.masks, tag_annotation=tag_annotation, pseudo_box=args.pseudo_box)
    return dataset
