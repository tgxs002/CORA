"""
LVIS dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import math
from numpy.random import choice

import datasets.transforms as T

from .torchvision_datasets import LvisDetection as TvLvisDetection


class LvisDetection(TvLvisDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, label_map, debug, repeat_factor_sampling=False, repeat_threshold=0.001, class_group=''):
        super(LvisDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.debug = debug
        self.prepare = ConvertCocoPolysToMask(return_masks, self.cat2label, label_map)

        self.all_categories = {k['id']: k['name'].replace('_', ' ') for k in self.lvis.dataset['categories']}
        self.category_list = [self.all_categories[k] for k in sorted(self.all_categories.keys())]
        self.category_ids = {v: k for k, v in self.all_categories.items()}
        self.label2catid = {k: self.category_ids[v] for k, v in enumerate(self.category_list)}
        if class_group:
            self.class_group = torch.load(class_group)
            self.class_group = {self.cat2label[k]: v for k, v in self.class_group.items() if k in self.cat2label}
            self.class_group = [self.class_group[i] for i in range(len(self.class_group))]
        else:
            self.class_group = None
        # self.catid2label = {v: k for k, v in self.label2catid.items()}
        # code adapted from Detectron2, many thanks to the authors
        if repeat_factor_sampling:
            # 1. For each category c, compute the fraction of images that contain it: f(c)
            counter = {k: 1e-3 for k in self.category_ids.values()}
            for id in self.ids:
                ann_ids = self.lvis.get_ann_ids(img_ids=[id])
                target = self.lvis.load_anns(ann_ids)
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
                ann_ids = self.lvis.get_ann_ids(img_ids=[id])
                target = self.lvis.load_anns(ann_ids)
                cats = [t['category_id'] for t in target]
                cats = set(cats)
                rep_factor = max({category_rep[cat_id] for cat_id in cats}, default=1.0)
                rep_factors.append(rep_factor)
            self.rep_factors = rep_factors

    def __getitem__(self, idx):
        img, target = super(LvisDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if len(target["labels"]) == 0:
            return self[(idx + 1) % len(self)]
        else:
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
    def __init__(self, return_masks=False, cat2label=None, label_map=False):
        self.return_masks = return_masks
        self.cat2label = cat2label
        self.label_map = label_map

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.label_map:
            classes = [self.cat2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
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
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

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

    normalize = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val" or image_set == "pseudo_label":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.lvis_path)
    assert root.exists(), f"provided LVIS path {root} does not exist"
    PATHS = {
        "train": (root, root / "lvis_v1_train_norare.json"),
        "val": (root, root / "lvis_v1_val.json"),
        "pseudo_label": (root, root / "lvis_v1_train_norare.json"),
    }

    if args.label_version == 'RN50x4base':
        PATHS['train'] = (root, root / "lvis_v1_train_norare_relabel.json")
    
    if args.no_target_eval:
        PATHS['val'] = (root, root / "lvis_v1_val_norare.json")

    img_folder, ann_file = PATHS[image_set]
    dataset = LvisDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        return_masks=args.masks,
        label_map=args.label_map,
        debug=args.debug,
        repeat_factor_sampling=args.repeat_factor_sampling and image_set == 'train',
        repeat_threshold=args.repeat_threshold and image_set == 'train',
        class_group=args.class_group if image_set == 'train' else '',
    )
    return dataset
