from collections import Counter
import json
import argparse
from util import box_ops
import torch

parser = argparse.ArgumentParser("merge lvis annotation")
parser.add_argument('--no_rare_path', type=str)
parser.add_argument('--reference_path', type=str)
parser.add_argument('--pseudo_path', type=str)
parser.add_argument('--output_file', type=str)
args = parser.parse_args()

with open(args.no_rare_path) as f:
    no_rare = json.load(f)

with open(args.pseudo_path) as f:
    pseudo_label = json.load(f)

with open(args.reference_path) as f:
    reference = json.load(f)

all_categories = reference['categories']
del reference
rare_ids = [t['id'] for t in all_categories if t['frequency'] == 'r']

for cat in all_categories:
    if cat['frequency'] == 'r':
        cat['image_count'] == 0
        cat['instance_count'] == 0

no_rare_by_img = dict()
for anno in no_rare['annotations']:
    image_id = anno['image_id']
    if image_id not in no_rare_by_img:
        no_rare_by_img[image_id] = list()
    no_rare_by_img[image_id].append(anno)

good_annotation = list()

# filter out the annotations that overlap with a base gt box
for anno in pseudo_label:
    image_id = anno['image_id']
    base_anno = no_rare_by_img[image_id] if image_id in no_rare_by_img else []
    pseudo = torch.tensor(anno['bbox'])
    base = torch.tensor([t['bbox'] for t in base_anno])
    pseudo[2:] += pseudo[:2]
    base[:,2:] += base[:,:2]
    pseudo = pseudo.unsqueeze(0)
    iou = box_ops.box_iou(pseudo, base)[0]
    if iou.max() > 0.5:
        continue
    good_annotation.append(anno)

image_counter = Counter()
instance_counter = Counter()
seen_image = set()
for anno in good_annotation:
    if (anno['image_id'], anno['category_id']) not in seen_image:
        image_counter[anno['category_id']] += 1
        seen_image.add((anno['image_id'], anno['category_id']))
    instance_counter[anno['category_id']] += 1

for k in image_counter.keys():
    cat = all_categories[k - 1]
    assert cat['id'] == k
    cat['image_count'] = image_counter[k]
    cat['instance_count'] = instance_counter[k]

max_id = max([k['id'] for k in no_rare['annotations']])
for i, anno in enumerate(good_annotation):
    anno['id'] = max_id + i + 1

no_rare['categories'] = all_categories
no_rare['annotations'].extend(good_annotation)

with open(args.output_file, 'w') as f:
    json.dump(no_rare, f)