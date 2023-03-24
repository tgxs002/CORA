import argparse
import json
import os
parser = argparse.ArgumentParser('', add_help=False)
parser.add_argument('--ori_annotation', required=True, type=str)
parser.add_argument('--relabel_folder', required=True, type=str)
parser.add_argument('--num_files', required=True, type=int)
parser.add_argument('--target_name', required=True, type=str)
args = parser.parse_args()

with open(args.ori_annotation, 'r') as f:
    anno = json.load(f)

d = dict()

for i in range(args.num_files):
    filename = os.path.join(args.relabel_folder, f"export_label_{i}.json")
    with open(filename, 'r') as f:
        d.update(json.load(f))

d = {int(k): v for k, v in d.items()}
total = len(d)
correct = 0

for inst in anno['annotations']:
    if inst['id'] in d:
        if inst['category_id'] == d[inst['id']]:
            correct += 1
        inst['category_id'] = d[inst['id']]

print(f"label accuracy: {correct / total}")
print(f"{total} relabeled instances in total")

with open(args.target_name, 'w') as f:
    json.dump(anno, f)
