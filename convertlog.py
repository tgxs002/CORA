import argparse
import json
import os
import pandas as pd
parser = argparse.ArgumentParser('', add_help=False)
parser.add_argument('--exp_folder', default='', type=str)
args = parser.parse_args()

with open(os.path.join(args.exp_folder, "log.txt")) as f:
    logs = f.readlines()

stats = []
for res in logs:
    stats.append(json.loads(res))

filtered_stats = []
for stat in stats:
    stat = {k: v for k, v in stat.items() if not any([t in k for t in ['_0', '_1', '_2', '_3', '_4', 'unscaled', 'parameters', 'error']])}
    test_coco_eval_bbox = stat.pop('test_coco_eval_bbox')
    stat['base_AP50'] = test_coco_eval_bbox[-4]
    stat['target_AP50'] = test_coco_eval_bbox[-3]
    stat['base_recall'] = test_coco_eval_bbox[-2]
    stat['target_recall'] = test_coco_eval_bbox[-1]
    filtered_stats.append(stat)

df = pd.DataFrame.from_records(filtered_stats)
df.to_excel(os.path.join(args.exp_folder, "log.xlsx"))