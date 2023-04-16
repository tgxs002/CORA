from numpy import require
import torch
import argparse

parser = argparse.ArgumentParser('export the region prompts from a tuned model checkpoint', add_help=False)
parser.add_argument('--model_path', default='logs/checkpoint.pth', type=str)
parser.add_argument('--name', required=True, type=str)
args = parser.parse_args()

model = torch.load(args.model_path, map_location='cpu')['model']

pe_dict = dict()

for k, v in model.items():
    if 'position' in k and 'attnpool' in k:
        assert k.startswith("backbone.0")
        pe_dict[k.replace("backbone.0.", "")] = v.clone()

torch.save(pe_dict, f"{args.name}.pth")
