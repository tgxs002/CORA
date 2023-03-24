import torch
import argparse

parser = argparse.ArgumentParser('', add_help=False)
parser.add_argument('--checkpoint', type=str, required=True)

args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location='cpu')
attn_pool = {k: v for k, v in checkpoint['model'].items() if 'attn_pool.positional_embedding' in k}

assert all([k.startswith("backbone.attn_pool.positional_embedding.") for k in attn_pool.keys()])

attn_pool = {k[40:]: v for k, v in attn_pool.items()}

torch.save(attn_pool, "pe.pth")