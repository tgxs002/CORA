import torch
import argparse

parser = argparse.ArgumentParser('', add_help=False)
parser.add_argument('--embeddings', default='', type=str)
parser.add_argument('--num_clusters', default=50, type=int)
args = parser.parse_args()

embed = torch.load(args.embeddings)
embed = embed / embed.norm(dim=-1, keepdim=True)

# random initialize the centroids
new_centroids = embed[torch.randperm(embed.size(0))[:args.num_clusters]]
centroids = torch.zeros_like(new_centroids)
cnt = 0

while (new_centroids - centroids).abs().sum() > 1e-5:
    cnt += 1
    centroids = new_centroids
    distance = embed @ centroids.t()
    assignment = distance.argmax(dim=-1)
    new_centroids = torch.zeros_like(centroids)
    new_centroids.scatter_add_(dim=0, index=assignment.unsqueeze(-1).expand_as(embed), src=embed)
    new_centroids = new_centroids / new_centroids.norm(dim=-1, keepdim=True)

out = {k+1: v.item() for k, v in enumerate(assignment)}
torch.save(out, f"lvis_cluster_{args.num_clusters}.pth")