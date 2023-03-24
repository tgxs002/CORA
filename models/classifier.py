# ------------------------------------------------------------------------
# Mofified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
import os
from unicodedata import category
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
from models.clip.clip import _MODELS, _download, available_models, tokenize
from models.clip.model import Transformer
from models.clip.prompts import imagenet_templates

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Classifier(torch.nn.Module):
    def __init__(self, name: str,
                 token_len = 77,
                 classifier_cache = ''):
        super().__init__()
        if classifier_cache == '':
            self.cache = {}
        else:
            self.cache = torch.load(classifier_cache)
        
        if "clip" in name:
            name = name.replace('clip_', '')
            if name in _MODELS:
                model_path = _download(_MODELS[name], os.path.expanduser("~/.cache/clip"))
            elif os.path.isfile(name):
                model_path = name
            else:
                raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

            with open(model_path, 'rb') as opened_file:
                model = torch.jit.load(opened_file, map_location="cpu")
                state_dict = model.state_dict()

            embed_dim = state_dict["text_projection"].shape[1]
            # self.context_length = state_dict["positional_embedding"].shape[0]
            self.context_length = token_len
            self.vocab_size = state_dict["token_embedding.weight"].shape[0]
            transformer_width = state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
            
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )
            
            self.token_embedding = nn.Embedding(self.vocab_size, transformer_width)
            self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
            self.ln_final = LayerNorm(transformer_width)
            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
            
            # load 
            self.transformer.load_state_dict({k.replace('transformer.', ''): v for k, v in state_dict.items() if k.startswith('transformer.')})
            self.token_embedding.load_state_dict({k.replace('token_embedding.', ''): v for k, v in state_dict.items() if 'token_embedding' in k})
            self.ln_final.load_state_dict({k.replace('ln_final.', ''): v for k, v in state_dict.items() if 'ln_final' in k})
            self.positional_embedding.data = state_dict['positional_embedding']
            self.text_projection.data = state_dict['text_projection']
            
            for v in self.parameters():
                v.requires_grad_(False)
            
        else:
            raise NotImplementedError
            
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_text(self, text):
        x = self.token_embedding(text)

        x = x + self.positional_embedding[:self.context_length]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.to(x.dtype)

        return x
    
    def forward_feature(self, category_list):
        templates = imagenet_templates
        texts = [template.format(cetegory) for cetegory in category_list for template in templates] #format with class
        texts = tokenize(texts, context_length=self.context_length, truncate=True).to(self.positional_embedding.device)
        class_embeddings = []
        cursor = 0
        step = 3000
        while cursor <= len(texts):
            class_embeddings.append(self.encode_text(texts[cursor:cursor + step]))
            cursor += step
        class_embeddings = torch.cat(class_embeddings)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.unflatten(0, (len(category_list), len(templates)))
        class_embedding = class_embeddings.mean(dim=1)
        class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        return class_embedding
    
    def forward(self, category_list):
        new_category = [category for category in category_list if category not in self.cache]
        with torch.no_grad():
            new_class_embedding = self.forward_feature(new_category)
            for category, feat in zip(new_category, new_class_embedding):
                self.cache[category] = feat.to('cpu')
        class_embedding = torch.stack([self.cache[category] for category in category_list]).to(self.positional_embedding.device)

        return class_embedding

def build_classifier(args):
    classifier = Classifier(args.backbone, token_len=args.text_len, classifier_cache=args.classifier_cache)
    return classifier
