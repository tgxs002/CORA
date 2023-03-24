# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import os, io
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset, DistributedWeightedSampler
from engine import evaluate, train_one_epoch, lvis_evaluate
from models import build_model
from timm.utils import get_state_dict

def get_args_parser():
    parser = argparse.ArgumentParser('CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_linear_proj_names', default=[], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--lr_backbone', default=0.0, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--text_len', default=77, type=int)
    parser.add_argument('--no_auto_resume', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--model_ema_decay', default=0.99992, type=float)
    parser.add_argument('--no_model_ema', action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--multiscale', default=False, action='store_true')
    parser.add_argument('--stage1_box', default=300, type=int)
    parser.add_argument('--use_deformable_attention', action='store_true')
    parser.add_argument('--only_deform_enc', action='store_true')
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine',),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--ovd', action='store_true')
    parser.add_argument('--no_clip_init_image', action='store_true')
    parser.add_argument('--pe_temperature', default=20, type=int)
    parser.add_argument('--box_conditioned_pe', action='store_true')
    parser.add_argument('--box_proj_dim', default=128, type=int)
    parser.add_argument('--pe_proj_dim', default=-1, type=int)
    parser.add_argument('--region_prompt_path', default='', type=str)
    parser.add_argument('--target_class_factor', default=1.0, type=float)
    parser.add_argument('--resample_factor', default=1.0, type=float)
    parser.add_argument('--filter_classes', action='store_true')
    parser.add_argument('--eval_embedding', default='', type=str)
    parser.add_argument('--rpn', action='store_true')
    parser.add_argument('--matching_threshold', default=-1., type=float)
    parser.add_argument('--pseudo_box', default='', type=str)
    parser.add_argument('--backbone_feature', default='layer3', choices=['layer3', 'layer4'], type=str)
    parser.add_argument('--end2end', action='store_true')
    parser.add_argument('--disable_init', action='store_true')
    parser.add_argument('--disable_spatial_attn_mask', action='store_true')

    # ovd control flags
    parser.add_argument('--use_nms', action='store_true')
    parser.add_argument('--no_nms', action='store_true')
    parser.add_argument('--iou_rescore', action='store_true')
    parser.add_argument('--eval_gt', action='store_true')
    parser.add_argument('--no_target_eval', action='store_true')

    parser.add_argument('--anchor_pre_matching', action='store_true')
    parser.add_argument('--aggresive_eval', action='store_true')
    parser.add_argument('--global_topk', action='store_true')
    parser.add_argument('--softmax_along', default='class', choices=['class', 'box', 'none'])
    parser.add_argument('--no_efficient_pooling', action='store_true')
    parser.add_argument('--use_efficient_pe_proj', action='store_true')
    parser.add_argument('--text_dim', default=1024, type=int)
    parser.add_argument('--add_gn', action='store_true')
    parser.add_argument('--bg_threshold', default=-1.0, type=float)
    parser.add_argument('--class_oracle', action='store_true')
    parser.add_argument('--score_threshold', default=2.0, type=float)
    parser.add_argument('--classifier_cache', default='', type=str)
    # debug
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--objectness_alpha', default=1.0, type=float)
    parser.add_argument('--split_class_p', default=0.0, type=float)
    parser.add_argument('--eval_tau', default=100, type=int)
    parser.add_argument('--iou_relabel_eval', action='store_true')
    parser.add_argument('--test_attnpool_path', default='', type=str)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="dimension of the FFN in the transformer")
    parser.add_argument('--hidden_dim', default=256, type=int, help="dimension of the transformer")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads in the transformer attention")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--fix_reference_points', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=2.0, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_class_rpn', default=2.0, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5.0, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2.0, type=float, help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1.0, type=float)
    parser.add_argument('--dice_loss_coef', default=1.0, type=float)
    parser.add_argument('--cls_loss_coef', default=2.0, type=float)
    parser.add_argument('--cls_loss_coef_rpn', default=2.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=5.0, type=float)
    parser.add_argument('--giou_loss_coef', default=2.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--contrastive_loss_coef', default=1.0, type=float)
    parser.add_argument('--class_group', default='', type=str)
    parser.add_argument('--semantic_cost', default=-1., type=float)
    parser.add_argument('--topk_matching', default=-1, type=int)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='data/coco')
    parser.add_argument('--lvis_path', default='', type=str)
    parser.add_argument("--label_map", default=False, action="store_true")
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_misclassified', action='store_true')
    parser.add_argument('--repeat_factor_sampling', action='store_true')
    parser.add_argument('--repeat_threshold', default=0.001, type=float)
    parser.add_argument('--condition_on_text', action='store_true')
    parser.add_argument('--condition_bottleneck', default=128, type=int)

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing. We must use cuda.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint, empty for training from scratch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_target', action='store_true')
    parser.add_argument('--eval_every_epoch', default=1, type=int, help='eval every ? epoch')
    parser.add_argument('--save_every_epoch', default=5, type=int, help='save model weights every ? epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--debug', action='store_true', 
                    help="For debug only. It will perform only a few steps during trainig and val.")
    parser.add_argument('--label_version', default='', choices=['', 'RN50x4base', 'RN50x4base_coconames', 'RN50x4base_prev', 'RN50base', 'ori', 'custom'])
    parser.add_argument('--num_label_sampled', default=-1, type=int)
    
    parser.add_argument('--clip_aug', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only."
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, post_processors = build_model(args)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = utils.fast_ema(
            model,
            decay=args.model_ema_decay,
            device='',
            resume='',
            skip_keywords=['classifier', 'backbone'])

    def match_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of params in model: ', n_parameters)

    def match_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if "backbone.0" not in n and not match_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if "backbone.0" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    dataset_val = build_dataset(image_set='val', args=args)
    if args.debug:
        dataset_train = dataset_val
    else:
        dataset_train = build_dataset(image_set='train', args=args)

    if args.distributed:
        # TODO: update case where not using distributed mode
        if args.repeat_factor_sampling:
            sampler_train = DistributedWeightedSampler(dataset_train, weight=dataset_train.rep_factors)
        else:
            sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)        
    if not args.no_auto_resume and not args.resume:
        checkpoint_dir = os.path.join(str(output_dir), "checkpoint.pth")
        if os.path.exists(checkpoint_dir):
            args.resume = checkpoint_dir

    best_performance = -1

    if args.resume:
        print(f"loading checkpoint from {args.resume}")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        if 'best_performance' in checkpoint:
            best_performance = checkpoint['best_performance']
        if 'model_ema'in checkpoint and model_ema is not None:
            _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

    if args.eval:
        # debug 
        if args.dataset_file == "lvis" and False:
            test_stats, coco_evaluator = lvis_evaluate(
                model_ema.ema if args.model_ema else model,
                criterion,
                post_processors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
                args.label_map,
                False,
                args,
            )
        else:
            test_stats, coco_evaluator = evaluate(
                model_ema.ema if args.model_ema else model, criterion, post_processors, data_loader_val, base_ds, device, args.output_dir, args=args
            )
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training...")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args=args, model_ema=model_ema
        )
        lr_scheduler.step()
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.save_every_epoch == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'best_performance': best_performance,
                    'model_ema': get_state_dict(model_ema) if model_ema else -1,
                }, checkpoint_path)

        if (epoch + 1) % args.eval_every_epoch == 0:
            if args.dataset_file == "lvis":
                test_stats, coco_evaluator = lvis_evaluate(
                    model if model_ema is None else model_ema.ema,
                    criterion,
                    post_processors,
                    data_loader_val,
                    base_ds,
                    device,
                    args.output_dir,
                    args.label_map,
                    False,
                    args,
                )
                if utils.is_main_process():
                    APR = test_stats['APr']
                    if APR > best_performance:
                        best_performance = APR
                        if args.save_best:
                            best_path = output_dir / 'checkpoint_best.pth'
                            utils.save_on_master({
                                'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'args': args,
                                'best_performance': best_performance,
                                'model_ema': get_state_dict(model_ema) if model_ema else -1,
                            }, best_path)
            else:
                test_stats, coco_evaluator = evaluate(
                    model if model_ema is None else model_ema.ema, criterion, post_processors, data_loader_val, base_ds, device, args.output_dir, args=args
                )
                targetAP50 = test_stats['coco_eval_bbox'][-3]
                if targetAP50 > best_performance:
                    best_performance = targetAP50
                    if args.save_best:
                        best_path = output_dir / 'checkpoint_best.pth'
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            'best_performance': best_performance,
                            'model_ema': get_state_dict(model_ema) if model_ema else -1,
                        }, best_path)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    filenames.append(f'{epoch:04}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training completed.\nTotal training time: {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CORA", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.no_model_ema:
        args.model_ema = False
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
