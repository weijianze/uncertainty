#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp

cp_dir   = './checkpoint/'

def cls_args():

    parser = argparse.ArgumentParser(description='PyTorch for arcface-UE')

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--workers', type=int,  default=0)  # TODO

    # -- model
    parser.add_argument('--drop_ratio', type=float, default=0.4)          # TODO
    parser.add_argument('--used_as',    type=str,   default='UE', choices=['baseline', 'UE', 'backbone'])   
    parser.add_argument('--in_feats',   type=int,   default=256)

    # -- fc-layer
    parser.add_argument('--t',          type=float,  default=0.2)          # MV
    parser.add_argument('--margin',     type=float,  default=0.5)          # [0.5-arcface, 0.35-cosface]
    parser.add_argument('--easy_margin',type=bool,   default=True)
    parser.add_argument('--scale',      type=float,  default=64)           # FIXED
    parser.add_argument('--kl_lambda',  type=float,  default=0.01)         # default = 0.01
    parser.add_argument('--fc_mode',    type=str,    default='arcface',  choices=['softmax', 'sphere', 'cosface', 'arcface', 'mvcos', 'mvarc'])
    parser.add_argument('--hard_mode',  type=str,    default='adaptive', choices=['fixed', 'adaptive']) # MV
    parser.add_argument('--loss_mode',  type=str,    default='ce',       choices=['ce', 'focal_loss', 'hardmining'])
    parser.add_argument('--hard_ratio', type=float,  default=0.9)          # hardmining
    parser.add_argument('--loss_power', type=int,    default=2)            # focal_loss

    # -- optimizer
    parser.add_argument('--start_epoch', type=int,   default=1)        #
    parser.add_argument('--end_epoch',   type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=300)      # TODO | 300
    parser.add_argument('--base_lr',     type=float, default=0.001)    
    parser.add_argument('--lr_adjust',   type=list,  default=[30, 50, 70])
    parser.add_argument('--gamma',       type=float, default=0.1)      # FIXED
    parser.add_argument('--weight_decay',type=float, default=1e-3)     # FIXED
    parser.add_argument('--resume',      type=str,   default='')       # ch

    # -- save or print
    parser.add_argument('--is_debug',  type=str,   default=False)   # TODO
    parser.add_argument('--print_freq',type=int,   default=10)  # (3804846, 512, 7432)
    parser.add_argument('--save_freq', type=int,   default=3)  # TODO

    # -- other setting
    parser.add_argument('--img_size', type=int,   default=128)  # TODO
    
    args = parser.parse_args()

    return args
