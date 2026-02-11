# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import sys
import os
from clustercontrast.models.classifier import NormalizedClassifier

sys.path.append(os.path.dirname(sys.path[0]))
import argparse
import os.path as osp
import random
import numpy as np
import sys
import wandb
from datetime import datetime

sys.path.append(' ')
import torch
from torch import nn
from torch.backends import cudnn
from clustercontrast.utils.osutils import str2bool
from clustercontrast.utils.prepare_optimizer import make_optimizer_1stage, make_optimizer_2stage
from clustercontrast.utils.prepare_scheduler import create_scheduler, create_scheduler_cos
from train.train_stage1 import do_train_text_stage1
from train.ics_train_stage2 import ics_intra_inter_stage
from train.RN50 import make_model
from clustercontrast import datasets
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.image_data_loader import Loaders

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(loaders):
    train_loader = loaders.train_iter
    return train_loader


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def get_test_loader(args, loaders):
    if args.dataset == 'market1501':
        query_loader = loaders.market_query_loader
        gallery_loader = loaders.market_gallery_loader
        test_loader = [query_loader, gallery_loader]
    elif args.dataset == 'msmt17':
        query_loader = loaders.msmt_query_loader
        gallery_loader = loaders.msmt_gallery_loader
        test_loader = [query_loader, gallery_loader]
    else:
        query_loader = loaders.duke_query_loader
        gallery_loader = loaders.duke_gallery_loader
        test_loader = [query_loader, gallery_loader]


    return test_loader


def get_cluster_loader(loaders):
    pe_loader = loaders.propagate_loader  # 会生成全局标签
    return pe_loader


def create_model(args):
    if args.dataset == 'market1501':
        cam_num = 6
    elif args.dataset == 'msmt17':
        cam_num = 15
    elif args.dataset == 'sysu':
        cam_num = 6
    else:
        cam_num = 8

    model = make_model(args, camera_num=cam_num)
    model.cuda()

    # model = nn.DataParallel(model)

    num_classes = 0  # Accumulation Label
    if args.dataset == 'market1501':
        num_classes = 3262
    elif args.dataset == 'msmt17':
        num_classes = 4821
    elif args.dataset == 'dukemtmc':
        num_classes = 2196
    elif args.dataset == 'sysu':
        num_classes = 123

    cam_classifier = NormalizedClassifier(feature_dim=2048, num_classes=num_classes)
    cam_classifier.cuda()
    # cam_classifier = nn.DataParallel(cam_classifier)

    return model, cam_classifier


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    sys.stdout = Logger(osp.join(args.logs_dir, args.dataset, f'log_{args.dataset}.txt'))
    print(datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒"))
    print("==========\nArgs:{}\n==========".format(args))
    # Create datasets
    selected_idx = None
    new_labels = ''

    if args.wandb_enabled:
        if args.wandb_group:
            wandb.init(project=f'CCAFL-ics-{args.dataset}',
                       group=args.wandb_group,
                       dir=f'wandb/{args.dataset}')
        else:
            wandb.init(project=f'CCAFL-ics-{args.dataset}',
                       group="parameter_tuning",
                       dir=f'wandb/{args.dataset}')

    loaders = Loaders(args, selected_idx, new_labels, learning_setting='semi_supervised')
    print("==> Load intra-camera labeled dataset: ", args.dataset)
    test_loader = get_test_loader(args, loaders)

    id_count_each_cam = loaders.id_count_each_cam
    # img_count_each_cam = loaders.img_count_each_cam

    id_count_each_cam = np.array(id_count_each_cam)
    cameras = len(id_count_each_cam)

    print('  累积ID个数: {}'.format(np.sum(id_count_each_cam)))
    print('  {} number of camera: {}'.format(args.dataset, cameras))
    # Create model
    model, _ = create_model(args)

    resume_path = '/media/deep/Data/Share/1.Pytorch_Re-id_Cross_SSL/CCAFL/logs/train_ics/msmt17/model_best.pth.tar'
    resume_path = ''
    if resume_path != '':
        print(f"==> Resuming from checkpoint...{resume_path}")
        checkpoint = torch.load(resume_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])


    # define optimizer
    # optimizer_1stage = make_optimizer_1stage(model)
    # scheduler_1stage = create_scheduler_cos(optimizer_1stage, num_epochs=args.epochs_stage1)

    '''
    do_train_text_stage1(args,
                    model,
                    loaders,
                    optimizer_1stage,
                    scheduler_1stage)
    '''

    optimizer = make_optimizer_2stage(model)  # Optimizer
    lr_scheduler = create_scheduler(optimizer)
    ics_intra_inter_stage(args,
                    model,
                    loaders,
                    test_loader,
                    optimizer,
                    lr_scheduler,
                    id_count_each_cam,
                    cameras)

    print(datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒"))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501', choices=['dukemtmc', 'market1501', 'msmt17'])
    # parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--num-instances', type=int, default=32)
    parser.add_argument('-b', '--batch-size', type=int, default=256)

    # model
    parser.add_argument('--momentum', type=float, default=0.1, help="update momentum for the hybrid memory")
    #vit
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')

    # Stage1
    parser.add_argument('--stage1_checkpoint_period', type=int, default=5)
    parser.add_argument('--epochs_stage1', type=int, default=50)  # Text Stage
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--intra_epoch', type=int, default=5)
    parser.add_argument('--start_adv_epoch', type=int, default=80)
    parser.add_argument('--end_adv_epoch', type=int, default=80)

    parser.add_argument('--iters', type=int, default=400)

    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05, help="temperature for scaling contrastive loss")

    # Wandb params
    parser.add_argument("--wandb_enabled", action='store_true', default=False)
    parser.add_argument("--wandb_group", type=str, default='')
    parser.add_argument("--wandb_index", type=int, default=0)

    # path
    working_dir = '/media/deep/SSD/Dataset_ReID'
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=working_dir)
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default='logs/train_ics/')

    parser.add_argument('--market_path', type=str, default='Market-1501-v15.09.15/')
    parser.add_argument('--duke_path', type=str, default='DukeMTMC-reID/DukeMTMC-reID')
    parser.add_argument('--msmt_path', type=str, default='MSMT17')

    # PK for Inter Stage
    parser.add_argument('--class_per_batch', type=int, default=16)  # triplet sampling, number of IDs per batch16
    parser.add_argument('--track_per_class', type=int, default=4)  # triplet sampling, number of images per ID per batch

    # 对比学习设置
    parser.add_argument('--use_inter_camera', type=str2bool, default=False)
    parser.add_argument('--use-intra-hard', type=str2bool, default=True)
    # parser.add_argument('--use-intra-hard',  action="store_true")

    parser.add_argument("--using_offset", action="store_true", default=False, help="feature offset")

    # 损失函数设置
    parser.add_argument('--lossweight', type=float, default=0.0)
    parser.add_argument('--epsilon', type=float, default=0.0)

    # parser.add_argument('--nt', type=int, default=5)
    parser.add_argument('-a', '--arch', type=str, default='CLIP', choices=['CLIP', 'agw'])
    parser.add_argument('-dis', '--distance', type=str, default='ICS', choices=['ICS', 'CAJ', 'UN'])

    parser.add_argument('--K_search', type=int, default=20)  # 初始候选池大小（仅用于 distance ranking，不等于邻居数）
    parser.add_argument('--K_intra', type=int, default=10)  # intra-camera positives 上限，保护性数值
    parser.add_argument('--K_cross', type=int, default=10)  # cross-camera reciprocal neighbors 上限

    parser.add_argument('--tau_intra', type=float, default=2.0)  # cross-camera reciprocal neighbors 上限
    parser.add_argument('--beta', type=float, default=0.5)  # cross-camera reciprocal neighbors 上限

    # -----------CA Jaccard ---------------
    parser.add_argument('--k1', type=int, default=30, help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6, help="hyperparameter for jaccard distance")
    parser.add_argument('--ckrnns', action='store_true')
    parser.add_argument('--k1-intra', type=int, default=5)
    parser.add_argument('--k1-inter', type=int, default=20)
    parser.add_argument('--clqe', action='store_true')
    parser.add_argument('--k2-intra', type=int, default=3)
    parser.add_argument('--k2-inter', type=int, default=3)
    # -----------CA Jaccard ---------------

    main()
