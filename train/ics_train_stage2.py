# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import os.path as osp
import numpy as np
import sys
import collections
import time
from datetime import timedelta
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

sys.path.append(' ')
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainer_ics import TrainerICS
from clustercontrast.evaluators import test, extract_text_features
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.cluster_labels import img_association
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance, compute_CAJ_jaccard_distance
from clustercontrast.evaluators import extract_vit_features, extract_vit_features_for_isc
from clustercontrast.utils.label_refinement_ics import compute_ics_ckrnn_jaccard_distance
from clustercontrast.utils.label_refinement_ics_improved import compute_ics_jaccard_distance
from clustercontrast.utils.label_refinement_ics_simplified import compute_ics_ckrnn_jaccard_distance_simplified
from clustercontrast.utils.label_refinement_ics_memory_efficient import compute_ics_ckrnn_jaccard_optimized
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.data import transforms as T

from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import wandb


def cams_offset(features, cams):
    cams = cams if isinstance(cams, torch.Tensor) else torch.tensor(cams).long()
    uniq_cam = torch.unique(cams)
    # cams_num = torch.max(cams) + 1
    cams_center = []
    for i in uniq_cam:
        cams_center.append(features[torch.where(cams == i)].mean(0))
    cams_center = torch.stack(cams_center)
    global_center = torch.mean(cams_center, 0)
    offset = cams_center - global_center
    expand_offset = offset[cams]
    features = features - expand_offset  # 每个样本的特征进行偏移补偿
    return features, offset


def get_train_loader(loaders):
    train_loader = loaders.train_iter
    return train_loader


def get_train_loader_ics(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=None, transform=train_transformer),
                   batch_size=batch_size,
                   num_workers=workers,
                   sampler=sampler,
                   shuffle=not rmgs_flag,
                   pin_memory=True,
                   drop_last=True), length=iters)

    return train_loader


'''
def get_cluster_loader(loaders):
    pe_loader = loaders.propagate_loader  # 会生成全局标签
    return pe_loader
'''

def generate_camera_mask(cam_num, id_count_each_cam):
    #########################################################
    accu2cam = []  # 存储每个摄像头的样本索引范围（np.arange）
    cam2accu = []  # 存储每个摄像头的二进制掩码（torch.Tensor）.生成一个掩码矩阵 cam2accu，用于标识每个样本属于哪个摄像头
    # cam2accu_unique = []
    before = 0
    after = 0

    # 生成摄像头掩码 cam2accu
    for cam_i in range(cam_num):
        if cam_i < cam_num:
            after += id_count_each_cam[cam_i]
        accu2cam.append(np.arange(before, after))

        if cam_i < cam_num:
            before += id_count_each_cam[cam_i]
        cam_mask = torch.zeros(sum(id_count_each_cam))

        cam2accu.append(cam_mask.scatter_(0, torch.LongTensor(accu2cam[cam_i]), 1))  # Create Mask

        # true_cam_mask = torch.zeros(cam_num)
        # cam2accu_unique.append(true_cam_mask.scatter_(0,torch.LongTensor([cam_i]),1))

    # cam2accu_unique = torch.stack(cam2accu_unique, dim=0)
    cam2accu = torch.stack(cam2accu, dim=0)  # Market表示为[6, 3262]， 即每个摄像头中，全局3262个是来自于哪个摄像头
    return cam2accu

def get_correct_num_and_pseudo2accu_list(pseudo_labels, id_count_each_cam, new_IDs):
    correct_num = 0
    pseudo2accu = collections.defaultdict()
    pseudo2accu_list = []
    # pid2cam = []
    # pid2cam_unique = []

    for i in np.unique(pseudo_labels):
        pseudo2accu[i] = np.where(pseudo_labels == i)  # batch中每个图片的accu_label

        if i != -1:
            pseudo2accu_tensor = torch.zeros(sum(id_count_each_cam))
            pseudo2accu_tensor.scatter_(0, torch.LongTensor(np.where(pseudo_labels == i)[0]),
                                        1)  # PyTorch 中的一个原地操作（in-place operation），用于将指定值填充到目标张量的指定位置。

            # 检测聚类后的标签是否存在噪声
            if len(np.unique(new_IDs[torch.where(pseudo2accu_tensor == 1)[0]])) == 1:
                correct_num += 1
            pseudo2accu_list.append(pseudo2accu_tensor)

    pseudo2accu_list = torch.stack(pseudo2accu_list, dim=0)
    return correct_num, pseudo2accu_list



# Single modality
def ics_intra_inter_stage(args,
                     model,
                     loaders,  # 训练Loader
                     test_loader,  # 测试Loader
                     optimizer,
                     scheduler,
                     id_count_each_cam,  # ???
                     cameras,
                     ):
    print('\nstart stage2 training---------------------------------')  # Inter Camera

    dataset_n_class = {'dukemtmc':702, 'market1501':751, 'msmt17':1041}
    global  best_mAP
    best_mAP = 0
    start_time = time.monotonic()
    # text_features = extract_text_features(model, loaders.propagate_loader)  # 后面哪里用到

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size=[256, 128]

    transform_train = transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    print('\n', '*'*10, 'Create TrainerICS')
    ics_trainer = TrainerICS(args,
                          model,
                          id_count_each_cam)

    # Stage2 Training
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch} Training start\n')
        if args.wandb_enabled: wandb.log({'epoch': epoch,})


        features, _, gt_labels, cam_ids, intra_ids, global_labels = extract_vit_features_for_isc(model, loaders.propagate_loader)

        # get_features_tensor, features_np.numpy(), id_labels.numpy(), cam_id_labels.numpy(), intra_cam_labels.numpy(), accu_labels.numpy(), gt_labels

        if args.distance == 'ICS':
            print('==> compute_ics_jaccard_distance ')
            rerank_dist = compute_ics_jaccard_distance(features,
                                                     cam_labels=cam_ids,
                                                     intra_camera_labels=intra_ids,
                                                     epoch=epoch,
                                                     args=args,
                                                     gt_labels=gt_labels)
        elif args.distance == 'CAJ':
            # CA Jaccard
            print('==> compute_CAJ_jaccard_distance ')
            rerank_dist = compute_CAJ_jaccard_distance(features, cam_labels=cam_ids, epoch=epoch, args=args)
        elif args.distance == 'UN':
            print('==> compute_Unsupervised ')
            # 完全无监督
            rerank_dist = compute_jaccard_distance(features, k1=30, k2=6, search_option=3)
        else:
            raise NotImplementedError




        pseudo_labels = DBSCAN(eps=0.6,
                               min_samples=4,
                               metric='precomputed',
                               n_jobs=-1).fit_predict(rerank_dist)

        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        # 构建新的Dataset
        pseudo_labeled_dataset = []  # 里面是元组
        p_labels = []
        gt_labels = []

        # images, label, cams, _, cam_label, accu_label, _
        for i, ((images, label, cams, _, cam_label, accu_label, _), p_label) in enumerate(zip(loaders.train_samples, pseudo_labels)):  # 因为前面sorted
            p_labels.append(p_label.item())
            gt_labels.append(label)

            if p_label != -1:
                pseudo_labeled_dataset.append((images, p_label.item(), cams))

        print(f'\n==> Statistics for epoch {epoch}: {num_cluster}/{dataset_n_class[args.dataset]} clusters')

        ari = adjusted_rand_score(p_labels, gt_labels)  # 伪标签在前, GT标签在后
        print('ARI 分数>>>>>>>>>> ', ari)
        if args.wandb_enabled:
            wandb.log({f'cluster': num_cluster, 'ARI': ari, 'Dataset Length': len(pseudo_labeled_dataset)})



        @torch.no_grad()
        def generate_center_features(labels, feat):
            centers = collections.defaultdict(list)

            for i, label in enumerate(labels):
                if int(label) == -1:
                    continue
                centers[int(labels[i])].append(feat[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        @torch.no_grad()
        def generate_instance_features(p_labels, feat, num_cluster, num_instances):
            indexes = np.zeros(num_cluster * num_instances)

            for i in range(num_cluster):
                index = [i + k * num_cluster for k in range(num_instances)]
                samples = np.random.choice(np.where(p_labels == i)[0], num_instances, True)
                indexes[index] = samples
            memory_features = feat[indexes]
            return memory_features

        print("==> Initialize global centroid features in the hybrid memory")
        cluster_features = generate_center_features(pseudo_labels, features)
        # text_cluster_features = generate_center_features(pseudo_labels[global_labels], text_features)

        memory = ClusterMemory(2048, num_cluster, temp=args.temp,momentum=args.momentum).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()
        ics_trainer.memory = memory

        # ics_trainer.new_labels = pseudo_labels  # 传入伪标签
        # ics_trainer.nums_class = num_cluster
        # ics_trainer.cluster_text_features = text_cluster_features

        # train_loader = get_train_loader(loaders)  # loader 集成了train test
        train_loader_ics = get_train_loader_ics(args, loaders.train_samples, args.height, args.width,
                                              args.batch_size, args.workers, args.num_instances,  400,
                                              trainset=pseudo_labeled_dataset, no_cam=False,
                                              train_transformer=transform_train)
        train_loader_ics.new_epoch()

        curr_lr = optimizer.param_groups[0]['lr']
        if args.wandb_enabled: wandb.log({f'Lr': curr_lr})
        print('=> Current Lr: {:.2e}'.format(curr_lr))

        # ************ 主要训练流程 ********************************
        ics_trainer.train_an_epoch(epoch,
                      train_loader_ics, # 训练数据
                      optimizer)

        print(f'Epoch {epoch} Training end\n')

        # Validation. >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            print('=> Epoch {} test: '.format(epoch + 1))
            eval_results = test(model, test_loader)
            mAP = eval_results[0]
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP,  best_mAP)
            if is_best: best_epoch = epoch

            save_checkpoint({
                'state_dict': model.state_dict(),
                 # 'cam_classifier_state_dict': cam_classifier.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                'R1': eval_results[1],
            }, is_best, fpath=osp.join(args.logs_dir, args.dataset, f'checkpoint_stage2_epoch.pth.tar'))

            print(f'rank1: {eval_results[1]:4.1%}, rank5: {eval_results[2]:4.1%}, rank10:{eval_results[3]:4.1%}, mAP: {eval_results[0]:4.1%}')

            if args.wandb_enabled:
                wandb.log({'R1': eval_results[1],
                          'mAP': eval_results[0]})

            print(f'\n * 结束 epoch {epoch:3d} 当前mAP: {mAP:5.1%} 最优mAP: {best_mAP:5.1%} 最优Epoch: {best_epoch}\n')

        scheduler.step()

        torch.cuda.empty_cache()  # 作用是？
        print('=> CUDA cache is released.')
        print('')

    '''
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_stage2.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    eval_results = test(model, test_loader)
    print('rank1: {:4.1%}, rank5: {:4.1%}, rank10:{:4.1%} , mAP: {:4.1%}'.format(
    eval_results[1], eval_results[2], eval_results[3], eval_results[0]))
    '''

    end_time = time.monotonic()
    dtime = timedelta(seconds=end_time - start_time)
    print('=> Task finished: {}'.format('CLIP_Stage2_ICS'))
    print('Stage2 running time: {}'.format(dtime))



