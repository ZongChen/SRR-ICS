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

sys.path.append(' ')
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import TrainerFp16
from clustercontrast.evaluators import test, extract_text_features
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.cluster_labels import img_association
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.evaluators import extract_vit_features, extract_vit_features_for_isc
from clustercontrast.utils.label_refinement_ics import compute_ics_ckrnn_jaccard_distance

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


def get_cluster_loader(loaders):
    pe_loader = loaders.propagate_loader  # 会生成全局标签
    return pe_loader

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
def do_train_intra_inter_stage(args,
                     model,
                     loaders,  # 训练Loader
                     test_loader,  # 测试Loader
                     optimizer,
                     optimizer_cc,
                     scheduler,
                     id_count_each_cam,  # ???
                     cameras,
                     cam_classifier
                     ):
    print('\nstart stage2 training---------------------------------')  # Inter Camera

    global  best_mAP
    best_mAP = 0
    start_time = time.monotonic()
    text_features = extract_text_features(model, loaders.propagate_loader)  # 后面哪里用到

    #########################################################
    accu2cam = []  # 存储每个摄像头的样本索引范围（np.arange）
    cam2accu = []  # 存储每个摄像头的二进制掩码（torch.Tensor）.生成一个掩码矩阵 cam2accu，用于标识每个样本属于哪个摄像头
    # cam2accu_unique = []
    before = 0
    after = 0

    # 生成摄像头掩码 cam2accu
    for cam_i in range(loaders.cam_num):
        if cam_i < loaders.cam_num:
            after += id_count_each_cam[cam_i]
        accu2cam.append(np.arange(before, after))

        if cam_i < loaders.cam_num:
            before += id_count_each_cam[cam_i]
        cam_mask = torch.zeros(sum(id_count_each_cam))

        cam2accu.append(cam_mask.scatter_(0, torch.LongTensor(accu2cam[cam_i]), 1))  # Create Mask

        true_cam_mask = torch.zeros(loaders.cam_num)
        # cam2accu_unique.append(true_cam_mask.scatter_(0,torch.LongTensor([cam_i]),1))

    # cam2accu_unique = torch.stack(cam2accu_unique, dim=0)
    cam2accu = torch.stack(cam2accu, dim=0)  # 表示为[6, 3262]， 即每个摄像头中，全局3262个是来自于哪个摄像头

    print('\n', '*'*10, 'Create TrainerFp16')
    trainer = TrainerFp16(args,
                          model,
                          id_count_each_cam)
    trainer.cam_classifier = cam_classifier  # NormalizedClassifier

    # Stage2 Training
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch} Training start\n')
        print('==> Create inter camera labels ')

        # select & cluster images as training set of this epochs
        propagate_loader = get_cluster_loader(loaders)

        # Part D, Part 1. 这段代码应是用了连通分量算法，只是加了约束
        # 连通 in each epoch
        # 完成特征提取，并进行标签关联 -------------------------------------------------------------
        result = img_association(args,
                                 model,
                                 propagate_loader,
                                 id_count_each_cam)

        # 通过Label Association获得伪标签，后续步骤与无监督类似
        pseudo_labels, features, global_labels, cam_ids, intra_ids, new_IDs, gt_labels = result

        if args.using_offset:
            all_cams_tensor =  torch.from_numpy(cam_ids)
            features, _ = cams_offset(features, all_cams_tensor)  # ------------------ offset

        # ARI
        # ari_value = adjusted_rand_score(pseudo_labels, gt_labels)
        # print('ari->>>', ari_value)

        # 通过伪标签，获得聚类个数
        # 有-1则减1
        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        ######################################################################################
        # pseudo2accu = collections.defaultdict()
        '''
        pseudo2accu_list = []  # 用于收集每个伪类对应的 mask
        # pid2cam = []
        # pid2cam_unique = []
        correct_num = 0

        # pseudo2accu_mask[i] 是一个 二值向量（mask），表示在当前 batch 中，所有属于伪标签 i 的图像位置，并且这个聚类是“可靠”的（即只包含一个真实 ID 时才保留）
        for i in np.unique(pseudo_labels):
            # pseudo2accu[i] = np.where(pseudo_labels == i)  # 记录下每一个聚类伪标签 i 对应的图像下标索引。用于建立后面的掩码

            if i != -1:
                pseudo2accu_tensor = torch.zeros(sum(id_count_each_cam))  # 对某一个伪标签 i 的所有图像建立的一个 0-1 掩码张量

                # 表示该伪标签 i 覆盖了哪些图像
                pseudo2accu_tensor.scatter_(0, torch.LongTensor(np.where(pseudo_labels == i)[0]), 1)  # PyTorch 中的一个原地操作（in-place operation），用于将指定值填充到目标张量的指定位置。

                # 检测聚类后的标签是否存在噪声 用于评价聚类伪标签的准确性。
                if len(np.unique(new_IDs[torch.where(pseudo2accu_tensor == 1)[0]])) == 1:
                    correct_num += 1  # 如果聚到一起的图像对应的 new_IDs 都一样，那说明这类是纯的。

                pseudo2accu_list.append(pseudo2accu_tensor)

        pseudo2accu_list = torch.stack(pseudo2accu_list, dim=0)  # [num_pseudo_labels, batch_size]
        print("do_train_stage2  -->>>  correct/伪标签个数:", correct_num, '/', len(np.unique(pseudo_labels)))

        if args.wandb_enabled:
            wandb.log({'S2 correct_num': correct_num,
                       'S2 pseudo num':len(np.unique(pseudo_labels)),
                       'S2 Epoch': epoch})
        '''

        ########################################################################################

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

        # Only for 5 epochs.
        if epoch < args.intra_epoch:  # intra_epoch set as 5, Part D Part 2.
            print("\nStrat intra-camera contrast learning")
            cam_cluster_features = []
            cam_text_cluster_features = []
            cam_memorys_features = []

            # 为每个Camera，生成Camera内的Cluster和Instance特征
            for c in np.unique(cam_ids):  # all_cams: 12936 ndarray
                idx = np.where(cam_ids == c)[0]

                cam_label = torch.from_numpy(intra_ids[idx])  # 摄像头内标签
                cam_features = features[idx]  # 摄像头内特征
                cam_text_features = text_features[idx]

                # center features
                cluster_feat = generate_center_features(cam_label, cam_features)  # Shape->
                cam_cluster_features.append(cluster_feat)
                cam_text_cluster_features.append(generate_center_features(cam_label, cam_text_features))

                # instance features
                # Shape->
                instance_feat = generate_instance_features(cam_label,
                                                           cam_features,
                                                           id_count_each_cam[c],
                                                           args.num_instances)  # 32
                cam_memorys_features.append(instance_feat)  # 为什么有num_instances
                del cam_label, cam_features

            cam_memorys = []
            for i in range(cameras):
                cam_memorys.append(ClusterMemory(2048,
                                                 id_count_each_cam[i],
                                                 temp=args.temp,
                                                 momentum=args.momentum,
                                                 num_instances=args.num_instances).cuda())

            for i, cam_memory in enumerate(cam_memorys):
                if args.use_intra_hard:
                    # 将Cluster Feature和Instance Feature进行拼接
                    temp = torch.cat([cam_cluster_features[i], cam_memorys_features[i]], dim=0)
                    cam_memory.features = F.normalize(temp, dim=1).cuda()
                else:
                    cam_memory.features = F.normalize(cam_cluster_features[i], dim=1).cuda()

            trainer.cam_text_features = cam_text_cluster_features  # Only used in intra
            trainer.cam_memorys = cam_memorys  # 只在intra阶段使用

        print("==> Initialize global centroid features in the hybrid memory")
        cluster_features = generate_center_features(pseudo_labels[global_labels], features)
        text_cluster_features = generate_center_features(pseudo_labels[global_labels], text_features)

        memory = ClusterMemory(2048, num_cluster, temp=args.temp,momentum=args.momentum).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()
        trainer.memory = memory

        trainer.new_labels = pseudo_labels  # 传入伪标签
        trainer.nums_class = num_cluster
        trainer.cluster_text_features = text_cluster_features

        # 注意这两个
        trainer.cam2accu = cam2accu  # [6, 3262]， cam2accu, 对抗阶段用
        # trainer.pseudo2accu_mask = pseudo2accu_list  # 对抗阶段用

        train_loader = get_train_loader(loaders)  # loader 集成了train test

        curr_lr = optimizer.param_groups[0]['lr']
        print('=> Current Lr: {:.2e}'.format(curr_lr))

        # ************ 主要训练流程 ********************************
        trainer.train_an_epoch(epoch,
                      train_loader, # 训练数据
                      optimizer,
                      optimizer_cc,
                      print_freq=args.print_freq,
                      train_iters=args.iters,
                      camera=cameras,
                      intra_epoch=args.intra_epoch,
                      )

        print(f'Epoch {epoch} Training end\n')

        # Validation. >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            print('=> Epoch {} test: '.format(epoch + 1))
            eval_results = test(model, test_loader)
            mAP = eval_results[0]
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP,  best_mAP)

            save_checkpoint({
                'state_dict': model.state_dict(),
                 # 'cam_classifier_state_dict': cam_classifier.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                'R1': eval_results[1],
            }, is_best, fpath=osp.join(args.logs_dir, args.dataset, f'model_stage2_epoch{epoch}.pth.tar'))

            print('rank1: {:4.1%}, rank5: {:4.1%}, rank10:{:4.1%} , mAP: {:4.1%}'.format(
                eval_results[1], eval_results[2], eval_results[3], eval_results[0]))

            if args.wandb_enabled:
                wandb.log({'R1': eval_results[1],
                          'mAP': eval_results[0]})

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        scheduler.step()

        torch.cuda.empty_cache()  # 作用是？
        print('=> CUDA cache is released.')
        print('')

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_stage2.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    eval_results = test(model, test_loader)
    print('rank1: {:4.1%}, rank5: {:4.1%}, rank10:{:4.1%} , mAP: {:4.1%}'.format(
    eval_results[1], eval_results[2], eval_results[3], eval_results[0]))

    end_time = time.monotonic()
    dtime = timedelta(seconds=end_time - start_time)
    print('=> Task finished: {}'.format('CLIP_Stage2_ICS'))
    print('Stage2 running time: {}'.format(dtime))



