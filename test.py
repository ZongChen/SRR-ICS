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
import scipy
import sys
import time
import wandb

# sys.path.append(' ')
import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from clustercontrast.utils.osutils import str2bool
from clustercontrast.utils.prepare_optimizer import make_optimizer_2stage
from clustercontrast.utils.prepare_scheduler import create_scheduler
from sklearn.cluster import DBSCAN

from train.RN50 import make_model
from clustercontrast import datasets
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.cluster_labels import img_association
from clustercontrast.image_data_loader import Loaders
from clustercontrast.losses.cam_based_adversarial_loss import arcAdversarialLoss
from clustercontrast.losses.cosface_loss import ArcFaceLoss
from clustercontrast.losses.triplet_loss_stb import TripletLoss
from clustercontrast.utils.meters import AverageMeter
from clustercontrast.losses.softmax_loss import CrossEntropyLabelSmooth
from clustercontrast.models.cm import ClusterMemory, ClusterMemoryCenter
from clustercontrast.evaluators import test
from clustercontrast.evaluators import extract_vit_features
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance


from sklearn import metrics
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from torch.cuda import amp

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

start_epoch = best_mAP = 0

def generate_curriculum(cam_num=6, iters_per_stage=5):
    curriculum = []
    for stage in range(cam_num):
        current_cams = list(range(stage + 1))  # 摄像头0~stage
        curriculum.append(current_cams * iters_per_stage)
    return curriculum


def generate_staged_curriculum(cam_list, iters_per_stage=1):
    curriculum = []
    for stage in range(1, len(cam_list) + 1):
        current_cams = cam_list[:stage]  # 当前阶段使用的摄像头
        stage_data = []
        for _ in range(iters_per_stage):
            stage_data.extend(current_cams)  # 每个摄像头重复 epoch_per_stage 次
        curriculum.append(stage_data)

    print(f'curriculum as {curriculum}')
    return curriculum


def get_current_cam(epoch, curriculum, epoch_per_stage=5):
    # 计算当前应该使用curriculum中的哪个索引
    idx = epoch // epoch_per_stage
    # 确保不超出curriculum的范围
    idx = min(idx, len(curriculum) - 1)
    return curriculum[idx]


class TrainerIntra(object):
    def __init__(self,
                 args ,
                 encoder,
                 encoder_ema,
                 id_count_each_cameras,
                 cam_memorys=None,
                 new_labels=None,
                 nums_class = 4821):
        super(TrainerIntra, self).__init__()

        self.args = args
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.cam_memorys = cam_memorys

        self.new_labels = new_labels
        self.device = 'cuda'
        self.nums_class = nums_class
        self.xent_sup = CrossEntropyLabelSmooth(self.nums_class).cuda()
        print("label smooth on, numclasses:", self.nums_class)

        self.id_count_each_cameras = id_count_each_cameras

        self.margin = 0.3
        print("using triplet loss with margin:{}".format(self.margin))
        self.criterion_triplet = TripletLoss(self.margin, 'euclidean')  # default margin=0.3

        if self.args.dataset == 'market1501':
            self.epsilon = 0.8
            self.loss_weight =  0.6
            self.start_adv_epoch = 40
        elif self.args.dataset == 'dukemtmc':
            self.epsilon = 0.8
            self.loss_weight = 0.6  # all 0.8  market1501 0.6
            self.start_adv_epoch = 40
        elif self.args.dataset == 'msmt17':
            self.epsilon = 0.8   # 0.8 Best
            self.loss_weight = 0.0
            self.start_adv_epoch = 0  # 40
        else:
            self.epsilon = 0.8   # 0.8 Best
            self.loss_weight = 0.0
            self.start_adv_epoch = 0  # 40

        if self.args.dataset == 'market1501':
            self.camera_sequence = [3,1,4,0,5,2]
            self.camera_sequence.reverse()  # reversed order
        elif self.args.dataset == 'dukemtmc':
            self.camera_sequence = None
        elif self.args.dataset == 'msmt17':
            self.camera_sequence = list(range(15))
            # self.camera_sequence.reverse()


        # self.curriculum = generate_curriculum(len(id_count_each_cameras), iters_per_stage=1)
        self.curriculum = generate_staged_curriculum(self.camera_sequence, iters_per_stage=1)

        print("using intra hard loss weight :{}".format( self.loss_weight))

    def train_an_epoch(self,
                       epoch,
                       train_data_loader,
                       optimizer,
                       cameras):
        self.encoder.train()
        self.encoder_ema.train()

        batch_time = AverageMeter()
        losses_h = AverageMeter()
        # global_ID_losses = AverageMeter()

        end = time.time()
        scaler = amp.GradScaler()
        cameras = get_current_cam(epoch, self.curriculum)

        # for cam_index in curriculum:
        for cam_index in cameras:
            cam_loader = train_data_loader[cam_index]

            for i, inputs in enumerate(cam_loader):
                # inter_loss = torch.tensor(0.).to(self.device)

                # process inputs，每张图片由3个label构成，cam为摄像头ID，cam_pid为摄像头内ID，gID为全局ID
                imgs, _, cams, cam_pid, _ = self._parse_data(inputs)

                with amp.autocast(enabled=True):
                    # ============== model forward ================

                    if self.args.arch == 'CLIP':
                        x_glb, _ = self.encoder(imgs)  # 两种特征 x_glb->2048， f_proj->1024 用于文本计算
                    elif self.args.arch == 'agw':
                        x_glb, _, _ = self.encoder(imgs)  # 两种特征 x_glb->2048， f_proj->1024 用于文本计算

                    # call into Memory forward
                    loss_online = self.cam_memorys[cam_index](x_glb.to(self.device), cam_pid.long().to(self.device))
                    # inter_loss += loss_centroid

                    '''
                    with torch.no_grad():
                        if self.args.arch == 'CLIP':
                            features_ema, _ = self.encoder_ema(imgs)  # 两种特征 x_glb->2048， f_proj->1024 用于文本计算
                        elif self.args.arch == 'agw':
                            features_ema, _, _ = self.encoder_ema(imgs) # 两种特征 x_glb->2048， f_proj->1024 用于文本计算

                    loss_ema = self.cam_memorys[cam_index](features_ema, cam_pid.long().to(self.device))
                    '''
                    loss = loss_online  # + loss_ema

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    losses_h.update(loss.item())

                    self._update_ema_variables(self.encoder, self.encoder_ema, 0.999)

                    torch.cuda.synchronize()

                    batch_time.update(time.time() - end)
                    end = time.time()

                    if (i + 1) % 10 == 0:
                        print(f'cam {cam_index} ')
                        print(f'Stage1 Epoch: [{epoch}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})  Inter Loss {losses_h.val:.3f} ({losses_h.avg:.3f})')

                        if self.args.wandb_enabled:
                            wandb.log({'Stage1 Epoch': epoch,
                                       'Stage1 Loss': losses_h.avg})


    def _parse_data(self, inputs):
        ori_data = inputs

        # [img, gt_label, cam_id, -1, cam_pid, global_label, index]
        imgs = ori_data[0]
        cams = ori_data[2]  #
        cam_pid = ori_data[4]  # [4]是相机内ID
        global_label = ori_data[5] #  每个相机内图像的索引编号

        # self.new_labels -> ndarray，3262个伪标签
        # global label -> Tensor， 原始的128个全局标签
        # converted_label = torch.tensor(self.new_labels[global_label]).long()  # predicted label
        converted_label = 0

        return imgs.cuda(), converted_label, cams.cuda(), cam_pid.cuda(), global_label.cuda()

    def _update_ema_variables(self, model, ema_model, alpha):
        """
        更新动量编码器的权重
        """
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class TrainerCross(object):
    def __init__(self,
                 args,
                 encoder,
                 id_count_each_cameras,
                 cam_classifier=None,
                 memory=None,
                 pseudo_labels=None,
                 cam2accu=None,
                 pseudo2accu_mask=None,
                 nums_class=4821,
                 inter_cam_memory=None,
                 all_cam=6):
        super(TrainerCross, self).__init__()

        self.args = args
        self.encoder = encoder
        self.memory = memory
        self.cam_classifier = cam_classifier
        self.inter_cam_memory = inter_cam_memory


        self.pseudo_labels = pseudo_labels
        self.device = 'cuda'
        self.nums_class = nums_class
        self.xent_sup = CrossEntropyLabelSmooth(self.nums_class).cuda()
        print("label smooth on, numclasses:", self.nums_class)

        self.xent_intra = []

        self.id_count_each_cameras = id_count_each_cameras

        for i in self.id_count_each_cameras:
            self.xent_intra.append(CrossEntropyLabelSmooth(i).cuda())

        self.margin = 0.3
        print("using triplet loss with margin:{}".format(self.margin))
        self.criterion_triplet = TripletLoss(self.margin, 'euclidean')  # default margin=0.3


        if self.args.dataset == 'market1501':
            self.epsilon = 0.8
            self.loss_weight = 0.6
        elif self.args.dataset == 'dukemtmc':
            self.epsilon = 0.8
            self.margin_adv = 0.3
            self.loss_weight = 0.6  # all 0.8  market1501 0.6
        else:
            self.epsilon = 0.8  # 0.8 Best
            self.loss_weight = 0.6

        print("using intra hard loss weight :{}".format(self.loss_weight))

        # Loss
        self.classifier = NormalizedClassifier(feature_dim=2048, num_classes=652).cuda()

    def train_an_epoch(self,
                       epoch,
                       train_data_loader,
                       optimizer,
                       cameras,
                       train_iters,
                       print_freq=50):
        self.encoder.train()

        batch_time = AverageMeter()
        losses_h = AverageMeter()
        start_time = time.time()

        scaler = amp.GradScaler()

        for i in range(train_iters):  # 12936
            inputs = train_data_loader.next_one()  # load data
            cross_loss = torch.zeros(1, device=self.device, requires_grad=True)

            imgs, converted_labels, cams, cam_pid, g_label = self._parse_data(inputs)

            with amp.autocast(enabled=True):
                # ============== model forward ================
                if self.args.arch == 'CLIP':
                    x_glb, _ = self.encoder(imgs)  # 两种特征 x_glb->2048， f_proj->1024 用于文本计算
                elif self.args.arch == 'agw':
                    x_glb, _, pool = self.encoder(imgs)  # 两种特征 x_glb->2048， f_proj->1024 用于文本计算

                # Intra-Camera Supervised Block, Adversarial Block
                '''
                if epoch > self.args.start_adv_epoch:
                    percam_V = []
                    for ii in range(cameras):
                        num_instances = 32
                        outputs = self.cam_cluster_memorys[ii].features.detach().clone()  # ClusterMemory取出来
                        out_list = torch.chunk(outputs, num_instances + 1, dim=0)  # 切为33份
                        percam_V.append(out_list[0])  # Cluster特征

                    # cache the per-camera memory bank before each batch training
                    for ii in range(cameras):  # 每个Camera单独处理
                        target_cam = ii

                        if torch.nonzero(cams == target_cam).size(0) > 0:
                            percam_feat1 = x_glb[cams == target_cam]
                            percam_label = cam_pid[cams == target_cam]  # 因此这里的ID虽然Cam间重叠，但只计算每个Camera的。

                            loss_centroid, loss_instance = self.cam_cluster_memorys[ii](percam_feat1.to(self.device), percam_label.long().to(self.device), cam=True, use_instance=False)

                            # ICDL
                            cross_loss += (self.loss_weight * loss_centroid + (1 - self.loss_weight) * loss_instance)

                            cam_memory_label = torch.arange(self.id_count_each_cameras[ii]).long().to(self.device)  # ？？？
                            memo_trip_loss = self.criterion_triplet(percam_feat1,
                                                                    percam_V[ii].to(self.device),  # Cluster特征
                                                                    percam_V[ii].to(self.device),  # Cluster特征
                                                                    percam_label.long().to(self.device),
                                                                    cam_memory_label,
                                                                    cam_memory_label)
                            cross_loss += memo_trip_loss

                            TRI_LOSS = self.criterion_triplet(percam_feat1,
                                                              percam_feat1,
                                                              percam_feat1,
                                                              percam_label.long().to(self.device),
                                                              percam_label.long().to(self.device),
                                                              percam_label.long().to(self.device))
                            cross_loss += TRI_LOSS
                '''


                selected_idx = np.where(converted_labels >= 0)[0]  # 返回的是行索引
                if len(selected_idx) != 0:  # 索引有值的条件下
                    f_out = x_glb[selected_idx].to(self.device)  # 取对应索引值训练
                    labels = converted_labels[selected_idx].to(self.device)  # 伪标签

                    # Loss 1, 对比学习的损失
                    cross_loss = self.memory(f_out, labels, cam=False) + cross_loss

                    # 需要明确公式，约带来0.2的提升
                    TRI_LOSS = self.criterion_triplet(f_out,
                                                      f_out,
                                                      f_out,
                                                      labels,
                                                      labels,
                                                      labels)
                    cross_loss = TRI_LOSS + cross_loss

                    optimizer.zero_grad()
                    scaler.scale(cross_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                losses_h.update(cross_loss.item())

                torch.cuda.synchronize()
                batch_time.update(time.time() - start_time)


                if (i + 1) % print_freq == 0:
                    print(f'Stage2 Epoch: [{epoch}][{i + 1}/{train_iters}]\t'
                          f'Time {batch_time.val:.3f}\t'
                          f'Global ID Loss {losses_h.val:.3f} ({losses_h.avg:.3f})\t')

                    if self.args.wandb_enabled:
                        wandb.log({'Stage2 Epoch': epoch, 'Stage2 Loss': losses_h.avg})

    def _parse_data(self, inputs):
        ori_data = inputs

        # [img, gt_label, cam_id, -1, cam_pid, global_label, index]
        imgs = ori_data[0]
        cams = ori_data[2]  # 摄像头ID
        cam_pid = ori_data[4]  # [4]是相机内ID
        global_label = ori_data[5]  # 每个相机内图像的索引编号

        # self.pseudo_labels -> ndarray，3262个伪标签
        # global label -> Tensor， 原始的128个全局标签

        # 全局标签到伪标签转换
        converted_label = torch.tensor(self.pseudo_labels[global_label]).long()  # predicted label

        return imgs.cuda(), converted_label, cams.cuda(), cam_pid.cuda(), global_label.cuda()

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


def get_cluster_loader(loaders):
    pe_loader = loaders.propagate_loader  # 会生成全局标签
    return pe_loader



def create_model(args):
    if args.dataset == 'market1501':
        cam_num = 6
    elif args.dataset == 'msmt17':
        cam_num = 15
    elif args.dataset == 'dukemtmc':
        cam_num = 8
    else:
        raise Exception

    model = make_model(args, camera_num=cam_num)
    model.cuda()

    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


# ========= 基本工具 ==========
def evaluate(gt_labels, pred_labels):
    ari = adjusted_rand_score(gt_labels, pred_labels)
    nmi = normalized_mutual_info_score(gt_labels, pred_labels)
    return ari, nmi


def main_worker(args):
    # sys.stdout = Logger(osp.join(args.logs_dir, f'log_{args.dataset}.log'))
    print("==========\nArgs:{}\n==========".format(args))
    # Create datasets
    selected_idx = None
    new_labels = None


    loaders = Loaders(args, selected_idx, new_labels, learning_setting='semi_supervised')
    propagate_loader = loaders.propagate_loader
    print("==> Load intra-camera labeled dataset: ", args.dataset)
    test_loader = get_test_loader(args, loaders)

    id_count_each_cam = loaders.id_count_each_cam
    id_count_each_cam = np.array(id_count_each_cam)
    cameras = len(id_count_each_cam)
    print('  number of ID each camera: {}'.format(np.sum(id_count_each_cam)))
    print('  {} number of camera: {}'.format(args.dataset, cameras))

    model = create_model(args)  # Create model

    # Load trained stage1 model
    # model_path = osp.join('logs/train_curriculum_ema', 'CCAFL_intra30_model_stage2_epoch29.pth.tar')
    model_path = '/media/deep/Data/Share/1.Pytorch_Re-id_Cross_SSL/CCAFL/logs/train_ics/market1501/sota.pth.tar'
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    # print('载入模型训练准确度为 ', checkpoint['R1'])


    eval_results = test(model, test_loader)
    print('rank1: {:4.1%}, rank5: {:4.1%}, rank10:{:4.1%} , mAP: {:4.1%}'.format(
        eval_results[1], eval_results[2], eval_results[3], eval_results[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501', choices=['dukemtmc', 'market1501', 'msmt17'])
    parser.add_argument('-a', '--arch', type=str, default='CLIP', choices=['CLIP', 'agw'])

    # parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--num-instances', type=int, default=32)  # Instance对比的特征数

    # model
    parser.add_argument('--momentum', type=float, default=0.1, help="update momentum for the hybrid memory")
    parser.add_argument('--features', type=int, default=2048, help="feature dim")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--pooling-type', type=str, default='gem')

    #vit
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')

    # Stage1
    # parser.add_argument('--epochs_stage1', type=int, default=60)  # Text Stage
    parser.add_argument('--intra_epoch', type=int, default=40)
    parser.add_argument('--cross_epochs', type=int, default=60)
    parser.add_argument('--start_adv_epoch', type=int, default=0)
    # parser.add_argument('--end_adv_epoch', type=int, default=50)

    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05, help="temperature for scaling contrastive loss")
    parser.add_argument("--wandb_enabled", default=False, type=bool)

    # path
    working_dir = '/media/deep/SSD/Dataset_ReID'
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=working_dir)
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default='logs/')

    parser.add_argument('--market_path', type=str, default='/media/deep/SSD/Dataset_ReID/Market-1501-v15.09.15/')
    parser.add_argument('--duke_path', type=str, default='/media/deep/SSD/Dataset_ReID/DukeMTMC-reID/DukeMTMC-reID')
    parser.add_argument('--msmt_path', type=str, default='/media/deep/SSD/Dataset_ReID/MSMT17')

    # Loader 设置
    parser.add_argument('--class_per_batch', type=int, default=16)  # triplet sampling, number of IDs per batch16
    parser.add_argument('--track_per_class', type=int, default=8)  # triplet sampling, number of images per ID per batch

    parser.add_argument('--use_inter_camera', type=str2bool, default=False)
    parser.add_argument('--use-intra-hard', type=str2bool, default=True)

    # parser.add_argument('--use-intra-hard',  action="store_true")
    # parser.add_argument('--lossweight', type=float, default=0.0)
    parser.add_argument('--epsilon', type=float, default=0.0)

    main()
