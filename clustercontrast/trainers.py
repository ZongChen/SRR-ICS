from __future__ import print_function, absolute_import
import copy
import time
import torch
import wandb

from .losses.cam_based_adversarial_loss import arcAdversarialLoss
from .losses.cosface_loss import ArcFaceLoss
from .losses.triplet_loss_stb import TripletLoss
from .utils.meters import AverageMeter
from torch import amp
import numpy as np
import torch.nn.functional as F
# from clustercontrast.losses.original_mcnl_loss import MCNL_Loss_global
from clustercontrast.losses.softmax_loss import CrossEntropyLabelSmooth
from clustercontrast.losses.make_loss import make_loss


class TrainerFp16(object):
    """
    trainer with FP16 forwarding.
    """
    def __init__(self,
                 args ,
                 encoder,
                 id_count_each_cameras,
                 cam_classifier=None,
                 memory = None ,
                 cam_memorys=None,
                 new_labels=None,
                 cluster_text_features = None,
                 cam2accu = None,
                 pseudo2accu_mask = None,
                 cam_text_features = None,
                 nums_class = 4821,
                 inter_cam_memory = None, all_cam = 6):
        super(TrainerFp16, self).__init__()

        self.args = args
        self.encoder = encoder
        self.memory = memory
        self.cam_classifier = cam_classifier
        self.cluster_text_features = cluster_text_features
        self.cam_text_features = cam_text_features
        self.cam_memorys = cam_memorys
        self.inter_cam_memory = inter_cam_memory

        self.cam2accu = cam2accu
        self.pseudo2accu_mask = pseudo2accu_mask  #

        self.new_labels = new_labels
        self.device = 'cuda'
        self.nums_class = nums_class
        self.xent_sup = CrossEntropyLabelSmooth(self.nums_class).cuda()
        print("label smooth on, numclasses:", self.nums_class)

        self.xent_intra = []

        self.id_count_each_cameras = id_count_each_cameras

        self.text_loss_intra = True

        for i in self.id_count_each_cameras:
            self.xent_intra.append(CrossEntropyLabelSmooth(i).cuda())

        self.margin = 0.3
        print("using triplet loss with margin:{}".format(self.margin))
        self.criterion_triplet = TripletLoss(self.margin, 'euclidean')  # default margin=0.3

        #############################  ADV  ##########################################
        #epsilon (float): a trade-off hyper-parameter.market 0.8 duke 0.8 msmt 1.0
        #margin (float): margin for the arcface loss. market 0.7 duke 0.3 msmt 0.5

        if self.args.dataset == 'market1501':
            self.epsilon = 0.8
            self.margin_adv = 0.7
            self.loss_weight =  0.6
            self.inter_text_loss_epoch = 10
            self.start_adv_epoch = 40
            self.end_adv_epoch = 50
        elif self.args.dataset == 'dukemtmc':
            self.epsilon = 0.8
            self.margin_adv = 0.3
            self.loss_weight = 0.6  # all 0.8  market1501 0.6
            self.inter_text_loss_epoch = 40
            self.start_adv_epoch = 40
            self.end_adv_epoch = 60
        else:
            self.epsilon = 0.8   # 0.8 Best
            self.margin_adv = 0.5
            self.loss_weight = 0.0
            self.inter_text_loss_epoch = 40
            self.start_adv_epoch = 0  # 40
            self.end_adv_epoch = 60
        print("start adv epoch: {}".format(self.start_adv_epoch))
        print("using intra hard loss weight :{}".format( self.loss_weight))
        print("using inter text loss with epoch: {}".format(self.inter_text_loss_epoch))
        print("using adv loss with epsilon: {} and margin:{}".format(self.epsilon, self.margin_adv))

        # Loss
        self.criterion_cam = ArcFaceLoss(scale=16., margin=0)
        self.criterion_adv = arcAdversarialLoss(scale=16., epsilon=self.epsilon, margin=self.margin_adv, tau=0.2)

    def train_an_epoch(self,
              epoch,
              train_data_loader,
              optimizer,
              optimizer_cc,
              print_freq=10,
              train_iters=400,
              camera = 15 ,
              intra_epoch = 5):
        self.encoder.train()
        self.cam_classifier.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_h = AverageMeter()
        cam_losses = AverageMeter()
        global_ID_losses = AverageMeter()

        end = time.time()

        # amp fp16 training
        scaler = amp.GradScaler()
        for i in range(train_iters):  # 12936
            inputs = train_data_loader.next_one()  # load data

            data_time.update(time.time() - end)
            intra_loss = torch.tensor(0.).to(self.device)
            inter_loss = torch.tensor(0.).to(self.device)

            # process inputs，每张图片由3个label构成，cam为摄像头ID，cam_pid为摄像头内ID，gID为全局ID
            # 1. 2.converted_labels Inter阶段使用 3. 4. 5.
            imgs, converted_labels, cams, cam_pid, g_label = self._parse_data(inputs)

            with amp.autocast(device_type='cuda', enabled=True):
                # ============== model forward ================

                # f_proj用于文本计算
                x_glb, f_proj = self.encoder(imgs)  # 两种特征 x_glb->2048， f_proj->1024

                # Inter-Cam Global ID Classifier. 惩罚
                cam_logits = self.cam_classifier(x_glb.detach())  # 摄像头分类器， detach是希望不参与梯度回传，即只优化单独的cam_classifier
                # [b, 3262]， 代表着行人特征被分为3262个ID中某一个的可能性

                cam_index = cams.long().to('cpu')

                # cam2accu -> (6, 3262)  cam_index -> 128
                pos_cam_mask = self.cam2accu[cam_index].to(self.device)  # 后续进行对抗训练

                # 摄像头预测值 与 全局摄像头ID，从而不陷于局部鉴别力
                gID_loss = self.criterion_cam(cam_logits, g_label.long().to(self.device))  # ArcFaceLoss

                optimizer_cc.zero_grad()
                scaler.scale(gID_loss).backward()
                scaler.step(optimizer_cc)
                scaler.update()
                global_ID_losses.update(gID_loss.item())

                # intra-camera training period --------------------------------------------------------------------
                if epoch < intra_epoch:  # 5
                    percam_V = []

                    for ii in range(camera):
                        num_instances = 32
                        outputs = self.cam_memorys[ii].features.detach().clone()  # ClusterMemory取出来
                        out_list = torch.chunk(outputs, num_instances + 1, dim=0)  # 切为33份
                        percam_V.append(out_list[0])  # 0是Cluster特征

                    # cache the per-camera memory bank before each batch training
                    for ii in range(camera):  # 每个Camera单独处理
                        target_cam = ii

                        if torch.nonzero(cams == target_cam).size(0) > 0:
                            # print('size ', torch.nonzero(cams == target_cam).size(0))
                            percam_feat1 = x_glb[cams == target_cam]
                            percam_feat2 = f_proj[cams == target_cam]
                            percam_label = cam_pid[cams == target_cam]  # 因此这里的ID虽然Cam间重叠，但只计算每个Camera的。

                            loss_centroid, loss_instance  = self.cam_memorys[ii](percam_feat1.to(self.device),
                                                                                 percam_label.long().to(self.device),
                                                                                 cam=True)


                            # ICDL
                            intra_loss += (self.loss_weight * loss_centroid + (1 - self.loss_weight) * loss_instance)

                            if self.text_loss_intra:
                                logits = percam_feat2 @ self.cam_text_features[ii].t()
                                loss_i2tc = self.xent_intra[ii](logits , percam_label.long().to(self.device))

                                intra_loss += loss_i2tc

                            cam_memory_label = torch.arange(self.id_count_each_cameras[ii]).long().to(self.device)  # ？？？

                            # 用的什么损失
                            memo_trip_loss = self.criterion_triplet(percam_feat1,
                                                                    percam_V[ii].to(self.device),
                                                                    percam_V[ii].to(self.device),
                                                                    percam_label.long().to(self.device),
                                                                    cam_memory_label,
                                                                    cam_memory_label)
                            intra_loss += memo_trip_loss

                            TRI_LOSS = self.criterion_triplet(percam_feat1,
                                                              percam_feat1,
                                                              percam_feat1,
                                                              percam_label.long().to(self.device),
                                                              percam_label.long().to(self.device),
                                                              percam_label.long().to(self.device))
                            intra_loss += TRI_LOSS

                    optimizer.zero_grad()

                    scaler.scale(intra_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    cam_losses.update(intra_loss.item())
                    torch.cuda.synchronize()
                    batch_time.update(time.time() - end)

                    end = time.time()
                    if (i + 1) % print_freq == 0:
                        print(f'Epoch: [{epoch}][{i + 1}/{train_iters}]\t'
                              f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              f'Intra Loss {cam_losses.val:.3f} ({cam_losses.avg:.3f})\t'
                              f'Global ID Loss {global_ID_losses.val:.3f} ({global_ID_losses.avg:.3f})\t')

                        if self.args.wandb_enabled:
                            wandb.log({'Intra Loss':cam_losses.avg,
                                       'Global ID Loss': global_ID_losses.avg})
                else:
                    # epoch >= intra_epoch
                    selected_idx = np.where(converted_labels >= 0)[0]  # 返回的是行索引

                    if len(selected_idx) != 0:  # 索引有值的条件下
                        # 取对应索引值训练
                        f_out = x_glb[selected_idx].to(self.device)
                        labels = converted_labels[selected_idx].to(self.device)  # 伪标签
                        # cams = cams[selected_idx].to(self.device)
                        f_proj = f_proj[selected_idx].to(self.device)

                        # Loss 1
                        inter_loss += self.memory(f_out,
                                                  labels,
                                                  cam=False)

                        if epoch >= self.inter_text_loss_epoch:  # 40
                            logits = f_proj @ self.cluster_text_features.t()
                            loss_itc = self.xent_sup(logits, labels)
                            # Loss 2
                            inter_loss  += loss_itc

                        # Loss 3, 论文中并未提及
                        TRI_LOSS = self.criterion_triplet(f_out,
                                                          f_out,
                                                          f_out ,
                                                          labels,
                                                          labels ,
                                                          labels)
                        inter_loss  += TRI_LOSS

                        # 中间加入对抗训练
                        if self.args.start_adv_epoch <= epoch < self.args.end_adv_epoch:  # Adversarial
                            print('---Adversarial')
                            labels_index = labels.cpu()

                            pos_mask = self.pseudo2accu_mask[labels_index].cuda()  # 伪标签之间的正样本掩码

                            mask_pos_cam = pos_mask - pos_cam_mask[selected_idx].cuda()  # 当前batch中已选样本的摄像头掩码

                            the_mask = mask_pos_cam < 0
                            mask_pos_cam[the_mask] = 0

                            adv_mask = copy.deepcopy(mask_pos_cam)
                            adv_mask.scatter_(1, g_label[selected_idx].data.view(-1, 1).long(), 1)

                            # 分类器预测值
                            all_new_pred_cam = self.cam_classifier(x_glb).to(self.device)  # 用到cam_classifier
                            new_pred_cam = all_new_pred_cam[selected_idx]

                            cams_index = cams.cpu()
                            selected_pos_cam_mask = self.cam2accu[cams_index].cuda()

                            # Loss 4 对抗损失
                            ICAL_loss = self.criterion_adv(new_pred_cam.to(self.device),
                                                          g_label[selected_idx].long().to(self.device),
                                                          adv_mask.to(self.device),
                                                          selected_pos_cam_mask,
                                                          self.cam2accu)
                            inter_loss += ICAL_loss
                    else:
                        print('-----------------len(selected_idx) != 0----------------------')

                    optimizer.zero_grad()
                    scaler.scale(inter_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    losses_h.update(inter_loss.item())

                    torch.cuda.synchronize()
                    # print log
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if (i + 1) % print_freq == 0:
                        print('Epoch: [{}][{}/{}]\t'
                              'Time {:.3f} ({:.3f})\t'
                              'Inter Loss {:.3f} ({:.3f})\t'
                              'Global ID Loss {:.3f} ({:.3f})\t'
                              .format(epoch, i + 1, train_iters,
                                      batch_time.val, batch_time.avg,
                                      losses_h.val, losses_h.avg,
                                      global_ID_losses.val, global_ID_losses.avg))

                        if self.args.wandb_enabled:
                            wandb.log({'Inter Loss':losses_h.avg,
                                       'Global ID Loss': global_ID_losses.avg})


    def _parse_data(self, inputs):
        ori_data = inputs

        # [img, gt_label, cam_id, -1, cam_pid, global_label, index]
        imgs = ori_data[0]
        cams = ori_data[2]  #
        cam_pid = ori_data[4]  # [4]是相机内ID
        global_label = ori_data[5] #  每个相机内图像的索引编号

        # self.new_labels -> ndarray，3262个伪标签
        # global label -> Tensor， 原始的128个全局标签
        # 作用
        converted_label = torch.tensor(self.new_labels[global_label]).long()  # predicted label

        return imgs.cuda(), converted_label, cams.cuda(), cam_pid.cuda(), global_label.cuda()


