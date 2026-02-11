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

from clustercontrast.losses.softmax_loss import CrossEntropyLabelSmooth


class TrainerICS(object):
    """
    trainer with FP16 forwarding.
    """

    def __init__(self,
                 args,
                 encoder,
                 id_count_each_cameras,
                 cam_classifier=None,
                 memory=None,
                 cam_memorys=None,
                 new_labels=None,
                 cluster_text_features=None,
                 cam2accu=None,
                 pseudo2accu_mask=None,
                 cam_text_features=None,
                 nums_class=4821,
                 inter_cam_memory=None, all_cam=6):
        super(TrainerICS, self).__init__()

        self.args = args
        self.encoder = encoder
        self.memory = memory

        self.new_labels = new_labels
        self.device = 'cuda'
        self.nums_class = nums_class


        self.xent_intra = []

        self.margin = 0.3
        print("using triplet loss with margin:{}".format(self.margin))
        self.criterion_triplet = TripletLoss(self.margin, 'euclidean')  # default margin=0.3

        if self.args.dataset == 'market1501':
            self.epsilon = 0.8
            self.loss_weight = 0.6
        elif self.args.dataset == 'dukemtmc':
            self.epsilon = 0.8
            self.loss_weight = 0.6  # all 0.8  market1501 0.6
        else:
            self.epsilon = 0.8  # 0.8 Best
            self.loss_weight = 0.0

        print("using intra hard loss weight :{}".format(self.loss_weight))


    def train_an_epoch(self,
                       epoch,
                       train_data_loader,
                       optimizer):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_h = AverageMeter()

        end = time.time()

        scaler = amp.GradScaler()
        for i in range(self.args.iters):  # 12936
            inputs = train_data_loader.next()  # load data

            data_time.update(time.time() - end)

            imgs, fname, pid, cam_pid, g_label = inputs

            with amp.autocast(device_type='cuda', enabled=True):
                imgs = imgs.cuda()
                pid = pid.cuda()
                x_glb, f_proj = self.encoder(imgs)

                inter_loss = self.memory(x_glb, pid)
                # inter_loss += self.criterion_triplet(x_glb, x_glb, x_glb, pid, pid, pid) * 1.6

                optimizer.zero_grad()
                scaler.scale(inter_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                losses_h.update(inter_loss.item())

                torch.cuda.synchronize()

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % self.args.print_freq == 0:
                    print(f'Epoch: [{epoch}][{i + 1}/{self.args.iters}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          f'Inter Loss {losses_h.val:.3f} ({losses_h.avg:.3f})')

                    if self.args.wandb_enabled:
                        wandb.log({'Inter Loss': losses_h.avg})
