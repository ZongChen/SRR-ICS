from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import scipy

import torch
from torch import amp
from torch.utils.data import DataLoader

import random
import copy
import tqdm


from clustercontrast.utils.meters import AverageMeter

from .evaluation_metrics import cmc, mean_ap
from .utils import to_torch
from sklearn import metrics as sk_metrics

from typing import Tuple, Union, List





def extract_text_features(model, propagate_loader):
    model.eval()
    print('Start get text Stage1 features...')
    text_features = []
    with torch.no_grad():
        for c, (images,label,cams,_,cam_label,g_label,_) in enumerate(tqdm.tqdm(propagate_loader)):
            with amp.autocast(device_type='cuda', enabled=True):
                text_feature = model(label=g_label, get_text=True)
                #######################################################################
                text_features.append(text_feature.cpu())
    get_text_features = torch.cat(text_features, dim=0).cuda()
    print('Text  features: shape= {}'.format(get_text_features.shape))  # 12936 X 2048
    return get_text_features


def extract_vit_features_for_isc(model, data_loader):
    model.eval()
    print('extract_vit_features_for_isc ->>> Start Inference...')
    features, accu_labels, id_labels, cam_id_labels, intra_cam_labels = [], [], [], [], []

    with torch.no_grad():
        for c, (images, label, cams, _, cam_label, accu_label, _) in enumerate(tqdm.tqdm(data_loader)):
            images = to_torch(images).cuda()
            with amp.autocast(device_type='cuda', enabled=True):
                feat = model(images)

                # Tensor加到列表
                features.append(feat.cpu())
                id_labels.append(label)  # ground_truth
                cam_id_labels.append(cams)  # Cam ID
                intra_cam_labels.append(cam_label)  # Intra Label
                accu_labels.append(accu_label)  # 全局累积标签，因为每个摄像头下共有[652, 541, 694, 241, 576, 558]行人ID，从0开始映射到3261，一共3262个全局标签

    get_features_tensor = torch.cat(features, dim=0)  # tensor

    # Tensor列表 to numpy
    features_np = torch.cat(features, dim=0)
    id_labels = torch.cat(id_labels, dim=0)
    cam_id_labels = torch.cat(cam_id_labels, dim=0)
    intra_cam_labels = torch.cat(intra_cam_labels, dim=0)
    accu_labels = torch.cat(accu_labels, dim=0)

    print('  features: shape= {}'.format(features_np.shape))  # 12936 X 2048

    return get_features_tensor, features_np.numpy(), id_labels, cam_id_labels, intra_cam_labels, accu_labels.numpy()





def extract_vit_features(
    model: torch.nn.Module,
    propagate_loader: DataLoader,
    get_gt_label: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    model.eval()
    print('extract_vit_features ->>> Start Inference...')
    features, accu_labels, id_labels, cam_id_labels, intra_cam_labels, gt_labels = [], [], [], [], [], []

    with torch.no_grad():
        for c, (images, label, cams, _, cam_label, accu_label, gt_label) in enumerate(tqdm.tqdm(propagate_loader)):
            images = to_torch(images).cuda()
            with amp.autocast(device_type='cuda', enabled=True):
                # feat = model(images, cam_label=cams)  # .detach().cpu().numpy() #[xx,128]
                feat = model(images)

                # Tensor加到列表
                features.append(feat.cpu())
                id_labels.append(label)  # ground_truth
                cam_id_labels.append(cams)  # Cam ID
                intra_cam_labels.append(cam_label)  # Intra Label
                accu_labels.append(accu_label)  # 全局标签，因为每个摄像头下共有[652, 541, 694, 241, 576, 558]行人ID，从0开始映射到3261，一共3262个全局标签
                gt_labels.append(gt_label)

    get_features_tensor = torch.cat(features, dim=0)  # tensor

    # Tensor列表 to numpy
    features_np = torch.cat(features, dim=0).numpy()
    id_labels = torch.cat(id_labels, dim=0).numpy()
    cam_id_labels = torch.cat(cam_id_labels, dim=0).numpy()
    intra_cam_labels = torch.cat(intra_cam_labels, dim=0).numpy()
    accu_labels = torch.cat(accu_labels, dim=0).numpy()
    gt_labels = torch.cat(gt_labels, dim=0)

    print('  features: shape= {}'.format(features_np.shape))  # 12936 X 2048

    if get_gt_label:
        return get_features_tensor, features_np, id_labels, cam_id_labels, intra_cam_labels, accu_labels, gt_labels
    else:
        return get_features_tensor, features_np, id_labels, cam_id_labels, intra_cam_labels, accu_labels

'''
def extract_cnn_feature(model, inputs, mode):
    inputs = to_torch(inputs).cuda()
    # inputs1 = inputs
    # print(inputs)
    outputs = model(inputs, inputs, modal=mode)
    # outputs = model(inputs, inputs)

    # outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs
'''

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_features(model, data_loader, print_freq=50, flip=True, mode=0):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs, mode)
            flip = fliplr(imgs)
            # print(flip)
            outputs_flip = extract_cnn_feature(model, flip, mode)

            for fname, output, output_flip, pid in zip(fnames, outputs, outputs_flip, pids):
                features[fname] = (output.detach() + output_flip.detach()) / 2.0
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _, _, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _, _, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _, _, _, _ in query]
        gallery_ids = [pid for _, pid, _, _, _, _ in gallery]
        query_cams = [cam for _, _, cam, _, _, _ in query]
        gallery_cams = [cam for _, _, cam, _, _, _ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True), }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k - 1]))
    return cmc_scores['market1501'], mAP


class CatMeter:
    '''
    Concatenate Meter for torch.Tensor
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = torch.cat([self.val, val], dim=0)

    def get_val(self):
        return self.val

    def get_val_numpy(self):
        return self.val.data.cpu().numpy()


def test(network, loaders):
    network.eval()

    # meters
    query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

    # init dataset

    # compute query and gallery features
    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            for data in tqdm.tqdm(loader):  # show progress bar
                # compute feautres

                # images, pids, cids = data
                images = to_torch(data[0]).cuda()
                pids = to_torch(data[1]).cuda()
                cids = to_torch(data[2].sub(1)).cuda()

                #features = network(images,cam_label = cids)

                features = network(images)
                # save as query features
                if loader_id == 0:
                    query_features_meter.update(features.data)
                    query_pids_meter.update(pids)
                    query_cids_meter.update(cids)
                # save as gallery features
                elif loader_id == 1:
                    gallery_features_meter.update(features.data)
                    gallery_pids_meter.update(pids)
                    gallery_cids_meter.update(cids)

        # ------------------ 修改开始 ------------------
        # 1. 获取所有需要的 Numpy 数据
        query_features = query_features_meter.get_val_numpy()
        gallery_features = gallery_features_meter.get_val_numpy()

        # 获取标签和摄像头ID (原本代码里是在 PersonReIDMAP 调用里直接 get 的，这里提前取出来)
        query_pids = query_pids_meter.get_val_numpy()
        query_cids = query_cids_meter.get_val_numpy()
        gallery_pids = gallery_pids_meter.get_val_numpy()
        gallery_cids = gallery_cids_meter.get_val_numpy()

        # 2. 插入保存逻辑
        extract_mat = True
        if extract_mat:
            result = {
                'gallery_f': gallery_features,
                'gallery_label': gallery_pids,
                'gallery_cam': gallery_cids,
                'query_f': query_features,
                'query_label': query_pids,
                'query_cam': query_cids
            }
            # 使用 scipy 保存
            scipy.io.savemat('./', result)
            print(f'Reference result saved, ready for visualization.')
        # ------------------ 修改结束 ------------------

        # compute mAP and rank@k
        # 这里直接使用上面提取好的变量，代码更整洁
        result = PersonReIDMAP(
            query_features, query_cids, query_pids,
            gallery_features, gallery_cids, gallery_pids, dist='cosine')

    return result.mAP, result.CMC[0], result.CMC[4], result.CMC[9], result.CMC[19]


class PersonReIDMAP:
    '''
    Compute Rank@k and mean Average Precision (mAP) scores
    Used for Person ReID
    Test on MarKet and Duke
    '''

    def __init__(self,
                 query_feature,
                 query_cam,
                 query_label,
                 gallery_feature,
                 gallery_cam,
                 gallery_label,
                 dist):
        '''
        :param query_feature: np.array, bs * feature_dim
        :param query_cam: np.array, 1d
        :param query_label: np.array, 1d
        :param gallery_feature: np.array, gallery_size * feature_dim
        :param gallery_cam: np.array, 1d
        :param gallery_label: np.array, 1d
        '''

        self.query_feature = query_feature
        self.query_cam = query_cam
        self.query_label = query_label
        self.gallery_feature = gallery_feature
        self.gallery_cam = gallery_cam
        self.gallery_label = gallery_label

        assert dist in ['cosine', 'euclidean']
        self.dist = dist

        # normalize feature for fast cosine computation
        if self.dist == 'cosine':
            self.query_feature = self.normalize(self.query_feature)
            self.gallery_feature = self.normalize(self.gallery_feature)

        APs = []
        CMC = []
        for i in range(len(query_label)):
            AP, cmc = self.evaluate(self.query_feature[i], self.query_cam[i], self.query_label[i],
                                    self.gallery_feature, self.gallery_cam, self.gallery_label)
            APs.append(AP)
            CMC.append(cmc)
            # print('{}/{}'.format(i, len(query_label)))

        self.APs = np.array(APs)
        self.mAP = np.mean(self.APs)

        min_len = 99999999
        for cmc in CMC:
            if len(cmc) < min_len:
                min_len = len(cmc)
        for i, cmc in enumerate(CMC):
            CMC[i] = cmc[0: min_len]
        self.CMC = np.mean(np.array(CMC), axis=0)

    def compute_AP(self, index, good_index):
        '''
        :param index: np.array, 1d
        :param good_index: np.array, 1d
        :return:
        '''

        num_good = len(good_index)
        hit = np.in1d(index, good_index)
        index_hit = np.argwhere(hit == True).flatten()

        if len(index_hit) == 0:
            AP = 0
            cmc = np.zeros([len(index)])
        else:
            precision = []
            for i in range(num_good):
                precision.append(float(i + 1) / float((index_hit[i] + 1)))
            AP = np.mean(np.array(precision))
            cmc = np.zeros([len(index)])
            cmc[index_hit[0]:] = 1

        return AP, cmc

    def evaluate(self, query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label):
        '''
        :param query_feature: np.array, 1d
        :param query_cam: int
        :param query_label: int
        :param gallery_feature: np.array, 2d, gallerys_size * feature_dim
        :param gallery_cam: np.array, 1d
        :param gallery_label: np.array, 1d
        :return:
        '''

        # cosine score
        if self.dist == 'cosine':
            # feature has been normalize during intialization
            score = np.matmul(query_feature, gallery_feature.transpose())
            index = np.argsort(score)[::-1]
        elif self.dist == 'euclidean':
            score = self.l2(query_feature.reshape([1, -1]), gallery_feature)
            index = np.argsort(score.reshape([-1]))

        junk_index_1 = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam == gallery_cam))
        junk_index_2 = np.argwhere(gallery_label == -1)
        junk_index = np.append(junk_index_1, junk_index_2)

        good_index = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam != gallery_cam))
        index_wo_junk = self.notin1d(index, junk_index)

        return self.compute_AP(index_wo_junk, good_index)

    def in1d(self, array1, array2, invert=False):
        '''
        :param set1: np.array, 1d
        :param set2: np.array, 1d
        :return:
        '''

        mask = np.in1d(array1, array2, invert=invert)
        return array1[mask]

    def notin1d(self, array1, array2):

        return self.in1d(array1, array2, invert=True)

    def normalize(self, x):
        norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
        return x / norm

    def cosine_dist(self, x, y):
        return sk_metrics.pairwise.cosine_distances(x, y)

    def euclidean_dist(self, x, y):
        return sk_metrics.pairwise.euclidean_distances(x, y)
