#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss

import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)


    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist


def compute_CAJ_jaccard_distance(features=None, cam_labels=None, epoch=None, args=None):
    end = time.time()
    print('Computing CA-jaccard/jaccard distance...')
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(cam_labels, torch.Tensor):
        cam_labels = cam_labels.cpu().numpy()

    # k1: k-reciprocal 方法中的近邻数 k
    # k2: 查询扩展（Query Expansion）中的近邻数
    k1, k2 = args.k1, args.k2

    # ckrnns: 是否开启 Camera-aware KRNNs
    # k1_intra: 同摄像头内的近邻数 k
    # k1_inter: 跨摄像头的近邻数 k
    ckrnns, k1_intra, k1_inter = args.ckrnns, args.k1_intra, args.k1_inter

    # clqe: 是否开启 Camera-aware Local Query Expansion
    # k2_intra/inter: 对应的同/跨摄像头扩展数量
    clqe, k2_intra, k2_inter = args.clqe, args.k2_intra, args.k2_inter


    if ckrnns and clqe:
        mode = f"EPOCH[{epoch}] [CAJaccard (CKRNNS + CLQE)]"
    elif ckrnns and not clqe:
        mode = f"EPOCH[{epoch}] [CAJaccard (CKRNNS + LQE)]"
    elif not ckrnns and clqe:
        mode = f"EPOCH[{epoch}] [CAJaccard (KRNNS + CLQE)]"
    else:
        mode = f"EPOCH[{epoch}] [Jaccard (KRNNS + LQE)]"
    print(mode)

    N = features.shape[0]
    mat_type = np.float32

    # cosine
    original_dist = 2 - 2 * np.matmul(features, features.T)

    # 生成摄像头掩码矩阵 (N x N)
    # cam_mask[i][j] 为 True 表示样本i 和 j 来自同一个摄像头
    cam_mask = (cam_labels.reshape(-1, 1) == cam_labels.reshape(1, -1))
    cam_diff = original_dist[np.triu(~cam_mask, k=1)].mean() - original_dist[np.triu(cam_mask, k=1)].mean()
    print('Camera difference: {:.2f}'.format(cam_diff))

    if ckrnns or clqe:
        # 计算跨摄像头排名 (Inter-camera rank)
        # 技巧：给同摄像头的距离加上一个巨大的数 (999.0)，使其排在最后，
        # 从而 argpartition 选出的前 k 个必然是跨摄像头的邻居。
        inter_rank = np.argpartition(original_dist + 999.0 * cam_mask, range(k1_inter + 2))

        # 计算同摄像头排名 (Intra-camera rank)
        # 技巧：给跨摄像头的距离加上巨大惩罚，迫使选出的前 k 个是同摄像头的邻居。
        intra_rank = np.argpartition(original_dist + 999.0 * (~cam_mask), range(k1_intra + 2))

    # 全局排名：不考虑摄像头标签，直接对原始距离进行部分排序（选出前 k1+2 个）
    global_rank = np.argpartition(original_dist, range(k1 + 2))

    ###################################
    #           KRNNs/CKRNNs          #
    ###################################
    if ckrnns:
        print(f"EPOCH[{epoch}] [CKRNNs] PARAMS: k1_intra: {k1_intra}, k1_inter: {k1_inter}")
    else:
        print(f"EPOCH[{epoch}] [KRNNs] PARAMS: k1: {k1}")

    if ckrnns:
        # CKRNNs 逻辑：
        # 分别计算跨摄像头的互反邻居 和 同摄像头的互反邻居
        # k_reciprocal_neigh 是一个外部函数（未在此代码块定义），用于寻找互为近邻的样本
        nn_inter = [k_reciprocal_neigh(inter_rank, i, k1_inter) for i in range(N)]
        nn_intra = [k_reciprocal_neigh(intra_rank, i, k1_intra) for i in range(N)]

        # 将两类邻居取并集，作为最终的邻居集合
        nn_k1 = [np.union1d(nn_intra[i], nn_inter[i]) for i in range(N)]
    else:
        nn_k1 = [k_reciprocal_neigh(global_rank, i, k1) for i in range(N)]
        nn_k1_half = [k_reciprocal_neigh(global_rank, i, int(np.around(k1 / 2))) for i in range(N)]

    #  初始化权重矩阵 V (N x N)，用于存储样本间的相似度权重
    V = np.zeros((N, N), dtype=mat_type)

    for i in range(N):
        # 获取第 i 个样本的 k-互反邻居索引
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index

        # Jaccard recall (仅在非 CKRNNs 模式下使用)
        # 这是一个启发式策略：如果邻居的邻居与当前集合重叠度很高，则将其扩充进来
        if not ckrnns:
            for candidate in k_reciprocal_index:
                candidate_k_reciprocal_index = nn_k1_half[candidate]

                # 如果交集大于候选者邻居数的 2/3
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index)):
                    # 扩充邻居集合
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        # 确保索引唯一 (去重)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)

        # 取出这些邻居与样本 i 的原始距离
        dist = torch.from_numpy(original_dist[i][k_reciprocal_expansion_index]).unsqueeze(0)

        # 使用 Softmax 将距离转换为权重（距离越小，权重越大，和为1）。这里给 V[i] 的特定列赋值
        V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    ################################
    #            LQE/CLQE          #
    ################################
    # warmup
    if epoch == 0:
        print("Warm-up...")
        k2_intra, k2_inter = 3, 3
    if clqe:
        print(f"EPOCH[{epoch}] [CLQE] PARAMS: k2_intra: {k2_intra}, k2_inter: {k2_inter}")
    else:
        print(f"EPOCH[{epoch}] [LQE] PARAMS: k2: {k2}")

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            if clqe:
                # CLQE: 混合取 同摄像头前 k2_intra 个 和 跨摄像头前 k2_inter 个邻居
                # CLQE 用不同数量的 intra/inter neighbor 来组合邻居，从而兼顾摄像头内相似性与跨摄像头相似性。
                k2nn = np.append(intra_rank[i, :k2_intra],
                                 inter_rank[i, :k2_inter])
            else:
                k2nn = global_rank[i, :k2]  # LQE: 直接取全局前 k2 个邻居

            # 核心操作：当前样本的新向量 V_qe[i] 等于其 k2 个近邻的向量 V 的平均值
            # 这一步利用邻居的信息来平滑特征
            V_qe[i, :] = np.mean(V[k2nn, :], axis=0)
        V = V_qe

    # -----------------------------------------------------------
    # 步骤 2: 计算最终 Jaccard 距离
    # -----------------------------------------------------------
    # 调用辅助函数将权重矩阵 V 转换为 Jaccard 距离矩阵
    jaccard_dist = v2jaccard(V, N, mat_type)

    print("Distance computing time cost: {}".format(time.time() - end))
    return jaccard_dist


def v2jaccard(V, N, mat_type):
    # 建立倒排索引 (Inverted Index)
    # invIndex[i] 存储了所有 "将图像 i 作为邻居" 的图像索引列表
    # 目的是为了加速计算，避免遍历全0元素

    invIndex = []

    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)

    # 遍历每一张图像 i
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)

        # 找到 V[i] 中非零元素的索引（即 i 的邻居）
        indNonZero = np.where(V[i, :] != 0)[0]

        # 利用倒排索引找到需要计算的图像 j
        # 只有当图像 j 和图像 i 有共同邻居时，交集才不为0
        indImages = [invIndex[ind] for ind in indNonZero]

        # 计算交集部分：sum(min(V[i], V[j]))
        for j in range(len(indNonZero)):
            # indNonZero[j] 是特定的特征维度（邻居ID）
            # indImages[j] 是在该维度上有非零值的所有其他图像 j 的列表
            # 累加 min 值到 temp_min
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        # 计算 Jaccard 距离
        # 公式推导：
        # Jaccard Sim = sum(min(a,b)) / sum(max(a,b))
        # 因为 sum(max(a,b)) = sum(a) + sum(b) - sum(min(a,b))
        # 且由于 Softmax，sum(a) ≈ 1, sum(b) ≈ 1
        # 所以分母 ≈ 2 - sum(min(a,b))
        # 最终距离 = 1 - 交集 / (2 - 交集)
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    # 修正数值误差，确保距离非负
    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    return jaccard_dist