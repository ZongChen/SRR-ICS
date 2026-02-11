import numpy as np
import torch
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from scipy.spatial.distance import cdist
from sklearn.cluster._dbscan import dbscan
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import networkx as nx
import os
import wandb
import matplotlib.pyplot as plt
from clustercontrast.evaluators import extract_vit_features

from collections import defaultdict


def img_association(args, model, propagate_loader, id_count_each_cam, rerank=False):

    # ------------------------------------
    # 1. Associate Rate 设定
    # ------------------------------------
    if args.dataset == 'market1501':
        associate_rate = 1.7
    elif args.dataset == 'msmt17':
        associate_rate = 1.3
    else:
        associate_rate = 1.50

    print(f'img_association ->>> dataset:{args.dataset}, associate rate: {associate_rate}')
    id_count_each_cam = np.sum(id_count_each_cam)

    # ------------------------------------
    # 2. 提取特征（变量名统一）
    # ------------------------------------
    all_features, all_features_dup, gt_ids, cam_ids, intra_ids, global_ids = \
        extract_vit_features(model, propagate_loader)

    # all_features 与 all_features_dup 实际一致
    # 之后代码全部使用 all_features

    # ------------------------------------
    # 3. 基于 global_ids 聚合特征
    # ------------------------------------
    cluster_features = []
    cluster_cam_ids = []
    cluster_gt_ids = []
    cluster_intra_ids = []
    cluster_ids = []

    print('feature_aggregation_method -> ', args.feature_aggregation_method)

    for gid in np.unique(global_ids):
        idx = np.where(global_ids == gid)[0]

        # Pooling / 聚合策略
        if args.feature_aggregation_method == 'original':
            pooled_feat = np.mean(all_features_dup[idx], axis=0)

        elif args.feature_aggregation_method == 'barycenter':
            center = np.mean(all_features_dup[idx], axis=0)
            dists = np.linalg.norm(all_features_dup[idx] - center, axis=1)
            pooled_feat = all_features_dup[idx[np.argmin(dists)]]

        elif args.feature_aggregation_method == 'attention_pooling':
            center = np.mean(all_features_dup[idx], axis=0)
            w = np.exp(-np.linalg.norm(all_features_dup[idx] - center, axis=1))
            w = w / (np.sum(w) + 1e-6)
            pooled_feat = np.sum(all_features_dup[idx] * w[:, None], axis=0)

        cluster_features.append(pooled_feat)

        # 这些必须唯一
        assert len(np.unique(cam_ids[idx])) == 1
        assert len(np.unique(gt_ids[idx])) == 1

        cluster_cam_ids.append(cam_ids[idx[0]])
        cluster_gt_ids.append(gt_ids[idx[0]])
        cluster_intra_ids.append(intra_ids[idx[0]])
        cluster_ids.append(global_ids[idx[0]])

    # ------------------------------------
    # 4. 转 numpy
    # ------------------------------------
    cluster_features = np.array(cluster_features)
    cluster_features = cluster_features / np.linalg.norm(cluster_features, axis=1, keepdims=True)

    cluster_cam_ids = np.array(cluster_cam_ids)
    cluster_gt_ids = np.array(cluster_gt_ids)
    cluster_intra_ids = np.array(cluster_intra_ids)
    cluster_ids = np.array(cluster_ids)

    # ------------------------------------
    # 5. 计算距离矩阵 W
    # ------------------------------------
    if rerank:
        print('Using Jaccard 距离')
        W = compute_jaccard_distance(torch.from_numpy(cluster_features), k1=20, k2=6)
    else:
        print('Using cdist 距离')
        W = cdist(cluster_features, cluster_features, 'sqeuclidean')

    print(f'img_association： distance matrix: shape= {W.shape}')

    # ------------------------------------
    # 6. 标签传播 (根据用户选择的方法)
    # ------------------------------------
    if args.label_propagete_method == 'connect':
        updated_labels = propagate_label(W, cluster_gt_ids, cluster_cam_ids,
                                         associate_rate * id_count_each_cam)

    elif args.label_propagete_method == 'mmn':
        updated_labels = propagate_label_mnn(W, cluster_gt_ids, cluster_cam_ids, top_k=5)

    elif args.label_propagete_method == 'absorb_repel':
        updated_labels = absorb_and_repel(
            W, cluster_gt_ids, cluster_cam_ids, cluster_intra_ids,
            theta_absorb=0.8, theta_repel=0.3
        )

    elif args.label_propagete_method == 'spectral':
        updated_labels = propagate_label_with_spectral(W, cluster_gt_ids, cluster_cam_ids)

    elif args.label_propagete_method == 'propagate_GT':
        updated_labels = propagate_GT(W, cluster_gt_ids, cluster_cam_ids,
                                      associate_rate * id_count_each_cam)

    elif args.label_propagete_method == 'multisteps':
        updated_labels = propagate_label_multisteps(W, cluster_cam_ids)

    elif args.label_propagete_method == 'gnn':
        updated_labels = propagate_label_gnn(W, cluster_gt_ids, cluster_cam_ids,
                                             associate_rate * id_count_each_cam)

    elif args.label_propagete_method == 'v3':
        updated_labels = propagate_label_with_printing(
            W, cluster_gt_ids, cluster_cam_ids,
            associate_rate * id_count_each_cam
        )

    else:
        raise Exception

    print(f'img_association：length of updated_label= {len(updated_labels)}, '
          f'min= {np.min(updated_labels)}, '
          f'max= {np.max(updated_labels)}')

    # ------------------------------------
    # 7. 伪标签质量评估
    # ------------------------------------
    pred_label = updated_labels[global_ids]
    valid_idx = np.where(pred_label >= 0)[0]

    print(f'valid indexs: {len(valid_idx)}')

    ari = metrics.adjusted_rand_score(gt_ids[valid_idx], pred_label[valid_idx])
    nmi = metrics.normalized_mutual_info_score(gt_ids[valid_idx], pred_label[valid_idx])

    print(f'评估伪标签质量 -->> ARI: {ari}, NMI: {nmi}')

    if args.wandb_enabled:
        wandb.log({'ARI': ari, 'NMI': nmi})

    del all_features_dup

    # ------------------------------------
    # 8. 返回（名称统一）
    # ------------------------------------
    return (
        updated_labels,      # np.ndarray, shape = [num_clusters]
        all_features,    # np.ndarray, 原始所有特征
        global_ids,          # np.ndarray, 原始全局标签（与 all_features 对应）
        cam_ids,             # np.ndarray, 原始摄像头标签
        intra_ids,           # np.ndarray, 原始 ICS intra ID
        cluster_ids,         # np.ndarray, 聚合后的 cluster ID
        cluster_gt_ids       # np.ndarray, 聚合后的 GT ID
    )



def propagate_GT(W, gt_IDs, cam_IDs, associate_class_pair=True, topk=5, sim_thresh=0.5):
    """
    基于相似度矩阵W进行标签传播，支持互为最近邻验证和多阶段筛选。
    """
    N = len(gt_IDs)
    updated_label = gt_IDs.copy()
    next_label = max(gt_IDs) + 1 if gt_IDs.max() >= 0 else 0

    # 遍历每个样本
    for i in range(N):
        if updated_label[i] != -1:
            continue
        idx_sorted = np.argsort(W[i])[:topk + 1]  # 包括自己在内前 topk
        found = False
        for j in idx_sorted[1:]:  # 排除自己
            if cam_IDs[i] == cam_IDs[j]:
                continue
            if W[i][j] > sim_thresh:
                continue
            if updated_label[j] != -1:
                # 互为top-k + 标签已知 + 相似度小于阈值
                updated_label[i] = updated_label[j]
                found = True
                break
        if not found:
            updated_label[i] = next_label
            next_label += 1

    return updated_label


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity


def propagate_label_gnn(new_features, new_gt_IDs, new_cams, associate_class_pair):
    """
    GNN-based label propagation for cross-camera association.

    Args:
        new_features (np.ndarray): Feature vectors, shape (n, d).
        new_gt_IDs (np.ndarray): Ground truth IDs, shape (n,).
        new_cams (np.ndarray): Camera IDs, shape (n,).
        associate_class_pair (float): Threshold for number of associations.

    Returns:
        np.ndarray: Updated labels, shape (n,), with cluster IDs (-1 for noise).
    """
    # Convert inputs to PyTorch tensors
    features = torch.from_numpy(new_features).float()
    cams = torch.from_numpy(new_cams).long()
    n = features.shape[0]

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(new_features)

    # Enforce cross-camera constraint: set same-camera similarities to 0
    for i in range(n):
        same_cam_mask = cams == cams[i]
        sim_matrix[i, same_cam_mask] = 0

    # Create edge index for GNN (sparse graph based on top similarities)
    threshold = np.percentile(sim_matrix.flatten(), 100 * (1 - associate_class_pair / (n * n)))
    edge_index = []
    edge_weight = []
    for i in range(n):
        for j in range(i + 1, n):  # Avoid self-loops and symmetric edges
            if sim_matrix[i, j] > threshold and cams[i] != cams[j]:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_weight.append(sim_matrix[i, j])
                edge_weight.append(sim_matrix[i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Build PyTorch Geometric data object
    data = Data(x=features, edge_index=edge_index, edge_attr=edge_weight)

    # Define simple GCN model
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, data):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight)
            return x

    # Initialize and run GCN
    model = GCN(in_channels=features.shape[1], hidden_channels=128, out_channels=64)
    model.eval()
    with torch.no_grad():
        refined_features = model(data)

    # Convert refined features to numpy for clustering
    refined_features = refined_features.numpy()

    # Apply HDBSCAN for clustering
    clusterer = HDBSCAN(min_cluster_size=2, metric='euclidean')
    updated_label = clusterer.fit_predict(refined_features)

    # Log association statistics using ground truth
    associ_num = np.sum(updated_label != -1)
    associ_pos_num = 0
    for cluster_id in np.unique(updated_label):
        if cluster_id == -1:
            continue
        indices = np.where(updated_label == cluster_id)[0]
        if len(np.unique(new_gt_IDs[indices])) == 1:
            associ_pos_num += len(indices)

    print(f'GNN Label Propagation ---> Associated pairs: {associ_pos_num}/{associ_num} correct')

    return updated_label


def propagate_label(W, cam_IDs, associate_threshold=5, topk=5, second_stage=True):
        """
        基于相似性矩阵W，进行跨摄像头的标签传播，并输出新的伪标签。

        参数：
            W: numpy array, [N, N] 相似度（小距离）矩阵（如Jaccard距离）
            cam_IDs: list or array of length N，表示每个样本来自哪个摄像头
            associate_threshold: 控制要连接的最近邻数量（跨摄像头）
            topk: 初步关联时考虑的最近邻个数
            second_stage: 是否启用第二阶段传播

        返回：
            updated_label: numpy array, 每个样本的新标签
        """

        N = W.shape[0]
        cam_IDs = np.array(cam_IDs)

        # 初始化邻接矩阵
        adjacency = np.zeros((N, N), dtype=bool)

        # 初步传播阶段：跨摄像头最近邻
        for i in range(N):
            dists = W[i]
            cam_mask = (cam_IDs != cam_IDs[i])  # 只考虑不同摄像头
            candidates = np.where(cam_mask)[0]

            if len(candidates) == 0:
                continue

            nearest = candidates[np.argsort(dists[candidates])[:topk]]
            for j in nearest:
                # 双向验证（互为topk邻居）
                j_candidates = np.where(cam_IDs != cam_IDs[j])[0]
                if i in j_candidates[np.argsort(W[j][j_candidates])[:topk]]:
                    adjacency[i, j] = True
                    adjacency[j, i] = True

        # 第二阶段传播：依据联通图聚类
        # 构建图的邻接表
        graph = defaultdict(list)
        for i in range(N):
            for j in range(N):
                if adjacency[i, j]:
                    graph[i].append(j)

        # 联通组件划分
        visited = np.zeros(N, dtype=bool)
        labels = -np.ones(N, dtype=int)
        current_label = 0

        def dfs(node, label):
            stack = [node]
            while stack:
                n = stack.pop()
                if not visited[n]:
                    visited[n] = True
                    labels[n] = label
                    stack.extend(graph[n])

        for i in range(N):
            if not visited[i]:
                dfs(i, current_label)
                current_label += 1

        # 可选：使用DBSCAN进一步细化大联通组件
        if second_stage:
            unique_labels = np.unique(labels)
            refined_labels = -np.ones(N, dtype=int)
            current_id = 0
            for lbl in unique_labels:
                if lbl == -1:
                    continue
                idx = np.where(labels == lbl)[0]
                if len(idx) < 2:
                    refined_labels[idx] = current_id
                    current_id += 1
                    continue

                sub_dist = W[idx][:, idx]
                db = DBSCAN(eps=0.6, min_samples=2, metric='precomputed')
                cluster_ids = db.fit_predict(sub_dist)
                for c in np.unique(cluster_ids):
                    refined_labels[idx[cluster_ids == c]] = current_id
                    current_id += 1
            labels = refined_labels

        return labels


def absorb_and_repel(W, IDs, all_cams, labeled_indices, theta_absorb=0.8, theta_repel=0.3):
    """
    加强调试输出的 Absorb & Repel 标签传播方法
    """

    print('>>> Start Absorb & Repel with camera-aware constraints...')
    N = len(W)
    refined_labels = -1 * np.ones(N, dtype=int)

    label_cam_to_indices = defaultdict(list)
    for idx in labeled_indices:
        key = (IDs[idx], all_cams[idx])
        label_cam_to_indices[key].append(idx)

    print(f'  Total labeled (ID, cam) groups: {len(label_cam_to_indices)}')
    cluster_id = 0
    used_indices = set()

    absorb_total = 0
    absorb_success = 0

    # --- Absorb Phase ---
    for (pid, cam), indices in label_cam_to_indices.items():
        center_vector = np.mean(W[indices], axis=0)
        similarity = 1 - center_vector  # 距离变为相似度
        candidate_indices = np.where(similarity >= theta_absorb)[0]

        absorbed = [i for i in candidate_indices if all_cams[i] != cam and refined_labels[i] == -1]

        absorb_total += len(absorbed)
        absorb_success += sum(IDs[i] == pid for i in absorbed)

        if len(absorbed) > 0:
            for i in absorbed + indices:
                refined_labels[i] = cluster_id
                used_indices.add(i)
            print(f'  [+] Cluster {cluster_id} ← ID={pid}@Cam{cam}: absorbed={len(absorbed)} (correct={sum(IDs[i] == pid for i in absorbed)})')
            cluster_id += 1

    print(f'\n>>> Absorb phase done: total absorbed {absorb_total}, correct {absorb_success} ({absorb_success / (absorb_total + 1e-6):.2%})')

    # --- Repel Phase ---
    repel_success, repel_total = 0, 0
    for i in range(N):
        if refined_labels[i] != -1:
            continue
        max_sim = 0
        for (pid, cam), indices in label_cam_to_indices.items():
            if all_cams[i] == cam:
                continue
            center_vector = np.mean(W[indices], axis=0)
            sim = 1 - center_vector[i]
            if sim > max_sim:
                max_sim = sim
        if max_sim < theta_repel:
            continue  # 被排斥
        else:
            refined_labels[i] = cluster_id
            if IDs[i] in [pid for (pid, _) in label_cam_to_indices.keys()]:
                repel_success += 1
            repel_total += 1
            cluster_id += 1

    print(f'\n>>> Repel phase done: assigned {repel_total}, correct {repel_success} ({repel_success / (repel_total + 1e-6):.2%})')

    # --- Fallback DBSCAN ---
    unassigned = np.where(refined_labels == -1)[0]
    print(f'\n>>> Unassigned samples: {len(unassigned)}')
    if len(unassigned) > 0:
        subW = W[np.ix_(unassigned, unassigned)]
        np.fill_diagonal(subW, 0)
        new_labels = DBSCAN(eps=0.4, min_samples=2, metric='precomputed').fit_predict(subW)

        label_map = {}
        for i, idx in enumerate(unassigned):
            if new_labels[i] != -1:
                label = new_labels[i]
                if label not in label_map:
                    label_map[label] = cluster_id
                    cluster_id += 1
                refined_labels[idx] = label_map[label]

        unique_final_labels = set(refined_labels) - {-1}
        print(f'>>> Final fallback clusters: {len(label_map)} newly formed, total clusters: {len(unique_final_labels)}')

    print('>>> Absorb & Repel completed.\n')

    return refined_labels


def propagate_label_multisteps(W, cam_IDs, topk=5, second_stage=True):
    """
    基于相似性矩阵W，进行跨摄像头的标签传播，并输出新的伪标签。

    参数：
        W: numpy array, [N, N] 相似度（小距离）矩阵（如Jaccard距离）
        cam_IDs: list or array of length N，表示每个样本来自哪个摄像头
        associate_threshold: 控制要连接的最近邻数量（跨摄像头）
        topk: 初步关联时考虑的最近邻个数
        second_stage: 是否启用第二阶段传播

    返回：
        updated_label: numpy array, 每个样本的新标签
    """

    N = W.shape[0]
    cam_IDs = np.array(cam_IDs)

    # 初始化邻接矩阵
    adjacency = np.zeros((N, N), dtype=bool)

    # 初步传播阶段：跨摄像头最近邻
    for i in range(N):
        dists = W[i]
        cam_mask = (cam_IDs != cam_IDs[i])  # 只考虑不同摄像头
        candidates = np.where(cam_mask)[0]

        if len(candidates) == 0:
            continue

        nearest = candidates[np.argsort(dists[candidates])[:topk]]
        for j in nearest:
            # 双向验证（互为top-k邻居）
            j_candidates = np.where(cam_IDs != cam_IDs[j])[0]
            if i in j_candidates[np.argsort(W[j][j_candidates])[:topk]]:
                adjacency[i, j] = True
                adjacency[j, i] = True

    # 第二阶段传播：依据联通图聚类
    # 构建图的邻接表
    graph = defaultdict(list)

    for i in range(N):
        for j in range(N):
            if adjacency[i, j]:
                graph[i].append(j)  #

    # 联通组件划分
    visited = np.zeros(N, dtype=bool)
    labels = -np.ones(N, dtype=int)
    current_label = 0

    def dfs(node, label):
        stack = [node]
        while stack:
            n = stack.pop()
            if not visited[n]:
                visited[n] = True
                labels[n] = label
                stack.extend(graph[n])  # Update Stack

    # DFS
    for i in range(N):  # 对每个AccuID进行DFS
        if not visited[i]:
            dfs(i, current_label)
            current_label += 1

    # 可选：使用DBSCAN进一步细化大联通组件
    if second_stage:
        unique_labels = np.unique(labels)
        refined_labels = -np.ones(N, dtype=int)
        current_id = 0

        for lbl in unique_labels:
            if lbl == -1:
                continue
            idx = np.where(labels == lbl)[0]
            if len(idx) < 2:
                refined_labels[idx] = current_id
                current_id += 1
                continue

            sub_dist = W[idx][:, idx]

            db = DBSCAN(eps=0.6, min_samples=2, metric='precomputed')
            cluster_ids = db.fit_predict(sub_dist)

            for c in np.unique(cluster_ids):
                refined_labels[idx[cluster_ids == c]] = current_id
                current_id += 1

        labels = refined_labels

    return labels



def propagate_label_mnn(W, global_IDs, cam_IDs, top_k=5, eps=3, min_samples=2):
    """
    Propagate cross-camera labels using Mutual Nearest Neighbor (MNN) filtering.

    Args:
        W (ndarray): Pairwise distance/similarity matrix between prototypes.
        global_IDs (ndarray): Ground truth identity labels (for evaluation).
        cam_IDs (ndarray): Camera ID per prototype.
        top_k (int): Number of nearest neighbors to consider for MNN.
        eps (float): DBSCAN epsilon threshold.
        min_samples (int): DBSCAN min_samples parameter.

    Returns:
        new_merged_label (ndarray): Pseudo-labels after MNN-based association and clustering.
    """
    print("Start cross-camera association using Mutual Nearest Neighbors (MNN)...")

    num_instances = W.shape[0]
    num_cams = len(np.unique(cam_IDs))

    # Build index lists of mutual nearest neighbors (MNN) across cameras
    mnn_pairs = set()
    for i in range(num_instances):
        cam_i = cam_IDs[i]

        # Candidate indices in different cameras
        cross_indices = np.where(cam_IDs != cam_i)[0]
        if len(cross_indices) == 0:
            continue

        # Distances to all other camera classes
        dists = W[i, cross_indices]
        top_k_idx = np.argsort(dists)[:top_k]
        nn_indices = cross_indices[top_k_idx]

        for j in nn_indices:
            cam_j = cam_IDs[j]
            if cam_i == cam_j:
                continue

            # Now check if i is also in top-k of j
            cross_j_indices = np.where(cam_IDs != cam_j)[0]
            dists_j = W[j, cross_j_indices]
            top_k_j_idx = np.argsort(dists_j)[:top_k]
            nn_j_indices = cross_j_indices[top_k_j_idx]

            if i in nn_j_indices:
                # Mutual nearest neighbor: add to set (ordered tuple to avoid duplicate)
                mnn_pairs.add(tuple(sorted((i, j))))

    print(f"  MNN pairs found: {len(mnn_pairs)}")

    # Build associate matrix based on MNN pairs
    associate_mat = np.full_like(W, fill_value=1000.0)
    for i, j in mnn_pairs:
        associate_mat[i, j] = 0
        associate_mat[j, i] = 0

    np.fill_diagonal(associate_mat, 0)

    # Apply DBSCAN
    new_merged_label = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(associate_mat)

    # (Optional) Evaluation stats if using ground truth
    hit = sum(1 for (i, j) in mnn_pairs if global_IDs[i] == global_IDs[j])
    print(f"  Ground truth match in MNN pairs: {hit}/{len(mnn_pairs)}")

    return new_merged_label


def propagate_label(W,
                    gt_IDs,
                    cam_IDs,
                    associate_class_pair):
    """
    通过标签传播（label propagation）的方法，将不同摄像头（camera）下的目标（如行人、车辆等）进行跨摄像头的关联（cross-camera association）
    """

    # start label propagation for
    print(f'propagate_label...使用传统连通分量  associate_class_pair={associate_class_pair}')

    '''
    np.savez('analysis_data/propagate_label_input_params.npz',
             W=W,
             gt_IDs=gt_IDs,
             cam_IDs=cam_IDs,
             associate_class_pair=associate_class_pair)
    '''

    associateMat = 1000 * np.ones(W.shape, W.dtype)  # 最终的关联矩阵，值为1000表示无连接

    # 屏蔽同摄像头内的目标 对角线和下三角边
    for i in range(len(W)):
        W[i, np.where(cam_IDs == cam_IDs[i])[0]] = 1000  # 同摄像头赋极大值

        # 下三角赋极大值
        lower_ind = np.arange(0, i)
        W[i, lower_ind] = 1000

    # 排序并选择最相似的目标对
    sorted_ind = np.argsort(W.flatten())[0:int(associate_class_pair)]  #
    row_ind = sorted_ind // W.shape[1]
    col_ind = sorted_ind % W.shape[1]

    # 跨摄像头关联
    C = len(np.unique(cam_IDs))
    cam_cover_info = np.zeros((len(W), C))  # 防止一个节点和同一摄像头中的多个目标连接，避免冗余传播。
    associ_num, ignored_num = 0, 0
    associ_pos_num, ignored_pos_num = 0, 0
    # print('  associate_class_pair: {}'.format(associate_class_pair))

    thresh = associate_class_pair  # 5545

    # 遍历目标对并进行关联
    for m in range(len(row_ind)):
        cls1 = row_ind[m]  # 行index
        cls2 = col_ind[m]  # 列index
        assert (cam_IDs[cls1] != cam_IDs[cls2])

        # 确保两个 global_ID 代表（cls1 和 cls2）尚未与对方对应的摄像头发生过任何关联，避免重复配对
        check = (cam_cover_info[cls1, cam_IDs[cls2]] == 0 and cam_cover_info[cls2, cam_IDs[cls1]] == 0)

        # 更新 associateMat[cls1, cls2] = 0 和 associateMat[cls2, cls1] = 0，并在 cam_cover_info 中标记
        if check:
            cam_cover_info[cls1, cam_IDs[cls2]] = 1
            cam_cover_info[cls2, cam_IDs[cls1]] = 1

            associateMat[cls1, cls2] = 0  # 关键的部分， 表示目标之间的有效关联
            associateMat[cls2, cls1] = 0

            associ_num += 1
            if gt_IDs[cls1] == gt_IDs[cls2]:
                associ_pos_num += 1  # 表示确实属于同一类别，命中
        else:
            ignored_num += 1
            if gt_IDs[cls1] == gt_IDs[cls2]:
                ignored_pos_num += 1  # 未命中

        if associ_num >= thresh:  # 只对其中5545进行判别
            break

    # 输出成功关联的目标对数、被忽略的目标对数，以及其中ID匹配的比例。
    print(f'\n标签传播 --->  associated class pairs: {associ_pos_num}/{associ_num} correct, ignored class pairs: {ignored_pos_num}/{ignored_num} correct')

    # 对角线处理
    for m in range(len(associateMat)):
        associateMat[m, m] = 0

    # Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1. oral = 3
    _, new_merged_label = dbscan(associateMat, eps=3, min_samples=2, metric='precomputed')
    # _, new_merged_label = dbscan(associateMat, eps=0.6, min_samples=4, metric='precomputed')
    # cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

    del associateMat

    return new_merged_label  # 是聚类后的标签，表示不同摄像头下的目标是否属于同一个簇（即是否关联成功）



def propagate_label_with_spectral(W,
                    global_IDs,
                    cam_IDs,
                    pseudo_labels_intra=None,
                    use_logits=False,
                    logits=None,
                    fusion_weight=0.7,
                    edge_percentile=5,
                    expected_cluster_num=500):
    """
    改进版标签传播：
    - 支持融合多个相似度源
    - 动态选边
    - 构建跨摄像头传播图
    - 使用谱聚类替代 DBSCAN
    """

    print('Start improved associating ID...')

    # Step 1: optional 多源融合
    if use_logits and logits is not None:
        def compute_kl_distance(logits):
            logits = np.clip(logits, 1e-6, 1)
            log_logits = np.log(logits)
            kl_matrix = np.zeros((logits.shape[0], logits.shape[0]))
            for i in range(len(logits)):
                for j in range(len(logits)):
                    kl_matrix[i, j] = np.sum(logits[i] * (log_logits[i] - np.log(logits[j])))
            return (kl_matrix + kl_matrix.T) / 2  # symmetrize

        W_logits = compute_kl_distance(logits)
        W = fusion_weight * W + (1 - fusion_weight) * W_logits

    # Step 2: 利用 Intra-camera pseudo labels 进行强连接
    if pseudo_labels_intra is not None:
        print("  [Info] Injecting intra-camera pseudo labels...")
        for label in np.unique(pseudo_labels_intra):
            inds = np.where(pseudo_labels_intra == label)[0]
            for i in inds:
                for j in inds:
                    if i != j:
                        W[i, j] = 0  # 强制连接

    # Step 3: mask 同摄像头内的边 & 下三角
    for i in range(len(W)):
        W[i, np.where(cam_IDs == cam_IDs[i])[0]] = 1000
        W[i, np.arange(0, i)] = 1000

    # Step 4: 构建 soft affinity matrix
    sigma = np.std(W[W < 1000])  # 排除屏蔽的大值
    S = np.exp(-W / (sigma + 1e-6))  # 高斯核转为相似度

    # Step 5: 动态选边（根据边百分位数）
    threshold = np.percentile(W[W < 1000], edge_percentile)
    keep_mask = W < threshold
    print(f"  [Info] edge threshold={threshold:.4f}, kept edge ratio={np.mean(keep_mask):.4f}")

    S[~keep_mask] = 0  # 删除高距离边

    # Step 6: 光谱聚类 代替DBSCAN
    clustering = SpectralClustering(n_clusters=expected_cluster_num, affinity='precomputed', assign_labels='kmeans')
    new_merged_label = clustering.fit_predict(S)

    # Step 7: 打印统计信息（命中情况）
    associ_num, associ_pos_num = 0, 0
    for i in range(len(W)):
        for j in range(i + 1, len(W)):
            if new_merged_label[i] == new_merged_label[j] and cam_IDs[i] != cam_IDs[j]:
                associ_num += 1
                if global_IDs[i] == global_IDs[j]:
                    associ_pos_num += 1
    print(f"\n改进标签传播 ---> associated cross-camera class pairs: {associ_pos_num}/{associ_num} correct")

    return new_merged_label


def propagate_label_with_printing(W, gt_IDs, cam_IDs, associate_class_pair, debug=False, log_dir="./logs", draw_plot=True):
    """
    改进版标签传播算法：
    1. 输出详细分析信息（距离、覆盖率、摄像头对分布）
    2. 记录日志文件
    3. 可选绘制正负样本距离分布直方图
    """

    associate_class_pair = int(associate_class_pair)
    print(f"[INFO] Start Label Propagation... associate_class_pair={associate_class_pair}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "label_propagation_log.txt")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    associateMat = 1000 * np.ones(W.shape, W.dtype)

    # 屏蔽同摄像头目标 & 下三角
    for i in range(len(W)):
        W[i, np.where(cam_IDs == cam_IDs[i])[0]] = 1000
        W[i, :i] = 1000

    sorted_ind = np.argsort(W.flatten())[:associate_class_pair]
    row_ind = sorted_ind // W.shape[1]
    col_ind = sorted_ind % W.shape[1]

    pos_dists, neg_dists = [], []
    cam_pair_count = {}
    C = len(np.unique(cam_IDs))
    cam_cover_info = np.zeros((len(W), C))
    associ_num, ignored_num = 0, 0
    associ_pos_num, ignored_pos_num = 0, 0

    for m in range(len(row_ind)):
        cls1, cls2 = row_ind[m], col_ind[m]
        assert cam_IDs[cls1] != cam_IDs[cls2]
        dist = W[cls1, cls2]

        if gt_IDs[cls1] == gt_IDs[cls2]:
            pos_dists.append(dist)
        else:
            neg_dists.append(dist)

        check = (cam_cover_info[cls1, cam_IDs[cls2]] == 0 and cam_cover_info[cls2, cam_IDs[cls1]] == 0)
        if check:
            cam_cover_info[cls1, cam_IDs[cls2]] = 1
            cam_cover_info[cls2, cam_IDs[cls1]] = 1
            associateMat[cls1, cls2] = 0
            associateMat[cls2, cls1] = 0
            associ_num += 1
            pair = tuple(sorted((cam_IDs[cls1], cam_IDs[cls2])))
            cam_pair_count[pair] = cam_pair_count.get(pair, 0) + 1

            if gt_IDs[cls1] == gt_IDs[cls2]:
                associ_pos_num += 1
        else:
            ignored_num += 1
            if gt_IDs[cls1] == gt_IDs[cls2]:
                ignored_pos_num += 1

        if associ_num >= associate_class_pair:
            break

        if debug:
            log(f"[DEBUG] Pair {m}: cls1={cls1}, cls2={cls2}, dist={dist:.4f}, same_ID={gt_IDs[cls1]==gt_IDs[cls2]}")

    connected_nodes = np.sum(np.min(associateMat, axis=1) == 0)
    coverage_ratio = connected_nodes / len(W)

    log(f"\n[RESULT] 标签传播:")
    log(f"  关联对数: {associ_num}, 正确关联: {associ_pos_num}")
    log(f"  被忽略对数: {ignored_num}, 其中正样本: {ignored_pos_num}")
    log(f"  覆盖率: {coverage_ratio:.2%} ({connected_nodes}/{len(W)})")

    if pos_dists and neg_dists:
        log(f"[DIST] 正样本均值={np.mean(pos_dists):.4f}, 负样本均值={np.mean(neg_dists):.4f}")
        log(f"[DIST] 正样本最小={np.min(pos_dists):.4f}, 负样本最小={np.min(neg_dists):.4f}")
    else:
        log("[DIST] 无法统计正负样本距离")

    '''
    log("[INFO] 每个摄像头对关联数量:")
    for k, v in sorted(cam_pair_count.items(), key=lambda x: x[1], reverse=True):
        log(f"  CamPair {k}: {v}")
    '''

    np.fill_diagonal(associateMat, 0)

    new_merged_label = DBSCAN(eps=3, min_samples=2, metric='precomputed').fit_predict(associateMat)

    if len(np.unique(gt_IDs)) > 1:
        ari = metrics.adjusted_rand_score(gt_IDs, new_merged_label)
        nmi = metrics.normalized_mutual_info_score(gt_IDs, new_merged_label)
        log(f"[QUALITY] 评估所有伪标签质量: ARI={ari:.4f}, NMI={nmi:.4f}")
    else:
        log("[QUALITY] 无法评估伪标签质量")

    log_file.close()

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 可选绘制距离直方图
    if draw_plot and pos_dists and neg_dists:
        plt.figure(figsize=(8, 6))

        plt.hist(pos_dists, bins=50, alpha=0.7, label="Positive Sample", color='g')
        plt.hist(neg_dists, bins=50, alpha=0.7, label="Negative Sample", color='r')

        plt.title("Positive and negative sample distance")
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.savefig(os.path.join(log_dir, "distance_distribution.png"))
        print("distance_distribution.png saved")
        if debug:
            plt.show()

    return new_merged_label
