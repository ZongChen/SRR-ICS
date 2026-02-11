import torch
import numpy as np
import time
import math
from typing import Optional, Union, Any
import torch.nn.functional as F
import wandb


def v2jaccard_numpy_chunked(V, chunk_size=512):
    """
    V: numpy array (N, N) float32 (sparse-like: most rows have many zeros)
    Returns jaccard_dist (N,N) numpy float32
    Performs processing in chunks to limit peak memory.
    """
    N = V.shape[0]
    # build invIndex: list of indices per column where V[:,col] != 0
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(N)]

    jaccard_dist = np.zeros((N, N), dtype=np.float32)
    # process rows in chunks
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        # prepare a block
        block = np.zeros((end - start, N), dtype=np.float32)
        for ii, i in enumerate(range(start, end)):
            indNonZero = np.where(V[i, :] != 0)[0]
            if indNonZero.size == 0:
                block[ii, :] = 1.0  # max distance if empty
                continue
            temp_min = np.zeros((N,), dtype=np.float32)
            for j in indNonZero:
                idxs = invIndex[j]
                # add min(V[i,j], V[idxs, j]) to the positions idxs
                # vectorized:
                vals = np.minimum(V[i, j], V[idxs, j])
                temp_min[idxs] += vals
            block[ii, :] = 1 - temp_min / (2 - temp_min + 1e-12)
        jaccard_dist[start:end, :] = block
    # clip numeric issues
    jaccard_dist[jaccard_dist < 0] = 0.0
    return jaccard_dist

def vectorization_ics_groupwise_fast(original_dist, nn_neighbors, same_cam_and_id, cross_cam, alpha=0.3, tau_intra=2.0):
    print("vectorization_ics_groupwise_fast")

    N = original_dist.shape[0]
    dev = original_dist.device

    lengths = [len(n) if n is not None else 0 for n in nn_neighbors]
    max_k = max(lengths)

    neighbor_indices = torch.full((N, max_k), -1, dtype=torch.long, device=dev)
    for i, n in enumerate(nn_neighbors):
        if n is not None and len(n) > 0:
            neighbor_indices[i, :len(n)] = torch.tensor(n, device=dev)

    valid_mask = (neighbor_indices != -1)

    safe_indices = neighbor_indices.clone()
    safe_indices[~valid_mask] = 0

    row_ids = torch.arange(N, device=dev).unsqueeze(1).expand(-1, max_k)
    d_neighbors = original_dist[row_ids, safe_indices]  # (N, max_k)

    is_intra = same_cam_and_id[row_ids, safe_indices] & valid_mask
    is_cross = cross_cam[row_ids, safe_indices] & valid_mask

    V_values = torch.zeros((N, max_k), dtype=torch.float32, device=dev)

    d_intra_temp = d_neighbors.clone()
    d_intra_temp[~is_intra] = float('inf')
    # Softmax(-d / tau)
    w_intra = F.softmax(-d_intra_temp / tau_intra, dim=1)
    # 填入 V_values (注意：F.softmax 对全 -inf 的行会输出 nan，需要处理)
    # 这里只把计算出的 intra 权重赋给 is_intra 的位置
    V_values = torch.where(is_intra, w_intra * alpha, V_values)

    # --- Cross 处理 (Masked Softmax) ---
    d_cross_temp = d_neighbors.clone()
    d_cross_temp[~is_cross] = float('inf')
    w_cross = F.softmax(-d_cross_temp, dim=1)  # dim=1 是对邻居维度
    V_values = torch.where(is_cross, w_cross * (1.0 - alpha), V_values)

    V = torch.zeros((N, N), dtype=torch.float32, device=dev)
    V.scatter_(1, safe_indices, V_values)  # 将计算好的 values 散射回 V 矩阵

    # 归一化
    row_sums = V.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    V = V / row_sums

    return V


def compute_ics_jaccard_distance(features: torch.Tensor,
                                   cam_labels: torch.Tensor,
                                   intra_camera_labels: torch.Tensor,
                                   epoch: int,
                                   args: Any,
                                   max_memory_mode: bool = False,
                                   ICS_strict_reciprocal: bool = True,
                                   gt_labels: torch.Tensor = None
                                   ) -> np.ndarray:
    t0 = time.time()

    K_search = args.K_search
    K_intra = args.K_intra
    K_cross = args.K_cross

    device = args.device if hasattr(args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(device)

    print(f'\n获取超参数为 K_search: {K_search},  K_intra: {K_intra}, K_cross: {K_cross} \n')

    feat = features.to(dev)
    cam = cam_labels.to(dev)
    intra_id = intra_camera_labels.to(dev)

    if feat.dtype == torch.float16:
        feat = feat.float()

    N = feat.shape[0]
    assert N == cam.shape[0] == intra_id.shape[0]

    feat = torch.nn.functional.normalize(feat, p=2, dim=1)

    # ---------- original distance ----------
    with torch.no_grad():
        sim = torch.matmul(feat, feat.t())  # (N,N)
        original_dist = 2.0 - 2.0 * sim
        original_dist = torch.clamp(original_dist, min=0.0)

    # ---------- 构建掩码 (Masks) ----------
    # cam_mask: 同摄像头为 True
    cam_mask = cam.view(-1, 1) == cam.view(1, -1)
    # id_mask: 同 ID (根据 intra_id) 为 True。
    id_mask = intra_id.view(-1, 1) == intra_id.view(1, -1)

    P2 = 2000.0
    same_cam_and_id = cam_mask & id_mask
    same_cam_diff_id = cam_mask & (~id_mask)
    cross_cam = ~cam_mask

    INF = 1e6
    penalty = torch.zeros_like(original_dist, dtype=torch.float32, device=dev)  # 初始化惩罚矩阵
    penalty = penalty + (cross_cam.float() * P2)  # 对跨摄像头样本加 P2 惩罚，使其排在"同摄像头同ID"之后
    penalty = penalty + (same_cam_diff_id.float() * INF)

    dist_priority = original_dist + penalty

    _, topk_idx = torch.topk(dist_priority, k=K_search, largest=False, sorted=True)  # (N, k_search)

    topk_mask = torch.zeros((N, N), dtype=torch.bool, device=dev)
    rows = torch.arange(N, device=dev).unsqueeze(1).repeat(1, K_search).reshape(-1)
    cols = topk_idx.reshape(-1)  # 选定的k_search个
    topk_mask[rows, cols] = True

    reciprocal_mask = topk_mask & topk_mask.t()  # 互反的掩码矩阵

    nn_neighbors = [None] * N

    same_cam_and_id_cpu = same_cam_and_id.cpu().numpy()
    cross_cam_cpu = cross_cam.cpu().numpy()

    for i in range(N):
        pr_cpu = topk_idx[i].cpu().numpy().astype(int)

        recips = pr_cpu[
            reciprocal_mask[i, topk_idx[i]].cpu().numpy()
        ]

        trusted_intra = [
            int(j) for j in pr_cpu
            if same_cam_and_id_cpu[i, j]
        ]

        if K_intra is not None:
            trusted_intra = trusted_intra[:K_intra]

        trusted_cross = [
            int(j) for j in pr_cpu
            if cross_cam_cpu[i, j] and j in recips  # 要求互反
        ][:K_cross]

        nn_neighbors[i] = np.array(
            trusted_intra + trusted_cross,
            dtype=int
        )


    V = vectorization_ics_groupwise_fast(
        original_dist,
        nn_neighbors,
        torch.as_tensor(same_cam_and_id_cpu).to(dev),  # 传tensor
        torch.as_tensor(cross_cam_cpu).to(dev),  # 传tensor
        alpha=0.3,
        tau_intra=args.tau_intra
    )

    enable_BQE = True
    if enable_BQE:
        print('--------- Executing ICS-BQE -----------')

        beta = getattr(args, 'beta', 0.5)
        print('beta as', beta)

        V_qe = torch.zeros_like(V)

        for i in range(N):
            k2nn = nn_neighbors[i]

            if k2nn is None or len(k2nn) == 0:
                V_qe[i] = V[i]
                continue

            k2nn_t = torch.as_tensor(k2nn, device=dev)

            is_intra = same_cam_and_id[i, k2nn_t]
            is_cross = cross_cam[i, k2nn_t]

            intra_idx = k2nn_t[is_intra]
            cross_idx = k2nn_t[is_cross]

            has_intra = (len(intra_idx) > 0)
            has_cross = (len(cross_idx) > 0)

            if has_intra and has_cross:
                # 两者都有：取各自均值后，按 beta 进行动态融合
                v_intra_mean = V[intra_idx].mean(dim=0)
                v_cross_mean = V[cross_idx].mean(dim=0)
                V_qe[i] = beta * v_intra_mean + (1.0 - beta) * v_cross_mean
            elif has_intra:
                V_qe[i] = V[intra_idx].mean(dim=0)
            elif has_cross:
                V_qe[i] = V[cross_idx].mean(dim=0)

        V = V_qe  # 更新 V 为平衡扩展后的特征
        # -----------------------------------------------------------


    # ---------- final jaccard ----------
    V_cpu = V.detach().cpu().numpy()
    jaccard_dist = v2jaccard_numpy_chunked(V_cpu, chunk_size=256)

    t1 = time.time()
    print(f"GPU ICS-CKRNN Jaccard finished in {t1 - t0:.3f}s (N={N})")
    return jaccard_dist


