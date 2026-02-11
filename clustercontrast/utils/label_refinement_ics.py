from torch import Tensor
import torch
import torch.nn.functional as F

import copy

import numpy as np

from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from collections import Counter
from collections import defaultdict

import wandb
import warnings
import time
import math
from typing import Optional, Union, Any


def k_reciprocal_neigh(initial_rank, i, k1):
    # 判断哪些样本既是第i个样本的k近邻，同时第i个样本也是它们的k近邻，形成双向的邻居关系。

    # 步骤1: 获取目标样本i的前k1+1个最近邻
    forward_k_neigh_index = initial_rank[i,:k1+1]

    # 步骤2: 获取这些邻居的前k1+1个最近邻
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]

    # 步骤3: 找出哪些邻居的邻居列表中包含目标样本i
    fi = np.where(backward_k_neigh_index==i)[0]

    # 步骤4: 返回与目标样本i形成互近邻关系的样本
    return forward_k_neigh_index[fi]



# ---------- helper: v2jaccard in numpy but chunked to reduce peak memory ----------
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


# ---------- main function ----------
def compute_ics_ckrnn_jaccard_distance(features: torch.Tensor,
                                       cam_labels: torch.Tensor,
                                       intra_camera_labels: torch.Tensor,
                                       epoch: int,
                                       args: Any,
                                       max_memory_mode: bool = False) -> np.ndarray:

    t0 = time.time()

    device = args.device if hasattr(args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(device)

    # --- prepare tensors on device
    feat = features.to(dev)
    cam = cam_labels.to(dev)
    intra_id = intra_camera_labels.to(dev)

    # convert to float32 for stable matmuls; if half precision desired, user may pass float16
    if feat.dtype == torch.float16:
        feat = feat.float()

    N = feat.shape[0]
    assert N == cam.shape[0] == intra_id.shape[0]

    # ---------- determine k search (how many top per row we examine) ----------
    k1 = args.k1
    k1_id = args.k1_id
    k1_intra = args.k1_intra
    k1_inter = args.k1_inter

    # k_search: we keep a bit more than k1 for reciprocity checks
    factor = args.k_search_factor
    k_search = min(N, int(math.ceil(k1 * factor)) + 1)  # +1 to include self

    print(f'\n获取超参数为 k1: {k1},  k1_id: {k1_id}, k1_intra: {k1_intra}, k1_inter: {k1_inter}, factor: {factor},  k_search: {k_search}\n')

    # normalize if not already
    # assume user may pass normalized; optionally normalize here:
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)

    # ---------- compute original distance (cosine -> distance) ----------
    # using cosine distance: original_dist = 2 - 2 * (feat @ feat.T)，keep as float32 on GPU
    with torch.no_grad():
        sim = torch.matmul(feat, feat.t())  # (N,N)
        original_dist = 2.0 - 2.0 * sim
        original_dist = torch.clamp(original_dist, min=0.0)  # numeric safety

    # ---------- masks ----------
    cam_mask = cam.view(-1, 1) == cam.view(1, -1)   # (N,N) bool on GPU
    id_mask = intra_id.view(-1, 1) == intra_id.view(1, -1)  # (N,N) bool on GPU

    # ---------- compute camera diff log ----------
    with torch.no_grad():
        # use upper triangle to compute averages (move small slices to CPU)
        # do this on CPU to avoid large GPU->CPU transfer? it's small slices though
        # convert to cpu numpy slices
        mask_cpu = cam_mask.cpu().numpy()

        orig_cpu = original_dist.cpu().numpy()
        tri_inter = orig_cpu[np.triu(~mask_cpu, k=1)]
        tri_intra = orig_cpu[np.triu(mask_cpu, k=1)]
        if tri_inter.size > 0 and tri_intra.size > 0:
            cam_diff = tri_inter.mean() - tri_intra.mean()
        else:
            cam_diff = 0.0
    print(f"摄像头差异Cam Diff: {cam_diff:.4f}")

    # ---------- priority penalty construction ----------
    # Priorities: same_cam & same_id (best), cross-camera (2nd), same_cam & diff_id (least)
    # We'll set penalties such that best group gets +0, 2nd gets +P2, 3rd gets +P3 (P3 > P2)
    P2 = 2000.0    # penalty for cross-camera (placed after sameID)
    P3 = 4000.0    # penalty for same-camera-different-ID (placed last)

    same_cam_and_id = cam_mask & id_mask
    same_cam_diff_id = cam_mask & (~id_mask)
    cross_cam = ~cam_mask

    # penalty matrix on GPU (float32)
    penalty = torch.zeros_like(original_dist, dtype=torch.float32, device=dev)
    penalty = penalty + (cross_cam.float() * P2)  # 跨摄像头（cross-camera） penalty：小的正值 P2
    penalty = penalty + (same_cam_diff_id.float() * P3)  # ICS 关键改动： 同摄像头但不同ID —— 完全禁止作为邻居， INF = 1e6

    dist_priority = original_dist + penalty  # same_cam_and_id 不加 penalty（最优先邻居）

    # ---------- get priority_rank: top-k_search per row (GPU) ----------
    # torch.topk returns largest by default; use largest=False to get smallest distances
    # we want indices sorted by distance ascending
    topk_vals, topk_idx = torch.topk(dist_priority, k=k_search, largest=False, sorted=True)  # (N, k_search)
    # topk_idx on GPU

    # ---------- build a topk boolean mask (sparse-like) ----------
    # We want a boolean matrix topk_mask where topk_mask[i,j]=True if j in topk_idx[i]
    # Instead of allocating full N*N bool (which may be huge), we create a sparse representation:
    # But for reciprocity check we need to test membership both ways. Here we build a dense bool if N moderate.
    build_dense_mask = not max_memory_mode and (N <= 18000)  # heuristic threshold; tune to your GPU memory

    if build_dense_mask:
        # dense boolean matrix on GPU
        topk_mask = torch.zeros((N, N), dtype=torch.bool, device=dev)
        rows = torch.arange(N, device=dev).unsqueeze(1).repeat(1, k_search).reshape(-1)
        cols = topk_idx.reshape(-1)
        topk_mask[rows, cols] = True
        # reciprocal mask:
        reciprocal_mask = topk_mask & topk_mask.t()  # (N,N) bool, True if both include each other in topk
    else:
        # Memory-conservative representation: store topk lists and build a dict-like structure on CPU (slower)
        # Build topk lists on CPU for reciprocity checks (rely on numpy sets)
        topk_idx_cpu = topk_idx.cpu().numpy()
        reciprocal_list = [None] * N  # will hold numpy boolean arrays of length k_search for each row
        # Build a mapping from j -> set(rows where j in topk of row)
        # To avoid huge loops, we build inverse lists:
        inv_lists = [[] for _ in range(N)]
        for i in range(N):
            for j in topk_idx_cpu[i]:
                inv_lists[j].append(i)
        # convert to numpy arrays for membership checks
        topk_mask = None
        reciprocal_mask = None
        # We'll use inv_lists and topk_idx_cpu for per-row reciprocity queries below
        # (this code path is slower but reduces GPU memory)
    # ---------- compute k-reciprocal neighbors and build nn_k1 ----------
    # We'll create nn_k1 as list of numpy arrays for further processing
    nn_k1 = [None] * N

    print(f'Build dense mask ->>> {build_dense_mask}')

    if build_dense_mask:
        # vectorized-ish with a python loop over rows, but membership checks are GPU boolean ops
        # to minimize Python overhead, we fetch tensors to CPU in moderately sized batches if needed
        # We'll do per-row but with GPU boolean indexing
        for i in range(N):
            pr = topk_idx[i]  # GPU tensor length k_search
            # get reciprocity boolean for this row among its candidates:
            recips_bool = reciprocal_mask[i, pr].cpu().numpy()  # small (k_search,)
            pr_cpu = pr.cpu().numpy()
            recips = pr_cpu[recips_bool]

            # Ensure self present; pr usually contains self at pos 0
            if i not in recips:
                recips = np.append(recips, i)
            # preserve priority order: pr_cpu is ordered by priority
            ordered_recips = [int(x) for x in pr_cpu if x in recips]
            ordered_recips = np.array(ordered_recips, dtype=int)
            if ordered_recips.size >= k1:
                chosen = ordered_recips[:k1]
            else:
                # supplement from priority list (non-reciprocal) to fill
                supplement = [int(x) for x in pr_cpu if x not in ordered_recips]
                need = k1 - ordered_recips.size
                if len(supplement) > 0 and need > 0:
                    supplement = np.array(supplement[:need], dtype=int)
                    chosen = np.concatenate([ordered_recips, supplement])
                else:
                    chosen = ordered_recips

            nn_k1[i] = np.unique(chosen)
    else:
        # memory-conservative path using topk_idx_cpu and inv_lists
        topk_idx_cpu = topk_idx_cpu  # alias
        for i in range(N):
            pr = topk_idx_cpu[i]
            recips = []
            for cand in pr:
                # check if i in inv_lists[cand] -> if cand's topk contains i
                # inv_lists[cand] is a list of indices whose topk include cand
                # membership test:
                if i in inv_lists[cand]:
                    recips.append(cand)
            if i not in recips:
                recips.append(i)
            # preserve priority order from pr
            ordered_recips = [c for c in pr if c in recips]
            ordered_recips = np.array(ordered_recips, dtype=int)
            if ordered_recips.size >= k1:
                chosen = ordered_recips[:k1]
            else:
                supplement = [c for c in pr if c not in ordered_recips]
                need = k1 - ordered_recips.size
                if len(supplement) > 0 and need > 0:
                    supplement = np.array(supplement[:need], dtype=int)
                    chosen = np.concatenate([ordered_recips, supplement])
                else:
                    chosen = ordered_recips
            nn_k1[i] = np.unique(chosen)

    # ---------- build V on GPU ----------
    # We'll build V by filling only selected columns per row
    V = torch.zeros((N, N), dtype=torch.float32, device=dev)
    # To reduce kernel launches we handle in batches
    batch_size = 512 if N > 2000 else N
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        rows = range(start, end)
        # collect unique column indices in this batch to index original_dist once
        cols_set = set()
        for i in rows:
            cols_set.update(nn_k1[i].tolist() if isinstance(nn_k1[i], np.ndarray) else [int(nn_k1[i])])
        cols = np.array(sorted(cols_set), dtype=int)
        if cols.size == 0:
            continue
        cols_t = torch.from_numpy(cols).to(dev)
        # fetch distances submatrix original_dist[rows, cols]
        dsub = original_dist[torch.tensor(list(rows), device=dev).unsqueeze(1), cols_t.unsqueeze(0)]
        # for each row build softmax weights along its nn_k1 entries
        for idx_row_local, i in enumerate(rows):
            chosen = nn_k1[i]
            if chosen is None or chosen.size == 0:
                continue
            # map chosen indices to positions within cols
            mask_positions = np.searchsorted(cols, chosen)  # because cols is sorted
            d_row = dsub[idx_row_local, mask_positions]  # distances on GPU
            w = torch.nn.functional.softmax(-d_row, dim=0)  # 1D
            V[i, torch.from_numpy(chosen).to(dev)] = w

    # normalize rows to sum to 1 (avoid division by zero)
    row_sums = V.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    V = V / row_sums

    # #################################################
    # ---------- Query Expansion (k2) on GPU ----------
    # #################################################
    # We'll implement CLQE: use separate rank matrices for id/intra/inter similar to earlier plan
    if epoch == 0:
        # warm-up small k2
        k2_id = k2_intra = k2_inter = 3
    else:
        k2_id = args.k2_id
        k2_intra = args.k2_intra
        k2_inter = args.k2_inter

    print(f'获取超参数为 k2_id: {k2_id},  k2_intra: {k2_intra},  k2_inter: {k2_inter}')

    if args.clqe:  # 似乎不工作，待查
        print('--------- CLQE -----------')
        # prepare three rank matrices (top k for each type)
        # create specialized penalty to enforce type-specific ordering and call topk
        big = 9999.0

        # id-first ranking (penalize non same_cam&same_id)
        mask_ok_id = (same_cam_and_id).float()
        dist_id = original_dist + big * (1.0 - mask_ok_id)
        _, rank_id = torch.topk(dist_id, k=(k2_id if k2_id + 1 < N else N), largest=False, sorted=True)

        # intra (same camera, different id)
        mask_ok_intra = (same_cam_diff_id).float()
        dist_intra = original_dist + big * (1.0 - mask_ok_intra)
        _, rank_intra = torch.topk(dist_intra, k=(k2_intra if k2_intra + 1 < N else N), largest=False, sorted=True)

        # inter (cross camera)
        mask_ok_inter = (cross_cam).float()
        dist_inter = original_dist + big * (1.0 - mask_ok_inter)
        _, rank_inter = torch.topk(dist_inter, k=(k2_inter if k2_inter + 1 < N else N), largest=False, sorted=True)

        # V_qe = mean( V[ k2nn, : ] ) for k2nn combined
        V_qe = torch.zeros_like(V)
        for i in range(N):
            # gather top neighbors (as CPU indices then to dev)
            k2nn = torch.cat([rank_id[i, :k2_id], rank_intra[i, :k2_intra], rank_inter[i, :k2_inter]]).unique()
            if k2nn.numel() == 0:
                continue
            V_qe[i] = V[k2nn].mean(dim=0)
        V = V_qe
    else:
        # simple LQE: average top-k2 rows in global ranking
        print('--------- LQE -----------')
        if args.k2 > 1:
            # compute global top-k2 from original_dist
            _, global_rank = torch.topk(original_dist, k=(getattr(args, 'k2', 6) if getattr(args, 'k2', 6) < N else N), largest=False, sorted=True)
            V_qe = torch.zeros_like(V)
            for i in range(N):
                k2nn = global_rank[i, :getattr(args, 'k2', 6)]
                V_qe[i] = V[k2nn].mean(dim=0)
            V = V_qe

    # ---------- move V to CPU in chunks and compute final jaccard (numpy chunked) ----------
    # V is float32 on device; move to cpu as numpy in chunks to avoid memory spike
    V_cpu = V.detach().cpu().numpy()  # NOTE: if N huge, consider saving sparse representation and streaming
    # use chunked numpy v2jaccard
    jaccard_dist = v2jaccard_numpy_chunked(V_cpu, chunk_size=256)

    t1 = time.time()
    print(f"GPU ICS-CKRNN Jaccard finished in {t1 - t0:.3f}s (N={N})")
    return jaccard_dist
