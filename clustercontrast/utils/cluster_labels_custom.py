from clustercontrast.evaluators import extract_vit_features
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn import metrics
from clustercontrast.utils.custom_propagation import greedy_camera_constrained_associate
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance


def img_association_optimized(args, model, propagate_loader, id_count_each_cam, rerank=False):
    print(f'[img_association] dataset={args.dataset}, rerank={rerank}')

    # 特征提取
    get_features, features, gt_labels, cam_ID, intra_cam_pID, global_labels = extract_vit_features(model, propagate_loader)

    # 按 global_label 聚类 + 聚合特征
    new_features, new_cams_ID, new_gt_IDs = [], [], []
    for glab in np.unique(global_labels):
        idx = np.where(global_labels == glab)[0]
        center = np.mean(features[idx], axis=0)
        weights = np.exp(-np.linalg.norm(features[idx] - center, axis=1))
        weights = weights / (np.sum(weights) + 1e-6)
        weighted_feature = np.sum(features[idx] * weights[:, None], axis=0)
        new_features.append(weighted_feature)

        new_cams_ID.append(cam_ID[idx[0]])
        new_gt_IDs.append(gt_labels[idx[0]])

    new_features = np.array(new_features)
    new_cams = np.array(new_cams_ID)
    new_gt_IDs = np.array(new_gt_IDs)

    # 构造距离矩阵
    if rerank:
        print('[img_association] Using Jaccard 距离')
        W = compute_jaccard_distance(torch.from_numpy(new_features), k1=20, k2=6)
    else:
        print('[img_association] Using Euclidean 距离')
        W = cdist(new_features, new_features, 'sqeuclidean')

    print(f'[img_association] W shape: {W.shape}')

    # 带 Constraint 1 和 2 的标签传播
    max_unions = int(args.associate_rate * np.sum(id_count_each_cam))
    print(f'max_unions is {max_unions}')
    updated_label = greedy_camera_constrained_associate(
        W=W,
        cam_ids=new_cams,
        topk=args.topK,
        max_unions=max_unions
    )

    print(f'[img_association] updated_label: len={len(updated_label)}, min={np.min(updated_label)}, max={np.max(updated_label)}')

    # 伪标签质量评估
    pred_label = updated_label[global_labels]
    index = np.where(pred_label >= 0)[0]
    print(f'[img_association] valid indices: {len(index)}')

    ari_value = metrics.adjusted_rand_score(gt_labels[index], pred_label[index])
    nmi_value = metrics.normalized_mutual_info_score(gt_labels[index], pred_label[index])

    print(f'[img_association] Pseudo label quality -> ARI: {ari_value:.4f}, NMI: {nmi_value:.4f}')

    if args.wandb_enabled:
        import wandb
        wandb.log({'ARI': ari_value, 'NMI': nmi_value})

    return updated_label, get_features, global_labels, cam_ID, intra_cam_pID, global_labels, new_gt_IDs
