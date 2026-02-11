# tools/custom_propagation.py

import numpy as np

class UnionFindCS:
    def __init__(self, n, cam_ids):  # 初始化n个节点
        self.parent = list(range(n))
        self.rank = [0] * n
        self.cam_sets = [{cam_ids[i]} for i in range(n)]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return False
        if self.cam_sets[rx] & self.cam_sets[ry]: return False  # Constraint 2
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.cam_sets[rx].update(self.cam_sets[ry])
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

def greedy_camera_constrained_associate(W, cam_ids, topk=30, max_unions=None):
    # W: 样本间的距离矩阵 (N x N)，越小越相似
    # cam_ids: 每个样本对应的摄像头编号
    # topk: 每个样本仅保留来自不同摄像头的 top-k 最相似的候选样本
    # max_unions: 跨摄像头的最大合并次数（防止过度聚合）

    N = W.shape[0]
    edges = []
    for i in range(N):
        mask = np.where(cam_ids != cam_ids[i])[0]  # mask: 仅选择不同摄像头的样本
        top_idx = mask[np.argsort(W[i][mask])[:topk]]  # 在这些候选中，选出 top-k 最相似的样本

        for j in top_idx:
            edges.append((W[i, j], i, j))  # # 将边 (距离, 样本i, 样本j) 加入候选集合

    edges.sort()  # 按照距离从小到大排序，优先合并最相似的样本对

    # union 过程中聚合  # 使用 Union-Find 结构进行跨摄像头 ID 合并
    uf = UnionFindCS(N, cam_ids)
    unions = 0
    for dist, i, j in edges:
        if uf.union(i, j):  # -------->>>  # 如果 i, j 成功合并（之前不在同一集合
            unions += 1
            if max_unions and unions >= max_unions:  # 如果合并次数达到 max_unions 上限，就停止
                break

    print(f'max_unions -> {max_unions}')

    # 重标号 relabel
    # 重新编号 (relabel)，把每个集合的 root id 映射为新的 ID
    label_map, new_label = {}, 0
    labels = np.zeros(N, dtype=int)
    for i in range(N):
        root = uf.find(i)  # -------->>> find
        if root not in label_map:
            label_map[root] = new_label
            new_label += 1
        labels[i] = label_map[root]
    return labels
