import torch
from typing import List, Dict

def fedavg(gradients_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """FedAvg: 普通联邦平均 (无防御)"""
    if not gradients_list:
        return {}
    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}
    for name in param_names:
        grads = [client_grad['gradients'][name] for client_grad in gradients_list]
        aggregated_gradients[name] = torch.stack(grads).mean(dim=0)
    return aggregated_gradients


def coordinate_wise_median(gradients_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    坐标级中位数聚合 (Coordinate-wise Median)。
    对于每个参数位置，取所有客户端梯度对应位置的中位数。
    """
    if not gradients_list:
        return {}

    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}

    for name in param_names:
        grads = [client_grad['gradients'][name] for client_grad in gradients_list]
        stacked_grads = torch.stack(grads, dim=0)  # (num_clients, *param_shape)

        # 计算坐标级中位数
        median_val = torch.median(stacked_grads, dim=0).values

        aggregated_gradients[name] = median_val

    return aggregated_gradients


def coordinate_wise_trimmed_mean(gradients_list: List[Dict], beta: float = 0.1) -> Dict[str, torch.Tensor]:
    """Definition 2: Coordinate-wise trimmed mean"""
    if not gradients_list:
        return {}

    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}
    m = len(gradients_list)
    k = int(m * beta)

    if 2 * k >= m:
        raise ValueError(f"Beta {beta} is too large for the number of clients {m}.")

    for name in param_names:
        grads = [client_grad['gradients'][name] for client_grad in gradients_list]
        stacked_grads = torch.stack(grads, dim=0)

        sorted_grads, _ = torch.sort(stacked_grads, dim=0)

        if k > 0:
            trimmed_grads = sorted_grads[k : m - k]
        else:
            trimmed_grads = sorted_grads

        mean_val = torch.mean(trimmed_grads, dim=0)
        aggregated_gradients[name] = mean_val

    return aggregated_gradients


def _compute_distances(grads_dict: Dict[int, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """计算所有梯度之间的欧氏距离矩阵"""
    ids = list(grads_dict.keys())
    m = len(ids)

    flat_grads = []
    for id_ in ids:
        grad_vec = torch.cat([grads_dict[id_][name].flatten() for name in grads_dict[id_].keys()])
        flat_grads.append(grad_vec)

    distances = torch.zeros(m, m)
    for i in range(m):
        for j in range(m):
            if i != j:
                distances[i, j] = torch.sum((flat_grads[i] - flat_grads[j]) ** 2)

    return distances


def multi_krum(gradients_list: List[Dict], n_remove: int = 4, n_select: int = 5) -> Dict[str, torch.Tensor]:
    """
    修正后的 Multi-Krum 聚合方法。
    算法逻辑：
    1. 为每个客户端计算其到最近的 (m - n_remove - 2) 个邻居的距离之和作为得分。
    2. 选取得分最低的前 n_select 个客户端。
    3. 将这 n_select 个客户端的梯度求平均。
    """
    if not gradients_list:
        return {}

    m = len(gradients_list)
    num_neighbors = max(1, m - n_remove - 2)
    n_select = min(n_select, m)

    grads_dict = {g['id']: g['gradients'] for g in gradients_list}
    distances = _compute_distances(grads_dict)

    scores = []
    for i in range(m):
        sorted_dists, _ = torch.sort(distances[i])
        score = torch.sum(sorted_dists[:num_neighbors+1]).item()
        scores.append(score)

    scores_tensor = torch.tensor(scores)
    _, selected_indices = torch.topk(scores_tensor, n_select, largest=False)

    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}
    for name in param_names:
        selected_grads = [gradients_list[i]['gradients'][name] for i in selected_indices]
        aggregated_gradients[name] = torch.stack(selected_grads).mean(dim=0)

    return aggregated_gradients


import torch.nn.functional as F
from typing import List, Dict

def _agglomerative_clustering_complete_linkage(distances_mat: torch.Tensor, n_clusters: int = 2) -> List[List[int]]:
    """
    原生实现：带有全连接 (Complete Linkage) 的凝聚层次聚类。
    输入距离矩阵，输出分簇的索引列表。
    """
    m = distances_mat.shape[0]
    # 初始时，每个节点自成一簇
    clusters = [[i] for i in range(m)]

    while len(clusters) > n_clusters:
        min_dist = float('inf')
        merge_indices = (-1, -1)

        # 遍历所有可能的簇对
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_i_idx = clusters[i]
                cluster_j_idx = clusters[j]

                # Complete Linkage: 两个簇之间的距离等于它们成员之间距离的 最大值
                max_d = 0.0
                for idx_i in cluster_i_idx:
                    for idx_j in cluster_j_idx:
                        d = distances_mat[idx_i, idx_j].item()
                        if d > max_d:
                            max_d = d

                # 寻找合并距离最小的两个簇
                if max_d < min_dist:
                    min_dist = max_d
                    merge_indices = (i, j)

        # 合并这两个簇
        i, j = merge_indices
        clusters[i].extend(clusters[j])
        clusters.pop(j) # 删除被合并的簇

    return clusters

def clustering_based(gradients_list: List[Dict], n_clusters: int = 2) -> Dict[str, torch.Tensor]:
    """
    严格实现图片算法：
    1. 计算成对的余弦距离 (Cosine Distance)
    2. 使用全连接凝聚聚类 (Agglomerative Clustering with Complete Linkage) 划分为两簇
    3. 取样本数最多的簇 (S_max) 进行均值聚合
    """
    if not gradients_list:
        return {}

    m = len(gradients_list)
    # 如果节点数太少，无法有效聚类，直接退化为 FedAvg
    if m <= 2:
        return fedavg(gradients_list)

    # 1. 扁平化所有客户端的梯度
    param_names = list(gradients_list[0]['gradients'].keys())
    flat_grads = []
    for client_grad in gradients_list:
        grad_vec = torch.cat([client_grad['gradients'][name].flatten() for name in param_names])
        flat_grads.append(grad_vec)
    
    # 形状: (m, d)，m 是客户端数量，d 是参数总维度
    X = torch.stack(flat_grads)

    # 2. 计算成对余弦距离 (Cosine Distance)
    # 公式: d_{i,j} = 1 - (g_i \cdot g_j) / (||g_i|| * ||g_j||)
    # 使用 unsqueeze 广播机制计算全连接相似度矩阵
    cos_sim = F.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=2)
    # 限制范围避免计算浮点误差导致的极小负数，并转为距离
    distances = 1.0 - torch.clamp(cos_sim, -1.0, 1.0)

    # 对角线上的距离严格置为 0
    distances.fill_diagonal_(0.0)

    # 3. 执行全连接凝聚聚类，分为 n_clusters=2 簇
    clusters = _agglomerative_clustering_complete_linkage(distances, n_clusters=2)

    # 4. 找到包含元素最多的簇 S_max
    # 如果两个簇大小刚好一样，默认选第一个
    s_max = max(clusters, key=len)

    # 5. 取 S_max 中的梯度进行平均聚合
    aggregated_gradients = {}
    for name in param_names:
        selected_grads = [gradients_list[i]['gradients'][name] for i in s_max]
        aggregated_gradients[name] = torch.stack(selected_grads).mean(dim=0)

    return aggregated_gradients
