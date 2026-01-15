import torch
from typing import List, Dict

def coordinate_wise_median(gradients_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    实现了 Definition 1: Coordinate-wise median。
    
    参数:
        gradients_list: 由 Orchestrator.train 传入的列表，
                        格式为 [{'id': id, 'gradients': {name: tensor}, 'loss': loss}, ...]
    返回:
        aggregated_gradients: 聚合后的梯度字典 {name: tensor}
    """
    if not gradients_list:
        return {}

    # 获取模型参数名称列表（从第一个客户端的数据中获取）
    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}

    for name in param_names:
        # 1. 收集所有客户端关于该参数(name)的梯度
        # client_grad['gradients'][name] 是对应的 tensor
        grads = [client_grad['gradients'][name] for client_grad in gradients_list]
        
        # 2. 堆叠: shape 变为 (m, *param_shape)，其中 m 是客户端数量
        stacked_grads = torch.stack(grads, dim=0)
        
        # 3. 计算坐标轴维度的中位数
        # torch.median 在指定维度返回 (values, indices) 元组，我们需要 values
        median_val = torch.median(stacked_grads, dim=0).values
        
        aggregated_gradients[name] = median_val

    return aggregated_gradients


def coordinate_wise_trimmed_mean(gradients_list: List[Dict], beta: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    实现了 Definition 2: Coordinate-wise trimmed mean。
    
    参数:
        gradients_list: 由 Orchestrator.train 传入的列表。
        beta: 修剪比例，范围 [0, 0.5)。默认设为 0.1。
              注意：如果在 Orchestrator.train 中直接作为 aggregation 参数传入，
              通常需要使用 functools.partial 固定 beta 值。
    返回:
        aggregated_gradients: 聚合后的梯度字典 {name: tensor}
    """
    if not gradients_list:
        return {}

    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}
    
    # 客户端总数 m
    m = len(gradients_list)
    # 计算需要移除的单侧样本数 k
    k = int(m * beta)

    # 安全性检查：确保修剪后至少剩下一个元素
    if 2 * k >= m:
        raise ValueError(f"Beta {beta} is too large for the number of clients {m}. No data remains after trimming.")

    for name in param_names:
        # 1. 收集并堆叠梯度
        grads = [client_grad['gradients'][name] for client_grad in gradients_list]
        stacked_grads = torch.stack(grads, dim=0) # (m, *param_shape)
        
        # 2. 在客户端维度 (dim=0) 上进行排序
        # sorted_grads shape: (m, *param_shape)
        sorted_grads, _ = torch.sort(stacked_grads, dim=0)
        
        # 3. 修剪 (Trim)
        # 去除最小的 k 个和最大的 k 个
        # 对应切片为 [k : m-k]
        if k > 0:
            trimmed_grads = sorted_grads[k : m - k]
        else:
            trimmed_grads = sorted_grads
            
        # 4. 求剩余元素的平均值 (Mean)
        mean_val = torch.mean(trimmed_grads, dim=0)
        
        aggregated_gradients[name] = mean_val

    return aggregated_gradients
