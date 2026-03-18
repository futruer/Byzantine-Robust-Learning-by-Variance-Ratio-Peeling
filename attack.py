import random
import torch
from torch.utils.data import DataLoader, Dataset

class LabelFlippedDataset(Dataset):
    """
    自定义 Dataset 包装器，用于在读取数据时动态翻转标签。
    对应逻辑：D[[k]]$Y <- 9 - D[[k]]$Y
    """
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        data, target = self.original_dataset[index]
        # 标签翻转操作: 9 - target
        return data, 9 - target

    def __len__(self):
        return len(self.original_dataset)

class TargetedLabelFlippedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        data, target = self.original_dataset[index]
        # 定向翻转：把 1 变成 7，把 3 变成 8
        if target == 1:
            target = torch.tensor(7)
        elif target == 2:
            target = torch.tensor(9)
        elif target == 3:
            target = torch.tensor(8)
        return data, target

    def __len__(self):
        return len(self.original_dataset)

def cnn_rev_attack(client_data_list: list, alpha: float) -> list:
    """
    实现标签翻转攻击 (Label Flipping Attack)。

    Args:
        client_data_list: get_mnist_client_dataloaders 的返回值，
                          格式为 [{"id": int, "dataloader": DataLoader}, ...]
        alpha: 攻击者比例 (0.0 ~ 1.0)

    Returns:
        attacked_client_data_list: 包含攻击后 dataloader 的新列表
    """
    # 1. 设置随机种子
    random.seed(42)

    # 2. 计算攻击者数量 K * alpha
    K = len(client_data_list)
    num_attackers = int(K * alpha)

    # 3. 随机选择攻击者索引
    attacker_indices = set(random.sample(range(K), num_attackers))

    attacked_client_data_list = []

    for i, client_dict in enumerate(client_data_list):
        new_dict = client_dict.copy()

        # 4. 如果当前索引被选中，则进行标签翻转
        if i in attacker_indices:
            original_loader = client_dict['dataloader']
            original_dataset = original_loader.dataset
            batch_size = original_loader.batch_size

            # 包装 Dataset 实现标签翻转
            flipped_dataset = LabelFlippedDataset(original_dataset)

            # 创建新的 DataLoader 加载翻转后的数据
            new_loader = DataLoader(flipped_dataset, batch_size=batch_size, shuffle=True)

            new_dict['dataloader'] = new_loader
            new_dict['is_attacker'] = True
        else:
            new_dict['is_attacker'] = False

        attacked_client_data_list.append(new_dict)

    return attacked_client_data_list


# ==========================================
# 梯度层面的攻击方法 (用于实验对比)
# ==========================================

def gaussian_attack(gradients_list: list, sigma: float = 1.0, seed: int = 42) -> list:
    """
    高斯噪声攻击 (Gaussian Attack)。
    在攻击者的梯度上添加高斯噪声。

    Args:
        gradients_list: 梯度列表，格式为 [{'id': int, 'gradients': {name: tensor}, 'is_attacker': bool}, ...]
        sigma: 高斯噪声的标准差
        seed: 随机种子

    Returns:
        添加噪声后的梯度列表
    """
    random.seed(seed)
    torch.manual_seed(seed)

    attacked_gradients_list = []

    for client_grad in gradients_list:
        new_grad = client_grad.copy()
        new_grad['gradients'] = {}

        if client_grad.get('is_attacker', False):
            # 对每个参数添加高斯噪声
            for name, grad in client_grad['gradients'].items():
                noise = torch.randn_like(grad) * sigma
                new_grad['gradients'][name] = grad + noise
        else:
            new_grad['gradients'] = client_grad['gradients'].copy()

        attacked_gradients_list.append(new_grad)

    return attacked_gradients_list

def sign_flipping_attack(gradients_list: list, factor: float = -1.0) -> list:
    """
    符号翻转攻击 (Sign Flipping Attack)。
    将攻击者的梯度乘以一个负数因子。

    Args:
        gradients_list: 梯度列表
        factor: 翻转因子，默认 -1.0

    Returns:
        翻转后的梯度列表
    """
    attacked_gradients_list = []

    for client_grad in gradients_list:
        new_grad = client_grad.copy()
        new_grad['gradients'] = {}

        if client_grad.get('is_attacker', False):
            # 翻转符号
            for name, grad in client_grad['gradients'].items():
                new_grad['gradients'][name] = grad * factor
        else:
            new_grad['gradients'] = client_grad['gradients'].copy()

        attacked_gradients_list.append(new_grad)

    return attacked_gradients_list


def scaling_attack(gradients_list: list, factor: float = 10.0) -> list:
    """
    缩放攻击 (Scaling Attack)。
    将攻击者的梯度放大或缩小一定倍数。

    Args:
        gradients_list: 梯度列表
        factor: 缩放因子，大于1表示放大，小于1表示缩小

    Returns:
        缩放后的梯度列表
    """
    attacked_gradients_list = []

    for client_grad in gradients_list:
        new_grad = client_grad.copy()
        new_grad['gradients'] = {}

        if client_grad.get('is_attacker', False):
            # 缩放梯度
            for name, grad in client_grad['gradients'].items():
                new_grad['gradients'][name] = grad * factor
        else:
            new_grad['gradients'] = client_grad['gradients'].copy()

        attacked_gradients_list.append(new_grad)

    return attacked_gradients_list


def omniscient_ipm_attack(gradients_list: list, epsilon: float = 100.0) -> list:
    """
    全知 IPM 攻击 (Omniscient Inner Product Manipulation Attack)。

    公式: g_malicious = -epsilon * mean(g_benign)

    攻击者拥有"上帝视角"，能够获取所有正常节点的梯度信息，
    然后将自己的梯度设置为正常节点梯度的负平均值，从而精确地
    站在系统真实优化方向的对立面。

    Args:
        gradients_list: 梯度列表，每个元素包含 'id', 'gradients', 'is_attacker' 字段
        epsilon: 攻击强度因子，默认 100.0

    Returns:
        攻击后的梯度列表
    """
    if gradients_list is None or len(gradients_list) == 0:
        return gradients_list

    # 收集正常节点的梯度 (is_attacker=False 表示正常节点)
    benign_gradients = []
    for g in gradients_list:
        if not g.get('is_attacker', False):
            benign_gradients.append(g['gradients'])

    # 如果没有正常节点，无法执行 IPM 攻击
    if len(benign_gradients) == 0:
        print("[Warning] IPM Attack: No benign gradients found!")
        return gradients_list

    # 计算正常节点梯度的平均值
    benign_mean = {}
    param_names = benign_gradients[0].keys()
    for name in param_names:
        stacked = torch.stack([grads[name] for grads in benign_gradients])
        benign_mean[name] = torch.mean(stacked, dim=0)

    # 对恶意节点执行 IPM 攻击
    attacked_gradients_list = []
    for g in gradients_list:
        if g.get('is_attacker', False):
            # 恶意节点：使用 -epsilon * mean 替换梯度
            new_grad = {}
            for name in g['gradients']:
                new_grad[name] = -epsilon * benign_mean[name]
            attacked_gradients_list.append(new_grad)
        else:
            # 正常节点：保持原梯度
            attacked_gradients_list.append(g['gradients'])

    # 构建返回列表，保持原始顺序
    result_list = []
    for i, g in enumerate(gradients_list):
        result_list.append({
            'id': g['id'],
            'gradients': attacked_gradients_list[i],
            'loss': g.get('loss', 0.0),
            'is_attacker': g.get('is_attacker', False)
        })

    return result_list
