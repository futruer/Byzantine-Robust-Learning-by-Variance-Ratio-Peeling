import torch
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import List, Dict

def get_mnist_client_dataloaders(
    num_clients: int, 
    batch_size: int, 
    root: str = './data', 
    download: bool = True
) -> List[Dict]:
    """
    加载 MNIST 数据集，将其随机均匀分成 num_clients 份，并返回每个 client 的 dataloader。
    
    Args:
        num_clients (int): 客户端数量 (K)
        batch_size (int): 每个 client 的 batch size
        root (str): 数据存储路径
        download (bool): 是否自动下载数据

    Returns:
        List[Dict]: 格式为 [{"id": int, "dataloader": DataLoader}, ...]
    """
    
    # 1. 定义数据预处理 (Transforms)
    # 基于 pretrainCNN.ipynb 中的代码:
    # Normalize: (0.1307,) 和 (0.3081,) 是 MNIST 数据集的全局平均值和标准差。
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])

    # 2. 下载并加载数据集 (使用训练集 train=True)
    train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    
    total_len = len(train_dataset)
    
    # 3. 计算划分长度
    # 确保所有样本都被分配，且尽可能均匀
    base_len = total_len // num_clients
    remainder = total_len % num_clients
    
    lengths = []
    for i in range(num_clients):
        # 将余数分配给前几个 client，确保总数匹配
        length = base_len + 1 if i < remainder else base_len
        lengths.append(length)
        
    # 4. 随机划分数据集
    # 使用 generator 确保随机性，如果需要完全复现，可以在外部设置 torch.manual_seed
    subsets = random_split(train_dataset, lengths)
    
    # 5. 构建返回结果列表
    client_data_list = []
    
    for client_id, subset in enumerate(subsets):
        # 创建 DataLoader
        # shuffle=True 表示 client 内部训练时数据是乱序的
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        
        client_dict = {
            "id": client_id,
            "dataloader": loader
        }
        client_data_list.append(client_dict)
        
    print(f"Data process finished: Split MNIST ({total_len} samples) into {num_clients} clients.")
    return client_data_list

def cnn_reform_func(labels_features: tuple, init_theta: torch.Tensor) -> pd.DataFrame:
    """
    实现了 Byzantine Robust Learning 中的 Variance Ratio Peeling 核心逻辑。
    对应 R 语言代码中的 form_gH 和 form_YX_tlide 函数。
    
    Args:
        labels_features (tuple): (labels, features)。
            - labels: 1D Tensor (N,)
            - features: 2D Tensor (N, D), 这里 D=50
        init_theta (torch.Tensor): 初始参数张量。
            - 期望形状为 (10,) 或 (50, 10)。如果是 (10,) 会自动广播。

    Returns:
        pd.DataFrame: 包含 Y_tlide 和 X_tlide 的 DataFrame。
                      维度为 (m*p, m*p + 1)。
    """
    labels, features = labels_features
    
    # 确保在同一设备上
    device = features.device
    labels = labels.to(device)
    init_theta = init_theta.to(device)

    N, D = features.shape
    C = 10  # 类别数量
    
    # 1. 构造 theta_mat (D, C)
    # R代码中 theta_mat 用于计算 Softmax(X @ theta)
    if init_theta.dim() == 1 and init_theta.shape[0] == C:
        # 如果传入的是 (10,) 的中位数向量，将其扩展为 (50, 10)
        # 假设每一列(每个类别)的所有特征权重初始值相同
        theta_mat = init_theta.unsqueeze(0).expand(D, C)
    elif init_theta.numel() == D * C:
        # 如果传入的是展平的参数或完整矩阵，重塑为 (D, C)
        theta_mat = init_theta.view(D, C)
    else:
        # 兜底：全零初始化或其他逻辑
        theta_mat = torch.zeros(D, C, device=device)

    # 2. 计算 Softmax (P)
    # features: (N, D), theta_mat: (D, C) -> logits: (N, C)
    logits = features @ theta_mat
    P = torch.softmax(logits, dim=1)  # (N, C)

    # 3. 计算 Gradient Mean (g)
    # g = (P - Y_one_hot) \otimes X
    Y_one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=C).float() # (N, C)
    diff = P - Y_one_hot # (N, C)
    
    # 计算每个样本的梯度矩阵 (C, D) = (C, 1) * (1, D)
    # 最终取平均 -> (C, D)
    grad_per_sample = diff.unsqueeze(2) * features.unsqueeze(1) # (N, C, D)
    g_mean_mat = grad_per_sample.mean(dim=0) # (C, D)
    
    # 展平 g。R语言使用的是列优先逻辑(Class 1 params, Class 2 params...)
    # PyTorch flatten 是行优先，所以我们需要先转置
    g_mean = g_mean_mat.T.flatten() # (C*D,)

    # 4. 计算 Hessian Mean (H)
    # H = (diag(P) - P P^T) \otimes (X X^T)
    # 构造 P_term: (N, C, C)
    P_unsqueeze = P.unsqueeze(2) # (N, C, 1)
    P_term = torch.diag_embed(P) - (P_unsqueeze @ P_unsqueeze.transpose(1, 2))
    
    # 构造 X_term: (N, D, D)
    X_unsqueeze = features.unsqueeze(2) # (N, D, 1)
    X_term = X_unsqueeze @ X_unsqueeze.transpose(1, 2)
    
    # 计算 Kronecker 积的均值
    # 目标 H 形状 (C*D, C*D)。块结构：[C, C] 个块，每个块大小 [D, D]
    # 需要先计算四维张量 (N, C, C, D, D) 的均值 -> (C, C, D, D)
    # 然后调整维度顺序为 (C, D, C, D) 实际上对应的扁平化顺序是 (D, C, D, C) ?
    # R代码逻辑: theta 排序是 [d1c1, d2c1, ... d1c2...] (列优先)
    # 等等，R中: theta[((i-1)*p+1):(i*p)] <- theta_mat[,i]
    # 这是取第 i 列(第 i 类)。所以向量顺序是: [Class 1 Params, Class 2 Params...]
    # 我的 g_mean 构造也是 [Class 1 Params...].
    # 所以 H 的行/列索引应该是 [c, d] (Class c, Dim d)。
    # 这里的 c 是慢变维度? 不，R中向量是 c1d1, c1d2... c1dp. 
    # 所以 c 是慢变维度 (Outer block), d 是快变维度 (Inner block).
    # 这意味着 H 应该由 C x C 个块组成，每个块是 D x D。
    # 块 (c, c') 对应 P_term[c, c'] * X_term。
    
    # H_4d: (C, C, D, D)
    H_4d = (P_term.unsqueeze(3).unsqueeze(4) * X_term.unsqueeze(1).unsqueeze(2)).mean(dim=0)
    
    # 我们需要排列成 (C, D, C, D) 然后 reshape
    H_mean = H_4d.permute(0, 2, 1, 3).reshape(C*D, C*D)
    
    # 注意：如果 g_mean 是按 [c1_d1... c1_dD, c2_d1...] 排列的 (行优先?)
    # g_mean_mat 是 (C, D)。flatten() 后是 [c0d0, c0d1... c0dD, c1d0...]
    # 刚才我做了 g_mean_mat.T.flatten() -> [d0c0, d0c1...] 这反了！
    # 修正 g_mean:
    # R: 循环 j=1:m (类), 内部取 X[i,] (特征). 
    # 所以 R 的 g 是 [Class 1 all features, Class 2 all features...]。
    # PyTorch g_mean_mat (C, D) 的第 0 行就是 Class 0 所有特征。
    # g_mean_mat.flatten() 就是 [Class 0 all features, Class 1 all features...]。
    # 所以不需要转置。
    g_mean = g_mean_mat.flatten() # (C*D,)
    
    # 同理修正 H_mean:
    # 行索引应该是 (c, d)，c 慢变。
    # H_4d 维度是 (c, c', d, d')。
    # 我们需要行 (c, d)，列 (c', d')。
    # permute(0, 2, 1, 3) 变成 (c, d, c', d')。
    # reshape(C*D, C*D) 即可。
    # 逻辑正确。
    
    # 5. 特征分解 (Eigen Decomposition)
    # H 是对称矩阵
    L, V = torch.linalg.eigh(H_mean)
    
    # 处理小特征值
    mask = L > 1e-16
    L_safe = torch.where(mask, L, torch.tensor(0., device=device))
    
    # 计算 H^(-1/2) 和 H^(1/2)
    # 注意：特征值小于阈值时，inv 设为 0
    L_sqrt = torch.sqrt(L_safe)
    L_inv_sqrt = torch.where(mask, 1.0 / L_sqrt, torch.tensor(0., device=device))
    
    sqrtH = V @ torch.diag(L_sqrt) @ V.T
    sqrtH_inv = V @ torch.diag(L_inv_sqrt) @ V.T
    
    # 6. 计算 X_tlide, Y_tlide
    # theta 也需要展平为 [Class 1 all features...]，即 row-major flatten
    theta_flat = theta_mat.T.flatten() # 这里 theta_mat 是 (D, C)，转置后 (C, D)，展平符合顺序
    
    X_tlide = -sqrtH
    Y_tlide = (sqrtH_inv @ g_mean) - (sqrtH @ theta_flat)
    
    # 7. 构建 DataFrame
    # 转换为 Numpy
    Y_np = Y_tlide.detach().cpu().numpy()
    X_np = X_tlide.detach().cpu().numpy()
    
    # 合并：第一列 Y，其余列 X
    data = np.column_stack((Y_np, X_np))
    columns = ["Y_tlide"] + [f"X_tlide_{i+1}" for i in range(X_np.shape[1])]
    
    return pd.DataFrame(data, columns=columns)

# ==========================================
# 简单的测试代码
# ==========================================
if __name__ == "__main__":
    # 测试将数据分给 10 个 client
    K = 10
    BATCH_SIZE = 64
    
    clients_data = get_mnist_client_dataloaders(num_clients=K, batch_size=BATCH_SIZE)
    
    print(f"生成了 {len(clients_data)} 个 client 的数据")
    print(f"Client 0 的 batch 数量: {len(clients_data[0]['dataloader'])}")
    
    # 验证数据格式
    first_client = clients_data[0]
    print(f"Key check: {first_client.keys()}") # 应输出 dict_keys(['id', 'dataloader'])