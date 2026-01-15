import torch
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
