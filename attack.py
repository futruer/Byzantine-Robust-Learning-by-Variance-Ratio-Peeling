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
    # 1. 设置随机种子 (对应 R 中的 set.seed(42))
    random.seed(42)
    
    # 2. 计算攻击者数量 K * alpha
    K = len(client_data_list)
    num_attackers = int(K * alpha)
    
    # 3. 随机选择攻击者索引 (bi)
    # random.sample 进行无放回采样
    attacker_indices = set(random.sample(range(K), num_attackers))
    
    attacked_client_data_list = []
    
    for i, client_dict in enumerate(client_data_list):
        # 浅拷贝字典，避免直接修改原列表中的对象引用（虽然 dataloader 是对象，但替换它需要新 key）
        new_dict = client_dict.copy()
        
        # 4. 如果当前索引被选中，则进行标签翻转
        if i in attacker_indices:
            original_loader = client_dict['dataloader']
            
            # 获取原始 Dataset (通常是 Subset)
            original_dataset = original_loader.dataset
            batch_size = original_loader.batch_size
            
            # 包装 Dataset 实现标签翻转
            flipped_dataset = LabelFlippedDataset(original_dataset)
            
            # 创建新的 DataLoader 加载翻转后的数据
            # 注意：通常训练数据需要 shuffle=True
            new_loader = DataLoader(flipped_dataset, batch_size=batch_size, shuffle=True)
            
            new_dict['dataloader'] = new_loader
            # (可选) 增加标记，方便后续验证，对应 R 中的 cluster=2
            new_dict['is_attacker'] = True
        else:
            # 正常客户端，保持 dataloader 不变
            new_dict['is_attacker'] = False
            
        attacked_client_data_list.append(new_dict)
        
    return attacked_client_data_list