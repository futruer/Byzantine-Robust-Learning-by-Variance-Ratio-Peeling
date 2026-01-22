import torch
import torch.nn as nn
import torch.nn.functional as F


class NIST_CNN(nn.Module):
    def __init__(self, save_path="./CNN/cnn_model_params.pth"):
        super(NIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(800, 50)
        self.fc2 = nn.Linear(50, 10)
        self.loss = nn.CrossEntropyLoss()
        self.save_path = save_path  # 添加保存路径属性

    def forward(self, x):
        # 这是原本的训练/分类流程
        x = self.extract_features(x)  # 复用下面的逻辑
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def extract_features(self, x):
        """
        提取特征的方法。
        - 如果输入是 Tensor (batch_size, 1, 28, 28)，返回特征 Tensor (batch_size, 50)。
        - 如果输入是 DataLoader，遍历提取所有数据特征，返回 (labels, features) 元组。
        """
        # Case 1: 输入是 DataLoader (用于 Client.get_XY_tlide)
        if isinstance(x, torch.utils.data.DataLoader):
            all_features = []
            all_labels = []
            # 获取模型当前所在的设备
            device = next(self.parameters()).device
            
            # 切换到评估模式，并在不计算梯度的情况下提取特征
            was_training = self.training
            self.eval()
            
            with torch.no_grad():
                for batch in x:
                    # 兼容常见的 (inputs, labels) 格式
                    if isinstance(batch, (list, tuple)):
                        inputs, labels = batch[0], batch[1]
                    elif isinstance(batch, dict):
                        inputs = batch.get('x', batch.get('input'))
                        labels = batch.get('y', batch.get('label'))
                    else:
                        raise ValueError("DataLoader batch format not supported. Expected tuple or list.")

                    inputs = inputs.to(device)
                    
                    # 递归调用自身处理 Tensor
                    features = self.extract_features(inputs)
                    
                    all_features.append(features.cpu())
                    all_labels.append(labels.cpu())
            
            # 恢复之前的训练模式
            self.train(was_training)
            
            # 拼接所有 batch 的结果
            # labels: 1D Tensor (N,), features: 2D Tensor (N, 50)
            return torch.cat(all_labels, dim=0), torch.cat(all_features, dim=0)

        # Case 2: 输入是 Tensor (用于 forward 或 递归调用)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)  # 这里输出的就是 50维 向量
        return x

    def compute_init_theta(self) -> torch.Tensor:
        """
        对模型最后一层（fc2）的参数取中位数，得到一个长度为10的tensor张量。
        对fc2的weight的每一行（每个输出类别）取中位数。
        """
        # 获取最后一层（fc2）的权重
        last_layer_weight = self.fc2.weight.data  # shape: (10, 50)
        
        # 对每一行（每个输出类别）取中位数
        # 每一行有50个权重值，取中位数后得到10个值
        median_values = torch.median(last_layer_weight, dim=1)[0]  # shape: (10,)
        
        return median_values
