import torch
import torch.nn as nn
import torch.nn.functional as F


class NIST_CNN(nn.Module):
    def __init__(self):
        super(NIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(800, 50)
        self.fc2 = nn.Linear(50, 10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # 这是原本的训练/分类流程
        x = self.extract_features(x)  # 复用下面的逻辑
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def extract_features(self, x):
        """专门用于提取50维特征向量的方法"""
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
