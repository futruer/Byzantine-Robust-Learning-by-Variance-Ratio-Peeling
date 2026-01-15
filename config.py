# config.py
import torch

# ==============================
# 基础训练超参数 (Training Hyperparameters)
# ==============================
EPOCHS = 10                   # model.py: Orchestrator.train 中的 epochs 参数
LEARNING_RATE = 0.01          # model.py: Orchestrator.train 中的 learning_rate 参数
BATCH_SIZE = 32               # 用于 Client 中 DataLoader 的 batch_size (虽然代码未显式展示DataLoader创建，但通常需要)

# ==============================
# 联邦学习/鲁棒性设置 (Federated Learning Settings)
# ==============================
NUM_CLIENTS = 10              # 参与训练的 Client 数量
AGGREGATION_METHOD = 'median' # 聚合方法选择: 'median' 或 'trimmed_mean'
TRIMMED_MEAN_BETA = 0.1       # aggregation.py: coordinate_wise_trimmed_mean 中的 beta 参数

# ==============================
# 模型与路径设置 (Model & Paths)
# ==============================
# CNN/cnn_model.py: NIST_CNN 类初始化及 Orchestrator.train 保存模型时使用
MODEL_SAVE_PATH = 'cnn_model_params.pth' 

# ==============================
# 运行设备 (Device)
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================
# 其他 (Misc)
# ==============================
# 随机种子，用于复现结果
SEED = 42
