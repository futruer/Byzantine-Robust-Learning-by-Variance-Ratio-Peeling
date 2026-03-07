import torch

# ==============================
# 基础训练超参数 (Training Hyperparameters)
# ==============================
EPOCHS = 10                   # 训练轮数
LEARNING_RATE = 0.1          # 学习率
BATCH_SIZE = 64               # 批次大小 (根据MNIST常用设置调整)

# ==============================
# 联邦学习设置 (Federated Learning Settings)
# ==============================
NUM_CLIENTS = 20              # 客户端总数 (K=20)
AGGREGATION_METHOD = 'median' # 聚合方法: 'median' 或 'trimmed_mean'
TRIMMED_MEAN_BETA = 0.1       # trimmed_mean 的 beta 参数

# ==============================
# 攻击设置 (Attack Settings)
# ==============================
ATTACK_RATIO = 0.2            # 攻击者比例 (20% -> 4个客户端)

# ==============================
# 模型与路径设置 (Model & Paths)
# ==============================
# 模型保存路径
MODEL_SAVE_PATH = './cnn_model_params.pth' 
# 数据集下载路径
DATA_ROOT = './data'

# ==============================
# 运行设备 (Device)
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================
# 初始估计训练参数 (Init Theta Training)
# ==============================
INIT_EPOCHS = 3              # 本地训练轮数（可调参数）
INIT_LR = 0.2                 # 本地学习率（增大以增加各客户端模型差异）

# ==============================
# 其他 (Misc)
# ==============================
SEED = 42                     # 随机种子