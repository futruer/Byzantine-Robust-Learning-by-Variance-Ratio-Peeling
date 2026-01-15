import argparse
import sys
import torch
from functools import partial

# 导入你的模块
import config
import aggregation
from CNN.cnn_model import NIST_CNN
from model import Client, Orchestrator

def parse_args():
    """
    解析命令行参数，如果没有提供参数，则使用 config.py 中的默认值
    """
    parser = argparse.ArgumentParser(description="Byzantine Robust Learning via Variance Ratio Peeling")

    # --- 训练参数 ---
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, 
                        help=f'训练轮数 (Default: {config.EPOCHS})')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, 
                        help=f'学习率 (Default: {config.LEARNING_RATE})')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, 
                        help=f'Batch size (Default: {config.BATCH_SIZE})')

    # --- 联邦学习参数 ---
    parser.add_argument('--num_clients', type=int, default=config.NUM_CLIENTS, 
                        help=f'客户端数量 (Default: {config.NUM_CLIENTS})')
    parser.add_argument('--aggregation', type=str, default=config.AGGREGATION_METHOD, 
                        choices=['median', 'trimmed_mean'],
                        help=f'聚合方法 (Default: {config.AGGREGATION_METHOD})')
    parser.add_argument('--beta', type=float, default=config.TRIMMED_MEAN_BETA, 
                        help=f'Trimmed Mean 的 beta 值 (Default: {config.TRIMMED_MEAN_BETA})')

    # --- 路径参数 ---
    parser.add_argument('--save_path', type=str, default=config.MODEL_SAVE_PATH, 
                        help=f'模型保存路径 (Default: {config.MODEL_SAVE_PATH})')
    
    # --- 设备参数 ---
    parser.add_argument('--device', type=str, default=config.DEVICE, 
                        help=f'运行设备 (Default: {config.DEVICE})')

    return parser.parse_args()

def main():
    # 1. 获取参数
    args = parse_args()
    print(f"Running with args: {args}")

    # 设置设备
    device = torch.device(args.device)

    # 2. 初始化模型
    # 注意：根据你的 cnn_model.py，NIST_CNN 接受 save_path 参数
    global_model = NIST_CNN(save_path=args.save_path).to(device)

    # 3. 准备数据和客户端 (这里需要你补充实际的数据加载逻辑)
    # 示例伪代码：
    # dataset = load_dataset(...) 
    # client_dataloaders = split_data(dataset, args.num_clients, args.batch_size)
    clients = []
    for i in range(args.num_clients):
        # 注意：这里需要传入 data (DataLoader) 和 reform_func
        # client_data = client_dataloaders[i]
        # client = Client(model=global_model, data=client_data, reform_func=YOUR_REFORM_FUNC, id=i)
        # clients.append(client)
        pass 
    
    if not clients:
        print("Warning: No clients initialized. Please implement data loading logic in main.py.")
        # 为了演示代码运行，这里暂时跳过后续 Orchestrator 的实例化
        # return 

    # 4. 选择聚合函数
    if args.aggregation == 'median':
        agg_func = aggregation.coordinate_wise_median
    elif args.aggregation == 'trimmed_mean':
        # 使用 partial 固定 beta 参数
        agg_func = partial(aggregation.coordinate_wise_trimmed_mean, beta=args.beta)
    else:
        raise ValueError(f"Unsupported aggregation method: {args.aggregation}")

    # 5. 初始化 Orchestrator
    orchestrator = Orchestrator(model=global_model, clients=clients)

    # 6. 开始训练
    # 传入从 argparse 解析得到的参数
    orchestrator.train(
        epochs=args.epochs,
        learning_rate=args.lr,
        aggregation=agg_func,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()
