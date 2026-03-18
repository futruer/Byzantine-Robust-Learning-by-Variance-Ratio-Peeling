#python run_experiments.py --dataset mnist

"""
实验脚本：运行所有对比实验并保存结果
支持 MNIST 和 CIFAR-10 数据集双轨运行
"""
import argparse
import torch
import random
import copy
import numpy as np
import pandas as pd
import os
import time
from functools import partial
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# 导入自定义模块
import config
import aggregation
from attack import cnn_rev_attack, gaussian_attack, sign_flipping_attack, scaling_attack, omniscient_ipm_attack
from cnn_model import NIST_CNN, CIFAR10_CNN
from cnn_data_process import (
    get_mnist_client_dataloaders, get_proxy_dataloader, 
    get_cifar10_client_dataloaders, get_cifar10_proxy_dataloader, 
    cnn_reform_func_with_gh
)
from model import Client, Orchestrator

def print_header(dataset_name):
    """打印实验头部信息"""
    print("=" * 70)
    print("    Byzantine-Robust FL Experiments (新想法: 代理数据筛选)")
    print("=" * 70)
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"  Clients: {config.NUM_CLIENTS}")
    print(f"  Epochs per round: {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Device: {config.DEVICE}")
    print("=" * 70)

def print_experiment_info(agg_name, attack_type, attack_ratio, exp_num, total):
    """打印当前实验的详细信息"""
    agg_descriptions = {
        'peeling': 'VR Peeling (新想法：代理数据+VR Peeling)',
        'fedavg': 'FedAvg (无防御基准)',
        'trimmed_mean': f'Trimmed Mean (β={config.TRIMMED_MEAN_BETA})',
        'multi_krum': 'Multi-Krum (n_remove=2, n_select=5)',
        'clustering': 'Clustering-based (基于聚类)',
        'median': 'Coordinate-wise Median (坐标级中位数)'
    }

    print("\n" + "=" * 70)
    print(f"  Experiment [{exp_num}/{total}]")
    print("-" * 70)
    print(f"  [聚合方法] {agg_name:15s} - {agg_descriptions.get(agg_name, '')}")
    print(f"  [攻击类型] {attack_type:15s}")
    print(f"  [攻击比例] {attack_ratio*100:.0f}%  ({int(config.NUM_CLIENTS * attack_ratio)}/{config.NUM_CLIENTS} clients)")
    print("=" * 70)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_attack_func(attack_type):
    attacks = {
        'label_flipping': None,
        'gaussian': gaussian_attack,
        'sign_flipping': sign_flipping_attack,
        'scaling': scaling_attack,
        'ipm': omniscient_ipm_attack
    }
    return attacks.get(attack_type, None)

def get_aggregation_func(agg_name):
    max_attackers = int(config.NUM_CLIENTS * 0.4)
    aggs = {
        'peeling': aggregation.fedavg,
        'fedavg': aggregation.fedavg,
        'trimmed_mean': partial(aggregation.coordinate_wise_trimmed_mean, beta=config.TRIMMED_MEAN_BETA),
        'multi_krum': partial(aggregation.multi_krum, n_remove=max_attackers, n_select=5),
        'clustering': aggregation.clustering_based,
        'median': aggregation.coordinate_wise_median,
    }
    return aggs.get(agg_name, None)

def evaluate_model(model, device, data_root='./data', dataset_name='mnist'):
    """在测试集上评估模型准确率（动态支持不同的数据集）"""
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy

def run_benchmark_training(model, clients, epochs, learning_rate, aggregation_func, gradient_attack_func, device, verbose=True):
    if verbose:
        print(f"       开始 {epochs} 轮 Benchmark 训练 (无筛选)...")

    for epoch in range(epochs):
        for client in clients:
            client.model = copy.deepcopy(model)

        gradients_list = []
        for client in clients:
            gradient_result = client.train(local_epochs=1, learning_rate=learning_rate)
            gradient_result['is_attacker'] = client.is_attacker
            gradients_list.append(gradient_result)

        local_train_loss = sum([g['loss'] for g in gradients_list]) / len(gradients_list)

        if gradient_attack_func is not None and gradient_attack_func.__name__ == 'omniscient_ipm_attack':
            gradients_list = gradient_attack_func(gradients_list)

        aggregated_gradients = aggregation_func(gradients_list)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in aggregated_gradients:
                    param.data -= 1.0 * aggregated_gradients[name]

        model.eval()
        total_loss_after = 0.0
        total_samples = 0
        with torch.no_grad():
            for client in clients:
                if not client.is_attacker:
                    for batch_data in client.data:
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                            inputs, labels = batch_data
                        else:
                            continue
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = F.cross_entropy(outputs, labels)
                        total_loss_after += loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)

        avg_loss_after = total_loss_after / total_samples if total_samples > 0 else 0
        model.train()

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Local Train Loss: {local_train_loss:.6f} | Global Loss: {avg_loss_after:.6f}")

def run_experiment(agg_name, attack_type, attack_ratio, epochs, dataset_name='mnist', device='cpu', seed=42, verbose=True):
    set_seed(seed)
    start_time = time.time()

    if verbose: print(f"\n  [1/6] 准备 {dataset_name.upper()} 数据...")

    # [核心修改]：根据数据集动态加载 DataLoaders 和全局模型
    if dataset_name == 'mnist':
        raw_client_data = get_mnist_client_dataloaders(config.NUM_CLIENTS, config.BATCH_SIZE, config.DATA_ROOT)
        proxy_data_loader = get_proxy_dataloader(config.BATCH_SIZE, config.DATA_ROOT)
        global_model = NIST_CNN().to(device)
        current_proxy_epochs = config.PROXY_EPOCHS
    elif dataset_name == 'cifar10':
        raw_client_data = get_cifar10_client_dataloaders(config.NUM_CLIENTS, config.BATCH_SIZE, config.DATA_ROOT)
        proxy_data_loader = get_cifar10_proxy_dataloader(config.BATCH_SIZE, config.DATA_ROOT)
        global_model = CIFAR10_CNN().to(device)
        # CIFAR10 较难，适当增加代理模型的训练轮数以提取更准确的初始特征
        current_proxy_epochs = config.PROXY_EPOCHS + 5 
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    gradient_attack_func = None

    if verbose: print(f"  [2/6] 设置攻击: {attack_type} (比例: {attack_ratio*100:.0f}%)")

    if attack_type == 'label_flipping':
        attacked_client_data = cnn_rev_attack(raw_client_data, alpha=attack_ratio)
    else:
        random.seed(42)
        K = len(raw_client_data)
        num_attackers = int(K * attack_ratio)
        attacker_indices = set(random.sample(range(K), num_attackers))

        attacked_client_data = []
        for i, client_dict in enumerate(raw_client_data):
            new_dict = client_dict.copy()
            new_dict['is_attacker'] = (i in attacker_indices)
            attacked_client_data.append(new_dict)

        gradient_attack_func = get_attack_func(attack_type)

    if verbose: print(f"  [3/6] 初始化模型和客户端...")

    clients = []
    for client_info in attacked_client_data:
        client = Client(
            model=global_model,
            data=client_info['dataloader'],
            reform_func_with_gh=cnn_reform_func_with_gh,
            id=client_info['id'],
            is_attacker=client_info.get('is_attacker', False),
            attack_type=attack_type
        )
        clients.append(client)

    agg_func = get_aggregation_func(agg_name)
    use_screening = (agg_name == 'peeling')

    if use_screening:
        if verbose: print("  [4/6] 训练模式: 启用 VR Peeling 筛选")
        ipm = (attack_type == 'ipm')
        # [核心修改] 传入针对当前数据集专属的 proxy_model_path
        orchestrator = Orchestrator(
            model=global_model,
            clients=clients,
            proxy_data_loader=proxy_data_loader,
            ipm=ipm,
            proxy_model_path=f'./proxy_model_{dataset_name}.pth',
            proxy_epochs=current_proxy_epochs
        )
        orchestrator.train(
            epochs=epochs,
            learning_rate=config.LEARNING_RATE,
            aggregation=agg_func,
            save_path=None
        )
    else:
        if verbose: print("  [4/6] 训练模式: Benchmark 无筛选")
        run_benchmark_training(
            model=global_model,
            clients=clients,
            epochs=epochs,
            learning_rate=config.LEARNING_RATE,
            aggregation_func=agg_func,
            gradient_attack_func=gradient_attack_func,
            device=device,
            verbose=verbose
        )

    if verbose: print(f"  [5/6] 评估模型...")
    accuracy = evaluate_model(global_model, device, config.DATA_ROOT, dataset_name)
    elapsed_time = time.time() - start_time

    if verbose: print(f"  [6/6] 训练完成! 测试准确率: {accuracy:.2f}% | 耗时: {elapsed_time:.2f}s")
    return accuracy, elapsed_time

def run_experiments(dataset_name='mnist', output_file=None):
    device = config.DEVICE
    print_header(dataset_name)
    
    if output_file is None:
        output_file = f'results_new_frame_{dataset_name}.csv'

    agg_methods = ['peeling', 'fedavg', 'clustering', 'trimmed_mean', 'multi_krum']
    attack_types = ['ipm', 'scaling', 'label_flipping', 'sign_flipping', 'gaussian']
    attack_ratios = [0.05, 0.15, 0.25, 0.35, 0.45]
    epochs = config.EPOCHS

    results = []
    valid_experiments = []
    for agg in agg_methods:
        for attack in attack_types:
            for ratio in attack_ratios:
                valid_experiments.append((agg, attack, ratio))

    total_experiments = len(valid_experiments)
    current = 0

    print(f"\n>>> 开始运行 {dataset_name.upper()} 上的 {total_experiments} 个实验...")
    total_start_time = time.time()

    for agg, attack, ratio in valid_experiments:
        current += 1
        print_experiment_info(agg, attack, ratio, current, total_experiments)

        try:
            accuracy, elapsed_time = run_experiment(
                agg_name=agg,
                attack_type=attack,
                attack_ratio=ratio,
                epochs=epochs,
                dataset_name=dataset_name,
                device=device,
                seed=config.SEED,
                verbose=True
            )
            results.append({
                'dataset': dataset_name,
                'aggregation': agg,
                'attack_type': attack,
                'attack_ratio': ratio,
                'accuracy': accuracy,
                'time': elapsed_time
            })
        except Exception as e:
            print(f"\n  [X] 实验失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'dataset': dataset_name,
                'aggregation': agg,
                'attack_type': attack,
                'attack_ratio': ratio,
                'accuracy': None,
                'error': str(e)
            })

    total_time = time.time() - total_start_time
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("                     实验完成 - 汇总信息")
    print("=" * 70)
    print(f"  总耗时: {total_time/60:.2f} 分钟")
    print(f"  结果保存: {output_file}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Byzantine-Robust FL Experiments")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='选择运行的数据集 (mnist 或 cifar10)')
    parser.add_argument('--output', type=str, default=None, help='输出 CSV 文件路径')
    args = parser.parse_args()

    run_experiments(dataset_name=args.dataset, output_file=args.output)

if __name__ == "__main__":
    main()