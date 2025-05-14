# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 15:15
# @Describe:


import torch.nn as nn
import torch.optim as optim
import argparse
import random
import numpy as np
import torch
import os
import pickle
from fl.data import datasets

# 导入必要的库和模块
from fl.fl_base import ModelConfig
from fl.fl_server import FLServer
from fl.model.model import FeMNISTNet, Generator, MNISTNet
from fl.utils import (
    optim_wrapper,
    plot_client_label_distribution,
    plot_global_metrics,
    plot_worker_metrics,
)
def setup_seed(seed):
    """设置随机种子，确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

def setup_and_train_federated_model(args):
    """
    设置联邦学习系统，训练模型并绘制结果。

    该函数初始化必要的配置，包括：
    - 加载数据集（FEMNIST 或 MNIST）
    - 定义模型、损失函数和优化器
    - 设置联邦学习服务器
    - 使用指定的联邦策略训练模型
    - 绘制全局和客户端级别的性能指标

    Args:
        args: 命令行参数，包含数据集、学习率、批次大小等配置
    """
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # 根据指定的数据集加载数据
    if args.dataset.lower() == 'femnist':
        client_list, dataset_dict = datasets.load_feminist_dataset()
        model_fn = FeMNISTNet
    elif args.dataset.lower() == 'mnist':
        client_list = ["client_" + str(i) for i in range(args.num_clients)]
        dataset_dict = datasets.load_mnist_dataset(client_list, partition=args.partition, beta=args.dir_beta, seed=args.seed)
        model_fn = MNISTNet
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
     # 打印数据分布信息
    print(f"\n数据集划分方式: {args.partition}")
    if args.partition == "dirichlet":
        print(f"狄利克雷分布参数 dir_beta: {args.dir_beta} (较小的值表示更高的异质性)")
    
    print("\n客户端数据统计:")
    for client in client_list:
        train_labels = [dataset_dict[client]["train_dataset"].Y[i].item() for i in range(len(dataset_dict[client]["train_dataset"]))]
        test_labels = [dataset_dict[client]["test_dataset"].Y[i].item() for i in range(len(dataset_dict[client]["test_dataset"]))]
        print(f"客户端 {client}: 训练样本总数: {len(train_labels)}, 测试样本总数: {len(test_labels)}")
        #训练样本标签分布
        unique, counts = np.unique(train_labels, return_counts=True)
        print(f"  训练样本标签分布: {dict(zip(unique, counts))}")
        #测试样本标签分布
        unique, counts = np.unique(test_labels, return_counts=True)
        print(f"  测试样本标签分布: {dict(zip(unique, counts))}")

    # 绘制客户端标签分布
    if args.plot_distribution:
        plot_client_label_distribution(dataset_dict)
    # 定义损失函数（分类任务使用交叉熵损失）
    loss_fn = nn.CrossEntropyLoss

    # 选择优化器
    if args.optimizer.lower() == 'adam':
        optim_fn = optim_wrapper(optim.Adam, lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optim_fn = optim_wrapper(optim.SGD, lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器: {args.optimizer}")


    # 创建FedProx和MOON算法的超参数
    strategy_params = {}
    if args.strategy.lower() == 'fedprox':
        if args.mu is not None:
            strategy_params['mu'] = args.mu
        else:
            raise ValueError("mu参数不存在")
    elif args.strategy.lower() == 'moon':
        if args.mu is not None:
            strategy_params['mu'] = args.mu
        else:
            raise ValueError("mu参数不存在")
        if args.temperature is not None:
            strategy_params['temperature'] = args.temperature
        else:
            raise ValueError("temperature参数不存在")
    elif args.strategy.lower() == 'fedgen':
        if args.latent_dim is not None:
            strategy_params['latent_dim'] = args.latent_dim
        else:
            raise ValueError("latent_dim参数不存在")
        if args.num_classes is not None:
            strategy_params['num_classes'] = args.num_classes
        else:
            raise ValueError("num_classes参数不存在")
        if args.feature_dim is not None:
            strategy_params['feature_dim'] = args.feature_dim
        else:
            raise ValueError("feature_dim参数不存在")
        strategy_params['generator_model'] = Generator(
            latent_dim=args.latent_dim,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes
        )
    
    # 配置模型和训练参数
    model_config = ModelConfig(
        model_fn=model_fn,  # 模型函数
        loss_fn=loss_fn,  # 损失函数
        optim_fn=optim_fn,  # 优化器函数
        epochs=args.local_epochs,  # 本地训练轮数
        batch_size=args.batch_size,  # 批次大小
    )

    

    # 使用给定参数初始化联邦学习服务器
    fl_server = FLServer(
        client_list=client_list,  # 客户端列表
        strategy=args.strategy.lower(),  # 联邦学习策略
        model_config=model_config,  # 模型配置
        client_dataset_dict=dataset_dict,  # 每个客户端的数据集字典
        **strategy_params,  # 直接解包策略特定参数
    )

    # 开始联邦学习训练过程
    history = fl_server.fit(
        comm_rounds=args.comm_rounds,  # 通信轮数（或联邦训练轮数）
        ratio_client=args.ratio_client,  # 每轮采样的客户端比例
    )

    # 绘制全局指标和客户端指标
    # 创建保存目录
    os.makedirs("./plots/history/", exist_ok=True)
    
    # 保存历史记录
    experiment_name = f"{args.strategy}_{args.dataset}_seed{args.seed}"
    with open(f"./plots/history/{experiment_name}.pkl", "wb") as f:
        pickle.dump(history, f)
        print(f"\n历史记录已保存到: ./plots/history/{experiment_name}.pkl")

    # 绘制客户端指标（各个客户端/工作节点的性能）
    plot_worker_metrics(history, experiment_name)
    # 绘制所有联邦对比图
    plot_global_metrics(history, experiment_name)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='联邦学习框架参数配置')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='femnist', choices=['femnist', 'mnist'],
                        help='要使用的数据集 (femnist 或 mnist)')
    parser.add_argument('--partition', type=str, default='noiid', choices=['iid', 'noiid', 'dirichlet'],
                        help='数据分区方式 (iid 或 noiid 或 dirichlet)')
    parser.add_argument('--num_clients', type=int, default=10,
                        help='当使用MNIST数据集时的客户端数量')
    parser.add_argument('--dir_beta', type=float, default=0.4,
                        help='当使用dirichlet划分方式时的狄利克雷分布的参数')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='训练的批次大小')
    parser.add_argument('--local_epochs', type=int, default=20,
                        help='每个客户端的本地训练轮数')
    parser.add_argument('--comm_rounds', type=int, default=50,
                        help='联邦学习的通信轮数')
    parser.add_argument('--ratio_client', type=float, default=1.0,
                        help='每轮参与训练的客户端比例')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='优化器类型')
    
    # 联邦学习算法相关参数
    parser.add_argument('--strategy', type=str, default='fedgen',
                        choices=['fedavg', 'fedprox', 'moon', 'scaffold', 'feddistill', 'fedgen'],
                        help='联邦学习策略')
    parser.add_argument('--mu', type=float, default=0.01,
                        help='FedProx和MOON算法的mu参数')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='MOON和FedDistill算法的temperature参数')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='FedDistill算法的知识蒸馏权重参数')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='FedGen算法的知识蒸馏权重参数')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='FedGen算法的生成样本损失权重参数')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='FedGen算法的潜在空间维度')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='FedGen算法的特征维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='FedGen算法的隐藏层维度')
    parser.add_argument('--num_classes', type=int, default=62,
                        help='FedGen算法的类别数量')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--plot_distribution', type=bool, default=False,
                        help='是否绘制客户端标签分布')
    
    return parser.parse_args()


# 运行联邦学习设置和训练
if __name__ == "__main__":
    args = parse_arguments()
    setup_seed(args.seed)
    setup_and_train_federated_model(args)

