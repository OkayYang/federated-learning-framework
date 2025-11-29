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
import ray
import logging
from fl.data import datasets

# 导入必要的库和模块
from fl.client.fl_base import ModelConfig
from fl.server.fl_server import FLServer
from fl.model.model import CIFAR10Net, CIFAR100Net, FeMNISTNet, MNISTNet, ResNet18_CIFAR10, ResNet18_CIFAR100, ResNet18_TinyImageNet, TinyImageNetNet, SVHNNet
from fl.model.fedgen_generator import FedGenGenerator
from fl.model.fedftg_generator import create_fedftg_generator
from fl.utils import (
    optim_wrapper,
    scheduler_wrapper,
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

def train_federated_model(args):
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
            
    # Initialize Ray
    ray.init(
    ignore_reinit_error=True,
    logging_level=logging.ERROR,  # 只打印 ERROR 级别
    log_to_driver=False           # 不往驱动进程（你的终端）打日志
    )
    
    # 根据指定的数据集加载数据
    num_classes = -1
    feature_dim = -1 # 生成器生成的特征纬度，根据不同的网络模型不同
    if args.dataset.lower() == 'femnist':
        client_list, dataset_dict = datasets.load_feminist_dataset()
        model_fn = FeMNISTNet
        num_classes = 62
        feature_dim = 128
    elif args.dataset.lower() == 'mnist':
        client_list = ["client_" + str(i) for i in range(args.num_clients)]
        dataset_dict = datasets.load_mnist_dataset(client_list, partition=args.partition, beta=args.dir_beta, seed=args.seed, data_fraction=args.data_fraction)
        model_fn = MNISTNet
        num_classes = 10
        feature_dim = 128
    elif args.dataset.lower() == 'svhn':
        client_list = ["client_" + str(i) for i in range(args.num_clients)]
        dataset_dict = datasets.load_svhn_dataset(client_list, partition=args.partition, beta=args.dir_beta, seed=args.seed, data_fraction=args.data_fraction)
        model_fn = SVHNNet
        num_classes = 10
        feature_dim = 256
    elif args.dataset.lower() == 'cifar10':
        client_list = ["client_" + str(i) for i in range(args.num_clients)]
        dataset_dict = datasets.load_cifar10_dataset(client_list, partition=args.partition, beta=args.dir_beta, seed=args.seed, data_fraction=args.data_fraction)
        model_fn = CIFAR10Net
        num_classes = 10
        feature_dim = 256
    elif args.dataset.lower() == 'cifar100':
        client_list = ["client_" + str(i) for i in range(args.num_clients)]
        dataset_dict = datasets.load_cifar100_dataset(client_list, partition=args.partition, beta=args.dir_beta, seed=args.seed, data_fraction=args.data_fraction)
        model_fn = CIFAR100Net
        num_classes = 100
        feature_dim = 512
    elif args.dataset.lower() == 'tinyimagenet':
        client_list = ["client_" + str(i) for i in range(args.num_clients)]
        dataset_dict = datasets.load_tinyimagenet_dataset(client_list, partition=args.partition, beta=args.dir_beta, seed=args.seed, data_fraction=args.data_fraction)
        model_fn = TinyImageNetNet
        num_classes = 200
        feature_dim = 512
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
        plot_client_label_distribution(dataset_dict, args.dataset.lower())
    # 定义损失函数（分类任务使用交叉熵损失）
    loss_fn = nn.CrossEntropyLoss

    # 选择优化器
    if args.optimizer.lower() == 'adam':
        optim_fn = optim_wrapper(optim.Adam, lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optim_fn = optim_wrapper(optim.SGD, lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器: {args.optimizer}")
    
    # 创建调度器函数
    scheduler_fn = scheduler_wrapper(
        scheduler_type=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        comm_rounds=args.comm_rounds
    )
    
    # 打印调度器信息
    print(f"学习率调度器配置: {args.scheduler}")
    if args.scheduler == 'step':
        print(f"   - 每{args.step_size}轮衰减{args.gamma}倍")
    elif args.scheduler == 'exp':
        print(f"   - 每轮衰减{args.gamma}倍")
    elif args.scheduler == 'cosine':
        print(f"   - {args.comm_rounds}轮余弦退火")


    # 创建策略特定的超参数
    strategy_params = {}
    strategy_params['num_classes'] = num_classes
    

    if args.strategy.lower() == 'fedgen':
        strategy_params['feature_dim'] = feature_dim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = FedGenGenerator(
            feature_dim=feature_dim,
            num_classes=num_classes,
        ).to(device)
        strategy_params['generator_model'] = generator
    # 为FedFTG策略创建生成器
    elif args.strategy.lower() == 'fedftg':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 根据数据集确定图像尺寸和通道数
        if args.dataset.lower() in ['mnist', 'femnist']:
            img_channels = 1
            img_size = 28
        else:  # cifar10, cifar100, tinyimagenet
            img_channels = 3
            img_size = 32 if args.dataset.lower() in ['cifar10', 'cifar100'] else 64
            
        # 创建FedFTG生成器
        fedftg_generator = create_fedftg_generator(
            model_type='standard',  # 可选 'standard' 或 'deepinversion'
            z_dim=100,
            num_classes=num_classes,
            img_channels=img_channels,
            img_size=img_size
        )
        
        # 添加到策略参数
        strategy_params['generator'] = fedftg_generator
        strategy_params['generator_lr'] = 0.001
        strategy_params['server_epochs'] = 5  # 服务器端训练轮数
    
    # 配置模型和训练参数
    model_config = ModelConfig(
        model_fn=model_fn,  # 模型函数
        loss_fn=loss_fn,  # 损失函数
        optim_fn=optim_fn,  # 优化器函数
        scheduler_fn=scheduler_fn,  # 调度器函数
        epochs=args.local_epochs,  # 本地训练轮数
        batch_size=args.batch_size,  # 批次大小
    )

    

    # 使用给定参数初始化联邦学习服务器
    fl_server = FLServer(
        client_list=client_list,  # 客户端列表
        strategy=args.strategy.lower(),  # 联邦学习策略
        model_config=model_config,  # 模型配置
        client_dataset_dict=dataset_dict,  # 每个客户端的数据集字典
        seed=args.seed,
        **strategy_params,  # 直接解包策略特定参数
    )

    # 开始联邦学习训练过程
    history = fl_server.fit(
        comm_rounds=args.comm_rounds,  # 通信轮数（或联邦训练轮数）
        ratio_client=args.ratio_client,  # 每轮采样的客户端比例
    )

    # 绘制全局指标和客户端指标
    # 创建保存目录
    dataset_dir = f"./plots/{args.dataset.lower()}"
    history_dir = f"{dataset_dir}/history"
    os.makedirs(history_dir, exist_ok=True)

    # 保存历史记录
    experiment_name = f"{args.strategy}_{args.dataset}_seed{args.seed}"
    with open(f"{history_dir}/{experiment_name}.pkl", "wb") as f:
        pickle.dump(history, f)
        print(f"\n历史记录已保存到: {history_dir}/{experiment_name}.pkl")

    # 绘制客户端指标（各个客户端/工作节点的性能）
    plot_worker_metrics(history, experiment_name)
    # 绘制所有联邦对比图
    plot_global_metrics(history, experiment_name)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='联邦学习框架参数配置')

     # 联邦学习算法相关参数
    parser.add_argument('--strategy', type=str, default='fedspd',
                        choices=['fedavg', 'fedprox', 'moon', 'scaffold', 'feddistill', 'fedgen', 'fedspd', 'fedspd-lc', 'fedalone', 'fedftg', 'fedgkd'],
                        help='联邦学习策略')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='femnist', 
                        choices=['femnist', 'mnist', 'svhn', 'cifar10', 'cifar100', 'tinyimagenet'],
                        help='要使用的数据集 (femnist, mnist, svhn, cifar10, cifar100, tinyimagenet)')
    parser.add_argument('--partition', type=str, default='dirichlet', choices=['iid', 'noiid', 'dirichlet'],
                        help='数据分区方式 (iid 或 noiid 或 dirichlet)')
    parser.add_argument('--num_clients', type=int, default=10,
                        help='当使用MNIST/CIFAR数据集时的客户端数量')
    parser.add_argument('--dir_beta', type=float, default=0.3,
                        help='当使用dirichlet划分方式时的狄利克雷分布的参数')
    parser.add_argument('--data_fraction', type=float, default=0.1,
                        help='数据集采样比例')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='训练的批次大小')
    parser.add_argument('--local_epochs', type=int, default=20,
                        help='每个客户端的本地训练轮数')
    parser.add_argument('--comm_rounds', type=int, default=30,
                        help='联邦学习的通信轮数')
    parser.add_argument('--ratio_client', type=float, default=0.8,
                        help='每轮参与训练的客户端比例')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='优化器类型')
    
    # 学习率调度器参数
    parser.add_argument('--scheduler', type=str, default='step', 
                        choices=['step', 'exp', 'cosine'],
                        help='调度器类型: step(阶梯), exp(指数), cosine(余弦)')
    parser.add_argument('--step_size', type=int, default=5,
                        help='StepLR每多少轮衰减一次')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='学习率衰减倍数')
    parser.add_argument('--patience', type=int, default=3,
                        help='ReduceLROnPlateau的耐心值')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--plot_distribution', type=bool, default=True,
                        help='是否绘制客户端标签分布')
    
    return parser.parse_args()


# 运行联邦学习设置和训练
if __name__ == "__main__":
    args = parse_arguments()
    setup_seed(args.seed)
    train_federated_model(args)

