# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 15:15
# @Describe:


import torch.nn as nn
import torch.optim as optim

from fl.data import datasets

# 导入必要的库和模块
from fl.fl_base import ModelConfig
from fl.fl_server import FLServer
from fl.model.model import FeMNISTNet, MNISTNet
from fl.utils import (
    optim_wrapper,
    plot_client_label_distribution,
    plot_global_metrics,
    plot_worker_metrics,
)


def setup_and_train_federated_model():
    """
    设置联邦学习系统，训练模型并绘制结果。

    该函数初始化必要的配置，包括：
    - 加载数据集（FEMNIST）
    - 定义模型、损失函数和优化器
    - 设置联邦学习服务器
    - 使用联邦平均（FedAvg）策略训练模型
    - 绘制全局和客户端级别的性能指标

    训练使用 `FeMNISTNet` 模型在 FEMNIST 数据集上进行。
    """
    # 加载 FEMNIST 数据集
    client_list, dataset_dict = datasets.load_feminist_dataset()

    # 加载 MNIST 数据集用于客户端
    # client_list = ["alice","bob","rick"]
    # dataset_dict = datasets.load_mnist_dataset(client_list)

    # 定义损失函数（分类任务使用交叉熵损失）
    loss_fn = nn.CrossEntropyLoss

    # 设置优化器（Adam，学习率为0.01）
    optim_fn = optim_wrapper(optim.Adam, lr=1e-2)

    # 配置模型和训练参数
    model_config = ModelConfig(
        model_fn=FeMNISTNet,  # 模型函数（在model文件中定义的FeMNISTNet）
        loss_fn=loss_fn,  # 损失函数
        optim_fn=optim_fn,  # 优化器函数
        epochs=10,  # 训练轮数
        batch_size=64,  # 批次大小
    )

    # 使用给定参数初始化联邦学习服务器
    fl_server = FLServer(
        client_list=client_list,  # 客户端列表（用户）
        strategy="fedavg",  # 联邦学习策略（FedAvg）
        model_config=model_config,  # 模型配置
        client_dataset_dict=dataset_dict,  # 每个客户端的数据集字典
    )

    # 开始联邦学习训练过程（20轮通信）
    history = fl_server.fit(
        comm_rounds=20,  # 通信轮数（或联邦训练轮数）
        ratio_client=1,  # 每轮采样的客户端比例（1表示所有客户端）
    )

    # 绘制所有联邦训练轮次的全局指标（如准确率、损失）
    plot_global_metrics(history)

    # 绘制客户端指标（各个客户端/工作节点的性能）
    plot_worker_metrics(history)


# 运行联邦学习设置和训练
if __name__ == "__main__":
    setup_and_train_federated_model()
