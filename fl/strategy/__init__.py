# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:30
# @Describe:
import torch
from torch import device
from torch.utils.data import DataLoader

from fl.fl_base import ModelConfig
from fl.strategy.fed_avg import FedAvg
from fl.strategy.fed_prox import FedProx
from fl.strategy.moon import Moon


def create_client(
        strategy: str,
        client_id: str,
        model_config: ModelConfig,
        client_dataset_dict,  # 暂定未实现
        **kwargs

):
    """构建模型并返回，自动设置损失函数和优化器"""
    if model_config.model_fn is None:
        raise ValueError("Model function is required.")
    if model_config.loss_fn is None:
        raise ValueError("Loss function is required.")
    if model_config.optim_fn is None:
        raise ValueError("Optimizer function is required.")

    # 创建模型
    model = model_config.get_model()
    # 设置损失函数
    loss = model_config.get_loss_fn()
    # 设置优化器
    optimizer = model_config.get_optimizer(model.parameters())
    epochs = model_config.get_epochs()
    batch_size = model_config.get_batch_size()

    client_dataset = client_dataset_dict[client_id]
    train_dataLoader = DataLoader(client_dataset['train_dataset'], batch_size=batch_size, shuffle=True)
    test_dataLoader = DataLoader(client_dataset['test_dataset'], batch_size=batch_size, shuffle=True)
    if strategy == "fedavg":
        return FedAvg(
            client_id,
            model,
            loss,
            optimizer,
            epochs,
            batch_size,
            train_dataLoader,
            test_dataLoader,
            **kwargs
        )
    elif strategy == "fedprox":
        return FedProx(
            client_id,
            model,
            loss,
            optimizer,
            epochs,
            batch_size,
            train_dataLoader,
            test_dataLoader,
            **kwargs
        )
    elif strategy == "moon":
        return Moon(
            client_id,
            model,
            loss,
            optimizer,
            epochs,
            batch_size,
            train_dataLoader,
            test_dataLoader,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")