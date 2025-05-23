# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:31
# @Describe:
from abc import ABC, abstractmethod
import copy
from typing import Callable
from torch.utils.data import DataLoader
import numpy as np
import torch
import random
from torch import device, nn, optim


class BaseClient(ABC):

    def __init__(
        self,
        client_id,
        model,
        loss,
        optimizer,
        epochs,
        batch_size,
        train_loader,
        test_loader,
        **kwargs,
    ):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.kwargs = kwargs
        
        # 设置客户端特定的随机种子，确保环境隔离
        self.seed = kwargs.get('seed', 42)
        self._set_seed(self.seed)
        
    def _set_seed(self, seed):
        """设置随机种子以确保可复现性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
    @abstractmethod
    def local_train(self, sync_round: int, weights=None):
        """
        训练方法，根据当前通信轮次(sync_round)进行相应的训练更新
        :param weights: 服务器传递过来的模型权重
        :param sync_round: 当前的通信轮次
        """
        pass

    def evaluate(self, test_dataset=None):
        """
        评估模型性能
        :param test_dataset: 可选的测试数据集，如果为None则使用客户端自己的测试数据
        :return: (accuracy, test_loss)
        """
        # 确保模型处于评估模式
        self.model.eval()
        correct = 0
        test_loss = 0
        total = 0
        
        # 根据输入选择测试数据集
        if test_dataset is None:
            test_loader = self.test_loader
        else:
            # 创建统一的数据加载器，确保评估条件一致
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,  # 评估时不打乱数据顺序，确保一致性
                drop_last=False  # 不丢弃最后一个批次
            )

        with torch.no_grad():  # 不计算梯度
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 累加批次损失
                batch_loss = self.loss(output, target).item()
                test_loss += batch_loss * len(data)  # 按样本数加权
                
                # 计算预测准确度
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # 计算平均损失和准确率
        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / total if total > 0 else float('inf')
        
        return accuracy, avg_loss

    def get_weights(self, return_numpy=False):
        if not return_numpy:
            return {k: v.cpu() for k, v in self.model.state_dict().items()}
        else:
            weights_list = []
            for v in self.model.state_dict().values():
                weights_list.append(v.cpu().numpy())
            return [e.copy() for e in weights_list]

    def update_weights(self, weights):
        if len(weights) != len(self.model.state_dict()):
            raise ValueError("传入的权重数组数量与模型参数数量不匹配。")
        keys = self.model.state_dict().keys()
        weights_dict = {}
        for k, v in zip(keys, weights):
            weights_dict[k] = torch.Tensor(np.copy(v)).to(self.device)
        self.model.load_state_dict(weights_dict)

    def get_gradients(self, parameters=None):
        if parameters is None:
            parameters = self.model.parameters()
        grads = []
        for p in parameters:
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return [g.copy() for g in grads]
    
    #获取模型副本
    def get_model_copy(self):
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()
        return model_copy


class ModelConfig:
    def __init__(
        self,
        model_fn: Callable[..., nn.Module],  # 用于生成模型的函数
        loss_fn: Callable[..., nn.Module],  # 用于生成损失函数的函数
        optim_fn: Callable[..., optim.Optimizer],  # 用于生成优化器的函数
        epochs: int,
        batch_size: int,
        **kwargs,
    ):
        """
        初始化模型配置
        :param model: 训练的模型
        :param loss_fn: 损失函数
        :param optimizer_fn: 优化器构造函数（如 optim.SGD 或 optim.Adam）
        :param epochs: 训练的轮数
        :param batch_size: 批次大小
        """
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.kwargs = kwargs
    def get_optimizer(self, parameters):
        """获取优化器"""
        return self.optim_fn(parameters)

    def get_loss_fn(self):
        """获取损失函数"""
        return self.loss_fn()

    def get_model(self):
        model = self.model_fn()
        device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device_type)
        return model

    def get_epochs(self):
        """获取训练轮数"""
        return self.epochs

    def get_batch_size(self):
        """获取批次大小"""
        return self.batch_size
