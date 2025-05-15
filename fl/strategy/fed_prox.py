# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/5/12 11:07
# @Describe:
import torch
from tqdm import tqdm

from fl.fl_base import BaseClient


class FedProx(BaseClient):
    """FedProx算法实现"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = kwargs['mu']   

    def proximal_term(self, w1:list, w2:list):
        """
        计算近端项（proximal term），用于FedProx算法
        :param w1: 全局模型权重列表
        :param w2: 本地模型权重列表
        :return: 近端项损失
        """
        l1 = len(w1)
        assert l1 == len(w2), "weights should be same in the shape"
        proximal_term = 0
        for i in range(l1):
            proximal_term += (torch.tensor(w1[i]) - w2[i]).norm(2) ** 2
        return proximal_term

    def local_train(self, sync_round: int, weights=None):
        """
        训练方法，根据当前通信轮次(sync_round)进行相应的训练更新
        :param weights: 服务器传递过来的模型权重
        :param sync_round: 当前的通信轮次
        """
        # 1. 加载服务器传来的全局模型权重
        if weights is not None:
            self.update_weights(weights)

        # 3. 开始本地训练
        self.model.train()
        total_loss = 0
        num_sample = len(self.train_loader.dataset)
        total_batches = len(self.train_loader) * self.epochs
        with tqdm(
            total=total_batches,
            desc=f"Client {self.client_id} Training Progress (FedProx)"
        ) as pbar:
            for epoch in range(self.epochs):  # 多轮本地训练
                epoch_loss = 0
                for data, target in self.train_loader:  # 获取每个 batch
                    self.optimizer.zero_grad()  # 清除之前的梯度
                    output = self.model(data)  # 前向传播
                    loss = self.loss(output, target)  # 计算损失
                    if weights is not None:
                        proximal_term = self.proximal_term(weights,self.get_weights(return_numpy=True))
                        loss += self.mu/2 * proximal_term
                    epoch_loss += loss.item()  # 累加损失
                    loss.backward()  # 反向传播
                    self.optimizer.step()  # 更新模型参数

                    # 更新进度条
                    pbar.update(1)
                total_loss += epoch_loss
                avg_loss = epoch_loss / len(self.train_loader)
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}"
                })
        # 4. 获取训练后的权重
        model_weights = self.get_weights(return_numpy=True)

        # 5. 返回更新后的权重给服务器，同时返回样本数和平均损失
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        # print(f'客户端:{self.client_id} 第{sync_round}轮通信:Training Loss: {avg_loss}')
        return model_weights, num_sample, avg_loss
