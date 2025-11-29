# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:30
# @Describe:
import torch
from tqdm import tqdm

from fl.client.fl_base import BaseClient

class FedAlone(BaseClient):
    """FedAlone算法实现"""
    
    def local_train(self, sync_round: int, weights=None):
        """
        训练方法，根据当前通信轮次(sync_round)进行相应的训练更新
        :param weights: 服务器传递过来的模型权重
        :param sync_round: 当前的通信轮次
        """
        # 1. 加载服务器传来的全局模型权重
        # if weights is not None:
        #     self.update_weights(weights)

        # 3. 开始本地训练
        self.model.train()
        total_loss = 0
        num_sample = len(self.train_loader.dataset)
        total_batches = len(self.train_loader) * self.epochs
        from fl.utils import ClientProgress
        
        with ClientProgress(
            progress_actor=self.progress_actor,
            client_id=self.client_id,
            total=total_batches,
            desc=f"Client {self.client_id} Training (FedAlone)"
        ) as pbar:
            for epoch in range(self.epochs):  # 多轮本地训练
                epoch_loss = 0
                for data, target in self.train_loader:  # 获取每个 batch
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()  # 清除之前的梯度
                    output = self.model(data)  # 前向传播
                    loss = self.loss(output, target)  # 计算损失
                    epoch_loss += loss.item()  # 累加损失
                    loss.backward()  # 反向传播
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()  # 更新模型参数

                    # 更新进度条
                    pbar.update(1)
                total_loss += epoch_loss
                avg_loss = epoch_loss / len(self.train_loader)
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
        self.scheduler.step()
        # 4. 获取训练后的权重
        model_weights = self.get_weights(return_numpy=True)

        # 5. 返回更新后的权重给服务器，同时返回样本数和平均损失
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        return model_weights, num_sample, avg_loss
