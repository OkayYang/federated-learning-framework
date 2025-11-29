# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:30
# @Describe:
import torch
import copy
from tqdm import tqdm
import numpy as np

from fl.client.fl_base import BaseClient

class FedProx(BaseClient):
    """FedProx算法实现"""
    # 自定义策略名
    _strategy_name_ = "fedprox"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # mu是关键超参数，控制正则化强度
        self.mu = kwargs.get('mu', 0.01)

    def proximal_term(self, local_weights, global_weights):
        proximal_term = 0
        total_params = 0
        for i in range(len(local_weights)):
            diff = (torch.tensor(local_weights[i]) - global_weights[i]).norm(2).pow(2)
            proximal_term += diff
            total_params += torch.tensor(local_weights[i]).numel()
    
        # 按参数数量归一化
        return proximal_term / (2.0 * total_params) 
        
    def local_train(self, sync_round: int, weights=None):
        """
        训练方法，根据当前通信轮次(sync_round)进行相应的训练更新
        :param weights: 服务器传递过来的模型权重
        :param sync_round: 当前的通信轮次
        """
        # 1. 记录训练前状态，保存全局模型副本
        if weights is not None:
            # 加载服务器传来的全局模型权重
            self.update_weights(weights)
                
            
        # 2. 开始本地训练
        self.model.train()
        total_loss = 0
        total_ce_loss = 0  # 交叉熵损失 
        total_prox_loss = 0  # 近端项损失
        num_sample = len(self.train_loader.dataset)
        total_batches = len(self.train_loader) * self.epochs
        
        from fl.utils import ClientProgress
        
        with ClientProgress(
            progress_actor=self.progress_actor,
            client_id=self.client_id,
            total=total_batches,
            desc=f"Client {self.client_id} Training (FedProx, μ={self.mu})"
        ) as pbar:
            for epoch in range(self.epochs):  # 多轮本地训练
                epoch_loss = 0
                epoch_ce_loss = 0
                epoch_prox_loss = 0
                
                for data, target in self.train_loader:  # 获取每个 batch
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()  # 清除之前的梯度
                    output = self.model(data)  # 前向传播
                    
                    # 计算任务损失
                    ce_loss = self.loss(output, target)
                    
                    # 计算近端正则化项
                    prox_loss = 0
                    if weights is not None:
                        prox_loss = self.mu * self.proximal_term(self.get_weights(return_numpy=True), weights)
                    
                    # 总损失 = 任务损失 + 近端正则化项
                    loss = ce_loss + prox_loss
                    
                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()  # 更新模型参数
                    
                    # 记录各部分损失
                    epoch_loss += loss.item()
                    epoch_ce_loss += ce_loss.item()
                    epoch_prox_loss += prox_loss.item() if isinstance(prox_loss, torch.Tensor) else prox_loss
                    
                    # 更新进度条
                    pbar.update(1)
                    
                # 累加每个epoch的损失
                total_loss += epoch_loss
                total_ce_loss += epoch_ce_loss
                total_prox_loss += epoch_prox_loss
                
                # 更新进度条显示
                avg_loss = epoch_loss / len(self.train_loader)
                avg_ce_loss = epoch_ce_loss / len(self.train_loader)
                avg_prox_loss = epoch_prox_loss / len(self.train_loader)
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'ce_loss': f"{avg_ce_loss:.4f}",
                    'prox_loss': f"{avg_prox_loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
        self.scheduler.step()
                
        # 3. 获取训练后的权重
        model_weights = self.get_weights(return_numpy=True)

        # 4. 返回更新后的权重给服务器，同时返回样本数和平均损失
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        # 额外信息用于日志记录
        avg_ce_loss = total_ce_loss / (len(self.train_loader) * self.epochs)
        avg_prox_loss = total_prox_loss / (len(self.train_loader) * self.epochs)
        
        print(f"客户端 {self.client_id} 第{sync_round}轮: "
              f"损失={avg_loss:.4f}, 任务损失={avg_ce_loss:.4f}, 近端损失={avg_prox_loss:.4f}")
              
        return model_weights, num_sample, avg_loss
