# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 11:07
# @Describe:
import copy
import torch
from tqdm import tqdm

from fl.client.fl_base import BaseClient

class Moon(BaseClient):
    """Moon算法实现"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = kwargs.get('mu', 1)
        self.temperature = kwargs.get('temperature', 2.0)  # 从kwargs中获取temperature参数
        self.cosine_similarity_fn = torch.nn.CosineSimilarity(dim=-1)
        # 设置MOON特定参数
        self.buffer_size = 1
        self.prev_model_list = []  # 历史模型列表
        self.global_model = copy.deepcopy(self.model).to(self.device)  # 全局模型副本
        

    def local_train(self, sync_round: int, weights=None):
        """
        训练方法，根据当前通信轮次(sync_round)进行相应的训练更新
        :param weights: 服务器传递过来的模型权重
        :param sync_round: 当前的通信轮次
        """
        # 1. 加载服务器传来的全局模型权重
        if weights is not None:
            self.update_weights(weights)
            self.global_model.load_state_dict(self.model.state_dict())
            # 设置全局模型为评估模式
            self.global_model.eval()
            for param in self.global_model.parameters():
                param.requires_grad = False

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
            desc=f"Client {self.client_id} Training (Moon)"
        ) as pbar:
            for epoch in range(self.epochs):  # 多轮本地训练
                epoch_loss = 0
                epoch_contrastive_loss = 0
                epoch_original_loss = 0
                for data, target in self.train_loader:  # 获取每个 batch
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()  # 清除之前的梯度
                    # 前向传播 - 获取中间表示和预测结果
                    _, pro1, output = self.model(data, return_all=True)
                    # 计算原始损失
                    original_loss = self.loss(output, target)
                    # 对比损失计算
                    contrastive_loss = 0
                    # 获取全局模型的中间表示
                    with torch.no_grad():
                        _, pro2, _ = self.global_model(data, return_all=True)
                    posi = self.cosine_similarity_fn(pro1, pro2)
                    logits = posi.reshape(-1, 1)
                    # 计算与历史模型的表示相似度（负样本）
                    for prev_model in self.prev_model_list:
                        with torch.no_grad():
                            _, pro3, _ = prev_model(data, return_all=True)
                        nega = self.cosine_similarity_fn(pro1, pro3)
                        logits = torch.cat((logits,nega.reshape(-1,1)), dim=1)
                    logits = logits / self.temperature
                    labels = torch.zeros(data.size(0), dtype=torch.long, device=self.device)
                    contrastive_loss = self.mu * self.loss(logits, labels)
                    
                    # 合并损失
                    loss = original_loss + contrastive_loss
                    loss.backward()  # 反向传播
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()  # 更新模型参数
                    epoch_loss += loss.item()  # 累加损失
                    epoch_contrastive_loss += contrastive_loss.item()
                    epoch_original_loss += original_loss.item()


                    # 更新进度条
                    pbar.update(1)
                total_loss += epoch_loss
                avg_loss = epoch_loss / len(self.train_loader)
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'original_loss': f"{epoch_original_loss:.4f}",
                    'contrastive_loss': f"{epoch_contrastive_loss:.4f}",
                    'μ': f"{self.mu}", 
                    'T': f"{self.temperature}",
                    'lr': f"{current_lr:.6f}"
                })
        self.scheduler.step()
        # 4. 获取训练后的权重
        model_weights = self.get_weights(return_numpy=True)
        # 创建当前模型的副本，并添加到历史模型列表
        history_model = copy.deepcopy(self.model).to(self.device)
        history_model.eval()
        for param in history_model.parameters():
            param.requires_grad = False
        self.prev_model_list.append(history_model)
        if len(self.prev_model_list) > self.buffer_size:
            self.prev_model_list.pop(0)
        # 5. 返回更新后的权重给服务器，同时返回样本数和平均损失
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        return model_weights, num_sample, avg_loss
