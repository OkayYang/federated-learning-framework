# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 11:07
# @Describe: Implementation of Federated Distillation (FD) algorithm

import torch
import torch.nn.functional as F
from tqdm import tqdm
from fl.client.fl_base import BaseClient
import torch.nn as nn

class FedDistill(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 获取模型输出层的类别数量
        self.num_classes = kwargs.get('num_classes')
        if self.num_classes is None:
            raise ValueError("num_classes must be provided")
        
        # 初始化logit存储用于知识蒸馏
        self.logit_storage = {}  # 存储每个类的平均logits
        self.logit_counts = {}   # 存储每个类的样本数量
        self.global_logits = {}  # 存储全局集成logits
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        # 蒸馏超参数
        self.gamma = kwargs.get('gamma', 0.5)  # 蒸馏损失的权重
        self.temperature = kwargs.get('temperature', 2.0)  # 温度用于软化概率分布
        
        # Initialize storage for each class
        for l in range(self.num_classes):
            self.logit_storage[l] = torch.zeros(self.num_classes, device=self.device)
            self.logit_counts[l] = 0
            self.global_logits[l] = torch.zeros(self.num_classes, device=self.device)

    def _compute_distillation_loss(self, student_logits, teacher_logits):
        
        
        # 应用温度缩放
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_softmax = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # 计算 KL 散度损失并应用温度平方
        return self.kl_div(student_log_softmax, soft_targets) * (self.temperature ** 2)

    def local_train(self, sync_round: int, weights=None, global_logits=None):
        """
        联邦学习算法：FedDistill, 联邦学习算法中，每个客户端独立训练，不进行互相通信
        """
        # 1. 更新模型权重(这里可以保持原论文一致不更新)
        if weights is not None:
            self.update_weights(weights)
        
        # 2. 更新全局logits
        if global_logits is not None:
            self.global_logits = global_logits

        # 3. 重置logit存储
        for l in range(self.num_classes):
            self.logit_storage[l] = torch.zeros(self.num_classes, device=self.device)
            self.logit_counts[l] = 0

        # 4. 开始本地训练
        self.model.train()
        total_loss = 0
        num_sample = len(self.train_loader.dataset)
        total_batches = len(self.train_loader) * self.epochs

        from fl.utils import ClientProgress
        
        with ClientProgress(
            progress_actor=self.progress_actor,
            client_id=self.client_id,
            total=total_batches,
            desc=f"Client {self.client_id} Training (FedDistill)"
        ) as pbar:
            for epoch in range(self.epochs):
                epoch_loss = 0
                kd_loss = 0
                ce_loss = 0
                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    logits = self.model(data)
                    
                    # 计算标准交叉熵损失
                    batch_ce_loss = self.loss(logits, target)
                    
                    # 如果全局logits可用，计算蒸馏损失
                    if global_logits is not None:
                        # 获取当前样本的教师logits
                        teacher_logits = torch.stack([self.global_logits[t.item()] for t in target])
                        distill_loss = self._compute_distillation_loss(logits, teacher_logits)
                        kd_loss += distill_loss.item()
                        ce_loss += batch_ce_loss.item()
                        # 联合损失
                        loss = batch_ce_loss + self.gamma * distill_loss
                    else:
                        loss = batch_ce_loss
                        ce_loss += batch_ce_loss.item()
                    
                    # 反向传播和优化
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # 更新logit存储
                    with torch.no_grad():
                        probs = F.softmax(logits, dim=1)
                        for i, t in enumerate(target):
                            label = t.item()
                            self.logit_storage[label] += probs[i]
                            self.logit_counts[label] += 1
                    
                    epoch_loss += loss.item()
                    pbar.update(1)
                    
                avg_loss = epoch_loss / len(self.train_loader)
                avg_kd_loss = kd_loss / len(self.train_loader)
                avg_ce_loss = ce_loss / len(self.train_loader)
                total_loss += epoch_loss
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'ce_loss': f"{avg_ce_loss:.4f}",
                    'kd_loss': f"{avg_kd_loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
        self.scheduler.step()

        # 5. 计算每个类的平均logits
        averaged_logits = {}
        for label in range(self.num_classes):
            if self.logit_counts[label] > 0:
                averaged_logits[label] = self.logit_storage[label] / self.logit_counts[label]
            else:
                averaged_logits[label] = torch.zeros(self.num_classes, device=self.device)

        # 6. 获取模型权重
        model_weights = self.get_weights(return_numpy=True)
        
        # 7. 返回结果
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        return model_weights, num_sample, avg_loss, averaged_logits

