# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/20 10:30
# @Describe: FEDGKD (Federated Global Knowledge Distillation) 客户端策略

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import copy

from fl.client.fl_base import BaseClient
from fl.utils import update_model_weights

class FedGKD(BaseClient):
    """
    FEDGKD (Federated Global Knowledge Distillation) 客户端策略
    
    通过历史全局模型的知识蒸馏引导本地训练，缓解客户端漂移问题
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 知识蒸馏参数
        self.temperature = kwargs.get('temperature', 1.0)  # 温度参数
        self.alpha = kwargs.get('alpha', 0.5)  # 蒸馏损失权重
        self.vote_mode = kwargs.get('vote_mode', False)  # 是否使用FEDGKD-VOTE模式
        
        # 初始化蒸馏损失函数
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def _compute_kd_loss(self, student_logits, teacher_logits):
        """
        计算知识蒸馏损失 (KL散度)
        
        Args:
            student_logits: 学生模型的logits
            teacher_logits: 教师模型的logits
            
        Returns:
            kd_loss: 知识蒸馏损失
        """
        # 教师logits已经在调用方法中detach，这里直接使用
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # 计算学生模型的log softmax
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # 计算KL散度
        kd_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        return kd_loss
    
    def _compute_ensemble_logits(self, data, ensemble_weights):
        """
        使用集成模型计算logits
        
        Args:
            data: 输入数据
            ensemble_weights: 集成模型权重
            
        Returns:
            ensemble_logits: 集成模型的logits
        """
        # 创建一个新的模型实例，避免修改当前模型
        ensemble_model = copy.deepcopy(self.model)
        ensemble_model.eval()
        
        # 将权重加载到新模型中
        update_model_weights(ensemble_model, ensemble_weights)
        
        # 计算集成模型的logits
        with torch.no_grad():
            ensemble_logits = ensemble_model(data).detach()
        
        return ensemble_logits
    
    def _compute_vote_logits(self, data, historical_models):
        """
        使用多个历史模型投票计算logits (FEDGKD-VOTE模式)
        
        Args:
            data: 输入数据
            historical_models: 历史模型权重列表
            
        Returns:
            vote_logits: 投票后的logits
        """
        # 创建一个新的模型实例，避免修改当前模型
        vote_model = copy.deepcopy(self.model)
        vote_model.eval()
        
        # 收集所有历史模型的logits
        all_logits = []
        
        for model_weights in historical_models:
            # 加载历史模型权重到新模型
            update_model_weights(vote_model, model_weights)
            
            # 计算该模型的logits
            with torch.no_grad():
                logits = vote_model(data).detach()
                all_logits.append(logits)
        
        # 简单平均所有模型的logits (软投票)
        if all_logits:
            vote_logits = torch.stack(all_logits).mean(dim=0)
            return vote_logits
        
        # 如果没有历史模型，返回None
        return None
    
    def local_train(self, sync_round: int, weights=None, ensemble_weights=None, historical_models=None):
        """
        FEDGKD本地训练方法
        
        Args:
            sync_round: 当前通信轮次
            weights: 服务器传递过来的当前全局模型权重
            ensemble_weights: 服务器传递的集成模型权重 (FEDGKD模式)
            historical_models: 服务器传递的历史模型权重列表 (FEDGKD-VOTE模式)
            
        Returns:
            tuple: (更新后的模型权重, 样本数, 平均损失)
        """
        # 启用异常检测，帮助定位问题
        torch.autograd.set_detect_anomaly(True)
        
        # 1. 加载服务器传来的全局模型权重
        if weights is not None:
            self.update_weights(weights)
        
        # 2. 开始本地训练
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_kd_loss = 0
        num_sample = len(self.train_loader.dataset)
        total_batches = len(self.train_loader) * self.epochs
        
        # 确定使用哪种教师模式
        if self.vote_mode and historical_models and len(historical_models) > 0:
            teacher_mode = "vote"
        elif ensemble_weights is not None:
            teacher_mode = "ensemble"
        else:
            teacher_mode = None
        
        from fl.utils import ClientProgress
        
        with ClientProgress(
            progress_actor=self.progress_actor,
            client_id=self.client_id,
            total=total_batches,
            desc=f"Client {self.client_id} Training (FedGKD)"
        ) as pbar:
            for epoch in range(self.epochs):
                epoch_loss = 0
                epoch_ce_loss = 0
                epoch_kd_loss = 0
                
                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    output = self.model(data)
                    
                    # 计算交叉熵损失
                    ce_loss = self.loss(output, target)
                    
                    # 初始化蒸馏损失
                    kd_loss = torch.tensor(0.0, device=self.device)
                    
                    # 根据模式计算蒸馏损失
                    if teacher_mode == "vote":
                        # FEDGKD-VOTE模式: 使用多个历史模型投票
                        teacher_logits = self._compute_vote_logits(data, historical_models)
                        if teacher_logits is not None:
                            kd_loss = self._compute_kd_loss(output, teacher_logits)
                    elif teacher_mode == "ensemble":
                        # 默认FEDGKD模式: 使用集成模型作为教师
                        teacher_logits = self._compute_ensemble_logits(data, ensemble_weights)
                        kd_loss = self._compute_kd_loss(output, teacher_logits)
                    
                    # 计算总损失: L = (1-α)*CE + α*KD
                    total_batch_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
                    
                    # 反向传播和优化
                    total_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # 累加损失
                    epoch_loss += total_batch_loss.item()
                    epoch_ce_loss += ce_loss.item()
                    epoch_kd_loss += kd_loss.item() if isinstance(kd_loss, torch.Tensor) else 0
                    
                    # 更新进度条
                    pbar.update(1)
                
                # 计算平均损失
                avg_loss = epoch_loss / len(self.train_loader)
                avg_ce_loss = epoch_ce_loss / len(self.train_loader)
                avg_kd_loss = epoch_kd_loss / len(self.train_loader)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 更新进度条信息
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'ce_loss': f"{avg_ce_loss:.4f}",
                    'kd_loss': f"{avg_kd_loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
                
                # 累加总损失
                total_loss += epoch_loss
                total_ce_loss += epoch_ce_loss
                total_kd_loss += epoch_kd_loss
        
        # 学习率调度
        self.scheduler.step()
        
        # 3. 获取训练后的权重
        model_weights = self.get_weights(return_numpy=True)
        
        # 4. 返回更新后的权重、样本数和平均损失
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        
        return model_weights, num_sample, avg_loss
