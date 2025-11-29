# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/5/16 11:07
# @Describe:
import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from fl.aggregation.aggregator import average_weight
from fl.client.fl_base import BaseClient
from fl.utils import update_model_weights

class FedSPDLC(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 知识蒸馏核心参数
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_buffer = []
        self.buffer_size = kwargs.get('buffer_size', 10)
        self.last_sync_round = -1  # 初始化轮次追踪

        self.temperature = kwargs.get('temperature', 1.0)  # 温度参数
        self.alpha = kwargs.get('alpha', 0.5)              # logits蒸馏权重
        self.beta = kwargs.get('beta', 0.3)                # 表征蒸馏权重
        self.rep_norm = kwargs.get('rep_norm', True)       # 表征归一化
        
        # 初始化损失函数
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')  # KL散度损失
        self.mse_loss = nn.MSELoss(reduction='mean')        # MSE损失用于表征蒸馏
        self.cosine_loss = nn.CosineEmbeddingLoss(reduction='mean')  # 余弦损失

    def _compute_representation_loss(self, student_reps, teacher_reps, loss_type='mse'):
        """
        计算表征蒸馏损失 - 针对异构数据优化
        
        Args:
            student_reps: 学生模型表征
            teacher_reps: 教师模型表征
            loss_type: 损失类型 ('mse', 'cosine', 'hybrid')
        """
        # 表征归一化 - 在异构数据中很重要
        if self.rep_norm:
            student_reps = F.normalize(student_reps, p=2, dim=1)
            teacher_reps = F.normalize(teacher_reps, p=2, dim=1)
        
        if loss_type == 'mse':
            return self.mse_loss(student_reps, teacher_reps)
        elif loss_type == 'cosine':
            # 余弦相似性损失 - 关注方向而非幅度
            target = torch.ones(student_reps.size(0), device=student_reps.device)
            return self.cosine_loss(student_reps, teacher_reps, target)
        elif loss_type == 'hybrid':
            # 混合损失 - 结合MSE和余弦
            mse_loss = self.mse_loss(student_reps, teacher_reps)
            target = torch.ones(student_reps.size(0), device=student_reps.device)
            cosine_loss = self.cosine_loss(student_reps, teacher_reps, target)
            return 0.7 * mse_loss + 0.3 * cosine_loss
        else:
            return self.mse_loss(student_reps, teacher_reps)

    def local_train(self, sync_round: int, weights=None):
        """
        FedSPD数据异构优化版本
        
        核心设计：
        - 针对数据异构场景优化表征蒸馏
        - 分离logits和表征蒸馏权重控制
        - 表征归一化提升对齐效果
        - 混合损失函数提升鲁棒性
        
        损失设计：
        L = CE + α * KD_logits + β * Rep_loss
        
        参数配置：
        - α: logits蒸馏权重 (默认0.5)
        - β: 表征蒸馏权重 (默认0.3)
        - rep_norm: 表征归一化 (默认True)
        
        :param weights: 服务器传递过来的当前全局模型权重
        :param sync_round: 当前的通信轮次  
        :param ensemble_weights: 服务器传递的教师权重（如果有的话）
        """
        # 1. 加载全局模型权重到本地模型
        if weights is not None:
            self.update_weights(weights)
        
        # 2. 创建教师模型（使用全局权重）- 确保每轮只添加一次
        if weights is not None:
            # 检查是否为新轮次或新权重，避免重复添加
            if not hasattr(self, 'last_sync_round') or sync_round != self.last_sync_round:
                self.teacher_buffer.append(weights)
                if len(self.teacher_buffer) > self.buffer_size:
                    self.teacher_buffer.pop(0)
                self.last_sync_round = sync_round
            
            # 构建教师模型（平均历史权重）
            if len(self.teacher_buffer) > 0:
                teacher_weights = average_weight(self.teacher_buffer)
                update_model_weights(self.teacher_model, teacher_weights)
                self.teacher_model.eval()
        
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
            desc=f"Client {self.client_id} Training (FedSPDLC)"
        ) as pbar:
            for epoch in range(self.epochs):
                epoch_loss = 0
                epoch_ce_loss = 0
                epoch_kd_loss = 0
                epoch_rep_loss = 0
                
                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # 学生模型前向传播
                    student_logits = self.model(data)
                    student_reps = None
                    
                    # 如果模型支持返回表征，获取表征
                    
                    _, student_reps, student_logits = self.model(data, return_all=True)
                        
                    
                    # 1. 本地监督学习损失
                    ce_loss = self.loss(student_logits, target)
                    
                    # 2. 教师模型指导（logits + 表征蒸馏）
                    kd_loss = torch.tensor(0.0, device=self.device)
                    rep_loss = torch.tensor(0.0, device=self.device)
                    
                    # 只有在有教师模型且teacher_buffer不为空时才进行蒸馏
                    if self.teacher_model is not None and len(self.teacher_buffer) > 0:
                        # 教师模型前向传播
                        with torch.no_grad():
                            teacher_logits = self.teacher_model(data)
                            teacher_reps = None
                            # 获取教师表征
                            _, teacher_reps, teacher_logits = self.teacher_model(data, return_all=True)
                            
                        # Logits蒸馏
                        with torch.no_grad():
                            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
                        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
                        kd_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
                        
                        # 表征蒸馏
                        if student_reps is not None and teacher_reps is not None:
                            rep_loss = self._compute_representation_loss(
                                student_reps, teacher_reps, loss_type='hybrid'
                            )
                    
                    # 3. 总损失：L = CE + α * KD + β * Rep
                    total_batch_loss = ce_loss + self.alpha * kd_loss + self.beta * rep_loss
                    
                    # 反向传播和优化
                    total_batch_loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    # 记录损失
                    epoch_loss += total_batch_loss.item()
                    epoch_ce_loss += ce_loss.item()
                    epoch_kd_loss += kd_loss.item()
                    epoch_rep_loss += rep_loss.item()

                    # 更新进度条
                    pbar.update(1)
                    
                total_loss += epoch_loss
                avg_loss = epoch_loss / len(self.train_loader)
                avg_ce_loss = epoch_ce_loss / len(self.train_loader)
                avg_kd_loss = epoch_kd_loss / len(self.train_loader)
                avg_rep_loss = epoch_rep_loss / len(self.train_loader)
                current_lr = self.optimizer.param_groups[0]['lr']

                # 打印损失信息
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'total': f"{avg_loss:.4f}",
                    'ce': f"{avg_ce_loss:.4f}",
                    'kd': f"{avg_kd_loss:.4f}",
                    'rep': f"{avg_rep_loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
        
        self.scheduler.step()
        
        # 获取训练后的权重
        model_weights = self.get_weights(return_numpy=True)

        # 返回更新后的权重、样本数、平均损失
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        return model_weights, num_sample, avg_loss