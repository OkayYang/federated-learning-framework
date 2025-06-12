# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/5/16 11:07
# @Describe:
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from fl.client.fl_base import BaseClient

class FedSPD(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 保持简单的类别平衡参数
        self.class_counts = self._compute_class_counts()
        self.class_weights = self._compute_class_weights(self.class_counts)
        
        # 知识蒸馏核心参数
        self.temperature = kwargs.get('temperature', 1.0)  # 温度参数
        self.gamma1 = kwargs.get('gamma1', 1.0)           # logit蒸馏权重系数
        self.gamma2 = kwargs.get('gamma2', 1.0)           # 表征蒸馏权重系数
        
        # 初始化损失函数
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')  # KL散度损失
        self.mse_loss = nn.MSELoss(reduction='mean')        # MSE损失用于表征蒸馏

    def local_train(self, sync_round: int, weights=None, global_reps=None, global_logits=None):
        """
        训练方法，实现新版本的本地蒸馏损失设计
        L = φ(z, y) + γ₁ · w_y^(i) · φ(z, ẑ_y) + γ₂ · w_y^(i) · ||r - r̂_y||²
        
        :param weights: 服务器传递过来的模型权重
        :param sync_round: 当前的通信轮次
        :param global_reps: 服务器聚合的类别表征 {class_id: representation}
        :param global_logits: 服务器聚合的类别logits {class_id: logits}
        """
        # 1. 加载服务器传来的全局模型权重（仅分类层）
        if weights is not None:
            self.update_weights(weights)

        # 2. 开始本地训练
        self.model.train()
        total_loss = 0
        num_sample = len(self.train_loader.dataset)
        total_batches = len(self.train_loader) * self.epochs
        
        # 用于收集类别平均表征和类别平均输出
        class_reps = {}
        class_logits = {}
        
        with tqdm(
            total=total_batches,
            desc=f"Client {self.client_id} Training Progress (FedSPD Enhanced)"
        ) as pbar:
            for epoch in range(self.epochs):
                epoch_loss = 0
                epoch_ce_loss = 0
                epoch_logit_kd_loss = 0
                epoch_rep_kd_loss = 0
                
                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # 获取中间表征和输出
                    hidden, repout, output = self.model(data, return_all=True)
                    
                    # 1. 本地监督损失（正常交叉熵）: L_ce = φ(z, y)
                    ce_loss = self.loss(output, target)
                    
                    # 2. 初始化蒸馏损失
                    logit_kd_loss = torch.tensor(0.0, device=self.device)
                    rep_kd_loss = torch.tensor(0.0, device=self.device)
                    
                    # 3. 如果有全局表征和logits，计算蒸馏损失
                    if global_reps is not None and global_logits is not None:
                        batch_logit_losses = []
                        batch_rep_losses = []
                        
                        # 对batch中的每个样本计算蒸馏损失
                        for i, t in enumerate(target):
                            class_id = t.item()
                            
                            # 检查该类别是否有全局知识
                            if class_id in global_reps and class_id in global_logits:
                                # 获取类别权重 w_y^(i)
                                
                                
                                # 2. Logit蒸馏: L_logit = w_y^(i) · φ(z, ẑ_y)
                                global_logit = torch.tensor(global_logits[class_id], device=self.device).unsqueeze(0)
                                student_logit = output[i].unsqueeze(0)
                                
                                # 使用温度缩放的KL散度
                                with torch.no_grad():
                                    teacher_prob = F.softmax(global_logit / self.temperature, dim=1)
                                student_log_prob = F.log_softmax(student_logit / self.temperature, dim=1)
                                
                                sample_logit_loss = self.kl_loss(student_log_prob, teacher_prob) * (self.temperature ** 2)
                                weighted_logit_loss = sample_logit_loss
                                batch_logit_losses.append(weighted_logit_loss)
                                
                                # 3. 表征蒸馏: L_rep = w_y^(i) · ||r - r̂_y||²
                                global_rep = torch.tensor(global_reps[class_id], device=self.device)
                                student_rep = repout[i]
                                
                                sample_rep_loss = self.mse_loss(student_rep, global_rep)
                                weighted_rep_loss =  sample_rep_loss
                                batch_rep_losses.append(weighted_rep_loss)
                        
                        # 计算batch平均蒸馏损失
                        if batch_logit_losses:
                            logit_kd_loss = torch.stack(batch_logit_losses).mean()
                        if batch_rep_losses:
                            rep_kd_loss = torch.stack(batch_rep_losses).mean()
                    
                    # 4. 最终损失函数: L = φ(z, y) + γ₁ · w_y^(i) · φ(z, ẑ_y) + γ₂ · w_y^(i) · ||r - r̂_y||²
                    total_batch_loss = ce_loss + self.gamma1 * logit_kd_loss + self.gamma2 * rep_kd_loss
                    
                    # 只在最后一个epoch收集类别表征和logits
                    if epoch == self.epochs - 1:
                        for i, t in enumerate(target):
                            y = t.item()
                            if y not in class_reps:
                                class_reps[y] = []
                                class_logits[y] = []
                            class_reps[y].append(repout[i].detach().cpu().numpy())
                            class_logits[y].append(output[i].detach().cpu().numpy())
                    
                    # 反向传播和优化
                    total_batch_loss.backward()
                    self.optimizer.step()
                    
                    # 记录损失
                    epoch_loss += total_batch_loss.item()
                    epoch_ce_loss += ce_loss.item()
                    epoch_logit_kd_loss += logit_kd_loss.item() if isinstance(logit_kd_loss, torch.Tensor) else 0
                    epoch_rep_kd_loss += rep_kd_loss.item() if isinstance(rep_kd_loss, torch.Tensor) else 0

                    # 更新进度条
                    pbar.update(1)
                    
                total_loss += epoch_loss
                avg_loss = epoch_loss / len(self.train_loader)
                avg_ce_loss = epoch_ce_loss / len(self.train_loader)
                avg_logit_kd_loss = epoch_logit_kd_loss / len(self.train_loader)
                avg_rep_kd_loss = epoch_rep_kd_loss / len(self.train_loader)
                current_lr = self.optimizer.param_groups[0]['lr']

                # 打印损失信息
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'total': f"{avg_loss:.4f}",
                    'ce': f"{avg_ce_loss:.4f}",
                    'logit_kd': f"{avg_logit_kd_loss:.4f}",
                    'rep_kd': f"{avg_rep_kd_loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
        self.scheduler.step()
        # 3. 计算每个类别的平均表征和平均输出
        avg_class_reps = {}
        avg_class_logits = {}
        for y in class_reps:
            if class_reps[y]:
                avg_class_reps[y] = np.mean(np.array(class_reps[y]), axis=0)
                avg_class_logits[y] = np.mean(np.array(class_logits[y]), axis=0)
        
        # 4. 获取训练后的权重
        model_weights = self.get_weights(return_numpy=True)

        # 5. 返回更新后的权重、样本数、平均损失以及类别表征和输出
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        return model_weights, num_sample, avg_loss, avg_class_reps, avg_class_logits
    
    def _compute_class_counts(self):
        """计算本地数据集中各类别的样本数量"""
        class_counts = {}
        for _, target in self.train_loader:
            for t in target:
                y = t.item()
                class_counts[y] = class_counts.get(y, 0) + 1
        return class_counts
    
    def _compute_class_weights(self, class_counts):
        """
        计算类别的加权系数 w_y^(i)
        使用平方根加权来平衡类别不均衡问题
        """
        weights = {}
        if not class_counts:
            return weights
            
        # 计算总样本数和平均每类样本数
        total_samples = sum(class_counts.values())
        avg_samples = total_samples / len(class_counts) if len(class_counts) > 0 else 0
        
        # 使用平方根加权平衡类别
        for y, count in class_counts.items():
            weights[y] = np.sqrt(avg_samples / max(count, 1))
            
        # 归一化权重
        if weights:
            sum_weights = sum(weights.values())
            if sum_weights > 0:
                for y in weights:
                    weights[y] = weights[y] / sum_weights
                    
        return weights