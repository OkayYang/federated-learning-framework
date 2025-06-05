# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/5/16 11:07
# @Describe: FedSmart - Intelligent Adaptive Federated Learning (Simple but Powerful)
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from fl.client.fl_base import BaseClient

class FedSPD(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # FedSmart核心参数 - 简单但强力
        self.temperature = kwargs.get('temperature', 3.0)       # 蒸馏温度
        self.kd_weight = kwargs.get('kd_weight', 0.5)           # 知识蒸馏权重
        self.momentum = kwargs.get('momentum', 0.9)             # 模型动量
        self.adaptive_lr = kwargs.get('adaptive_lr', True)      # 自适应学习率
        
        # 自适应训练参数
        self.min_epochs = 1
        self.max_epochs = min(self.epochs * 2, 10)              # 最大训练轮数
        self.patience = 3                                       # 早停耐心
        self.loss_threshold = 0.01                              # 损失变化阈值
        
        # 数据分布感知
        self.local_data_stats = {}
        self.data_heterogeneity = 0.0
        
        # 模型动量缓存
        self.momentum_params = None
        self.prev_global_params = None
        
        # 简单有效的损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        print(f"🚀 FedSmart客户端 {self.client_id} 初始化完成")

    def local_train(self, sync_round: int, weights=None, global_reps=None, global_logits=None):
        """
        FedSmart训练 - 智能自适应联邦学习
        
        核心创新：
        1. 自适应本地训练轮数
        2. 模型动量更新
        3. 简单但有效的知识蒸馏
        4. 数据分布感知学习率调整
        """
        
        # 1. 更新模型权重和动量
        # if weights is not None:
        #     self._smart_model_update(weights, sync_round)
        
        # 2. 分析本地数据分布
        self._analyze_local_data()
        
        # 3. 自适应确定训练轮数
        adaptive_epochs = self._compute_adaptive_epochs(sync_round)
        
        # 4. 自适应学习率
        if self.adaptive_lr:
            self._adjust_learning_rate(sync_round)
        
        # 5. 开始智能训练
        self.model.train()
        total_loss = 0
        num_sample = len(self.train_loader.dataset)
        
        # 收集类别数据
        class_reps = {}
        class_logits = {}
        
        # 早停机制
        prev_loss = float('inf')
        patience_counter = 0
        
        print(f"📊 客户端 {self.client_id}: 自适应轮数={adaptive_epochs}, 异构度={self.data_heterogeneity:.3f}")
        
        with tqdm(
            total=adaptive_epochs * len(self.train_loader),
            desc=f"Client {self.client_id} FedSmart Training"
        ) as pbar:
            
            for epoch in range(adaptive_epochs):
                epoch_loss = 0
                epoch_ce_loss = 0
                epoch_kd_loss = 0
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    if hasattr(self.model, 'forward') and 'return_all' in self.model.forward.__code__.co_varnames:
                        hidden, repout, output = self.model(data, return_all=True)
                    else:
                        output = self.model(data)
                        hidden = repout = output  # fallback
                    
                    # 1. 监督学习损失
                    ce_loss = self.ce_loss(output, target)
                    
                    # 2. 智能知识蒸馏
                    kd_loss = self._compute_smart_kd_loss(output, target, global_logits)
                    
                    # 3. 自适应损失权重
                    adaptive_kd_weight = self._compute_adaptive_kd_weight(sync_round, kd_loss, ce_loss)
                    
                    # 4. 总损失
                    total_batch_loss = ce_loss + adaptive_kd_weight * kd_loss
                    
                    # 反向传播
                    total_batch_loss.backward()
                    
                    # 梯度裁剪（防止梯度爆炸）
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    # 收集数据（最后一个epoch）
                    if epoch == adaptive_epochs - 1:
                        self._collect_class_data(target, repout, output, class_reps, class_logits)
                    
                    # 记录损失
                    epoch_loss += total_batch_loss.item()
                    epoch_ce_loss += ce_loss.item()
                    epoch_kd_loss += kd_loss.item() if isinstance(kd_loss, torch.Tensor) else 0
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'epoch': f"{epoch+1}/{adaptive_epochs}",
                        'ce': f"{ce_loss.item():.4f}",
                        'kd': f"{kd_loss.item() if isinstance(kd_loss, torch.Tensor) else 0:.4f}",
                        'α': f"{adaptive_kd_weight:.3f}"
                    })
                
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                total_loss += epoch_loss
                
                # 早停检查
                if abs(prev_loss - avg_epoch_loss) < self.loss_threshold:
                    patience_counter += 1
                    if patience_counter >= self.patience and epoch >= self.min_epochs:
                        print(f"🛑 客户端 {self.client_id} 早停于第 {epoch+1} 轮")
                        break
                else:
                    patience_counter = 0
                    
                prev_loss = avg_epoch_loss
        
        # 6. 计算平均类别输出
        avg_class_reps, avg_class_logits = self._compute_average_class_outputs(
            class_reps, class_logits
        )
        
        # 7. 返回结果
        model_weights = self.get_weights(return_numpy=True)
        avg_loss = total_loss / (len(self.train_loader) * adaptive_epochs)
        
        print(f"✅ 客户端 {self.client_id} 训练完成: 损失={avg_loss:.4f}")
        
        return model_weights, num_sample, avg_loss, avg_class_reps, avg_class_logits

    def _smart_model_update(self, weights, sync_round):
        """智能模型更新 - 带动量的平滑更新"""
        self.update_weights(weights)
        
        # 保存当前全局参数
        current_global_params = {}
        for name, param in self.model.named_parameters():
            current_global_params[name] = param.data.clone()
        
        # 如果有历史参数，使用动量更新
        if self.prev_global_params is not None and sync_round > 1:
            for name, param in self.model.named_parameters():
                if name in self.prev_global_params:
                    # 计算全局模型的变化
                    global_change = current_global_params[name] - self.prev_global_params[name]
                    
                    # 动量更新
                    if self.momentum_params is None:
                        self.momentum_params = {}
                    
                    if name not in self.momentum_params:
                        self.momentum_params[name] = torch.zeros_like(global_change)
                    
                    self.momentum_params[name] = (
                        self.momentum * self.momentum_params[name] + 
                        (1 - self.momentum) * global_change
                    )
                    
                    # 应用动量
                    param.data = current_global_params[name] + 0.1 * self.momentum_params[name]
        
        self.prev_global_params = current_global_params

    def _analyze_local_data(self):
        """分析本地数据分布"""
        class_counts = {}
        total_samples = 0
        
        for _, target in self.train_loader:
            for t in target:
                y = t.item()
                class_counts[y] = class_counts.get(y, 0) + 1
                total_samples += 1
        
        # 计算数据异构度（基于类别分布的熵）
        if len(class_counts) > 1:
            probs = np.array(list(class_counts.values())) / total_samples
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            max_entropy = np.log(len(class_counts))
            self.data_heterogeneity = 1.0 - entropy / max_entropy  # 异构度：0=均匀分布，1=单一类别
        else:
            self.data_heterogeneity = 1.0
        
        self.local_data_stats = {
            'class_counts': class_counts,
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'heterogeneity': self.data_heterogeneity
        }

    def _compute_adaptive_epochs(self, sync_round):
        """自适应计算训练轮数"""
        base_epochs = self.epochs
        
        # 基于数据异构度调整
        if self.data_heterogeneity > 0.8:  # 高异构度
            epochs_factor = 1.5
        elif self.data_heterogeneity > 0.5:  # 中异构度
            epochs_factor = 1.2
        else:  # 低异构度
            epochs_factor = 1.0
        
        # 基于轮次调整（早期多训练，后期少训练）
        if sync_round <= 5:
            round_factor = 1.3
        elif sync_round <= 15:
            round_factor = 1.0
        else:
            round_factor = 0.8
        
        adaptive_epochs = int(base_epochs * epochs_factor * round_factor)
        return max(self.min_epochs, min(adaptive_epochs, self.max_epochs))

    def _adjust_learning_rate(self, sync_round):
        """自适应调整学习率"""
        # 基于轮次的学习率衰减
        if sync_round <= 10:
            lr_factor = 1.0
        elif sync_round <= 30:
            lr_factor = 0.8
        else:
            lr_factor = 0.6
        
        # 基于数据异构度调整
        if self.data_heterogeneity > 0.7:
            lr_factor *= 0.8  # 高异构度降低学习率
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group.get('initial_lr', param_group['lr']) * lr_factor

    def _compute_smart_kd_loss(self, student_logits, target, global_logits):
        """智能知识蒸馏损失 - 简单但有效"""
        if not global_logits:
            return torch.tensor(0.0, device=self.device)
        
        kd_losses = []
        
        for i, t in enumerate(target):
            class_id = t.item()
            
            if class_id in global_logits:
                # 获取全局知识
                teacher_logits = torch.tensor(
                    global_logits[class_id], 
                    device=self.device, 
                    dtype=torch.float32
                ).unsqueeze(0)
                
                student_logit = student_logits[i].unsqueeze(0)
                
                # 简单KL散度损失
                teacher_prob = F.softmax(teacher_logits / self.temperature, dim=1)
                student_log_prob = F.log_softmax(student_logit / self.temperature, dim=1)
                
                kd_loss = self.kl_loss(student_log_prob, teacher_prob.detach())
                kd_loss *= (self.temperature ** 2)
                
                kd_losses.append(kd_loss)
        
        return torch.stack(kd_losses).mean() if kd_losses else torch.tensor(0.0, device=self.device)

    def _compute_adaptive_kd_weight(self, sync_round, kd_loss, ce_loss):
        """自适应计算知识蒸馏权重"""
        base_weight = self.kd_weight
        
        # 基于损失比例调整
        if isinstance(kd_loss, torch.Tensor) and kd_loss.item() > 0:
            loss_ratio = ce_loss.item() / (kd_loss.item() + 1e-8)
            if loss_ratio > 10:  # CE损失远大于KD损失
                ratio_factor = 2.0
            elif loss_ratio > 3:
                ratio_factor = 1.5
            else:
                ratio_factor = 1.0
        else:
            ratio_factor = 0.0
        
        # 基于轮次调整
        if sync_round <= 3:
            round_factor = 2.0  # 早期更依赖知识蒸馏
        elif sync_round <= 10:
            round_factor = 1.0
        else:
            round_factor = 0.5  # 后期减少知识蒸馏
        
        # 基于数据异构度调整
        hetero_factor = 1.0 + self.data_heterogeneity  # 异构度越高，越需要知识蒸馏
        
        adaptive_weight = base_weight * ratio_factor * round_factor * hetero_factor
        return min(adaptive_weight, 2.0)  # 限制最大权重

    def _collect_class_data(self, target, repout, output, class_reps, class_logits):
        """收集类别数据"""
        for i, t in enumerate(target):
            y = t.item()
            if y not in class_reps:
                class_reps[y] = []
                class_logits[y] = []
            class_reps[y].append(repout[i].detach().cpu().numpy())
            class_logits[y].append(output[i].detach().cpu().numpy())

    def _compute_average_class_outputs(self, class_reps, class_logits):
        """计算平均类别输出"""
        avg_class_reps = {}
        avg_class_logits = {}
        
        for y in class_reps:
            if class_reps[y]:
                avg_class_reps[y] = np.mean(np.array(class_reps[y]), axis=0)
                avg_class_logits[y] = np.mean(np.array(class_logits[y]), axis=0)
        
        return avg_class_reps, avg_class_logits