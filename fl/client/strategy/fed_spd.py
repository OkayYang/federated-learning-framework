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
        self.temperature = kwargs.get('temperature', 2.0)  # 使用更低的温度使分布更加锐利
        self.alpha = kwargs.get('alpha', 0.5)             # 软目标和硬目标的平衡系数
        
        # 初始化KL散度损失函数，使用batchmean模式避免警告
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def local_train(self, sync_round: int, weights=None, global_reps=None, global_logits=None):
        """
        训练方法，根据当前通信轮次(sync_round)进行相应的训练更新
        :param weights: 服务器传递过来的模型权重
        :param sync_round: 当前的通信轮次
        :param global_reps: 服务器聚合的类别表征
        :param global_logits: 服务器聚合的类别logits
        """
        # 1. 加载服务器传来的全局模型权重
        # if weights is not None:
        #     self.update_weights(weights)

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
            desc=f"Client {self.client_id} Training Progress (FedSPD)"
        ) as pbar:
            for epoch in range(self.epochs):  # 多轮本地训练
                epoch_loss = 0
                epoch_ce_loss = 0
                epoch_kd_loss = 0
                for data, target in self.train_loader:  # 获取每个 batch
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()  # 清除之前的梯度
                    
                    # 获取中间表征和输出
                    head_out, _, output = self.model(data, return_all=True)
                    
                    # 初始化损失
                    ce_loss = self.loss(output, target)  # 交叉熵损失
                    kd_loss = torch.tensor(0.0, device=self.device)          # 知识蒸馏损失                    
                    # 如果有全局表征和logits，添加蒸馏损失
                    if global_reps is not None and global_logits is not None:
                        # 按类别收集当前批次的数据和对应的全局知识
                        batch_kd_loss = 0.0
                        valid_samples = 0
                        
                        # 收集当前批次中存在于全局知识中的类别
                        available_classes = []
                        for t in target:
                            class_id = t.item()
                            if class_id in global_reps and class_id in global_logits and class_id not in available_classes:
                                available_classes.append(class_id)
                        
                        # 如果有可用的类别知识
                        if available_classes:
                            # 将全局表征转换为张量并输入到模型中
                            global_features = []
                            global_targets = []
                            
                            for class_id in available_classes:
                                # 获取全局表征并转换为张量
                                global_rep = torch.tensor(global_reps[class_id], device=self.device)
                                global_features.append(global_rep)
                                
                                # 获取全局logits
                                global_target = torch.tensor(global_logits[class_id], device=self.device)
                                global_targets.append(global_target)
                            
                            # 堆叠为批次
                            global_features_batch = torch.stack(global_features)
                            global_targets_batch = torch.stack(global_targets)
                            
                            # 使用torch.no_grad()确保教师模型的输出不会记录梯度
                            with torch.no_grad():
                                # 将全局表征转换为不需要梯度的张量
                                global_features_batch = global_features_batch.detach()
                                global_targets_batch = global_targets_batch.detach()
                                
                                # 应用温度缩放到教师模型的输出
                                global_targets_scaled = global_targets_batch / self.temperature
                                probs = F.softmax(global_targets_scaled, dim=1)
                            
                            # 将全局表征输入到模型中获取输出
                            model_outputs = self.model(global_features_batch, start_layer=True)
                            
                            # 应用温度缩放到学生模型的输出
                            model_outputs_scaled = model_outputs / self.temperature
                            
                            # 计算KL散度损失
                            log_probs = F.log_softmax(model_outputs_scaled, dim=1)
                            kd_loss = self.kl_loss(log_probs, probs) * (self.temperature ** 2)
                        else:
                            kd_loss = torch.tensor(0.0, device=self.device)
                    
                    # 总损失 = CE损失 + KD损失
                    total_batch_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
                    
                    # 只在最后一个epoch收集类别表征和logits
                    if epoch == self.epochs - 1:
                        for i, t in enumerate(target):
                            y = t.item()
                            if y not in class_reps:
                                class_reps[y] = []
                                class_logits[y] = []
                            class_reps[y].append(head_out[i].detach().cpu().numpy())
                            class_logits[y].append(output[i].detach().cpu().numpy())
                    
                    # 反向传播和优化
                    total_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # 记录损失
                    epoch_loss += total_batch_loss.item()
                    epoch_ce_loss += ce_loss.item()
                    epoch_kd_loss += kd_loss.item() if isinstance(kd_loss, torch.Tensor) else 0

                    # 更新进度条
                    pbar.update(1)
                    
                total_loss += epoch_loss
                avg_loss = epoch_loss / len(self.train_loader)
                avg_ce_loss = epoch_ce_loss / len(self.train_loader)
                avg_kd_loss = epoch_kd_loss / len(self.train_loader)
                
                # 打印损失信息
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'ce': f"{avg_ce_loss:.4f}",
                    'kd': f"{avg_kd_loss:.4f}"
                })
        
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
        """计算类别的加权系数（使用简单的平方根加权）"""
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
