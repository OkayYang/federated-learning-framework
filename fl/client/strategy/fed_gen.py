# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/10 11:07
# @Describe: Implementation of FedGen (Federated Learning with Generative Models)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from fl.client.fl_base import BaseClient
from fl.model.fedgen_generator import FedGenGenerator

class FedGen(BaseClient):
    """
    FedGen客户端实现
    
    基于原论文: "Data-Free Knowledge Distillation for Heterogeneous Federated Learning"
    原始实现: https://github.com/zhuangdizhu/FedGen
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 获取必要参数
        self.feature_dim = kwargs.get('feature_dim')
        self.num_classes = kwargs.get('num_classes')  
        self.dataset_name = kwargs.get('dataset', 'cifar10').lower()
        
        if self.num_classes is None or self.feature_dim is None:
            raise ValueError("num_classes and feature_dim must be provided")
        
        # 根据数据集设置不同的超参数
        self._set_dataset_params(kwargs)
        
        # 初始化生成器
        generator_model = kwargs.get('generator_model')
        if generator_model is not None:
            self.generator = generator_model
        else:
            self.generator = FedGenGenerator(
                feature_dim=self.feature_dim,
                num_classes=self.num_classes,
                dataset_name=self.dataset_name
            ).to(self.device)

        # 初始化KL散度损失函数
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        # 统计标签分布，用于服务器端生成器训练
        self.label_count = self._collect_label_counts()

    def _set_dataset_params(self, kwargs):
        """根据数据集设置不同的超参数"""
        # 默认参数
        self.alpha = kwargs.get('alpha', 1.0)  # 知识蒸馏损失的权重
        self.beta = kwargs.get('beta', 1.0)   # 生成器损失的权重
        self.temperature = kwargs.get('temperature', 1.0)  # 温度系数
        
        # 根据数据集设置特定参数
        if self.dataset_name == "mnist":
            self.alpha = kwargs.get('alpha', 1.0)
            self.beta = kwargs.get('beta', 1.0)
            self.temperature = kwargs.get('temperature', 1.0)
        elif self.dataset_name == "femnist" or self.dataset_name == "emnist":
            self.alpha = kwargs.get('alpha', 1.0)
            self.beta = kwargs.get('beta', 1.0)
            self.temperature = kwargs.get('temperature', 1.0)
        elif self.dataset_name == "cifar10":
            self.alpha = kwargs.get('alpha', 10.0)
            self.beta = kwargs.get('beta', 5.0)
            self.temperature = kwargs.get('temperature', 1.0)
        elif self.dataset_name == "cifar100":
            self.alpha = kwargs.get('alpha', 10.0)
            self.beta = kwargs.get('beta', 5.0)
            self.temperature = kwargs.get('temperature', 1.0)

    def _collect_label_counts(self):
        """收集本地数据集的标签分布"""
        label_count = {}
        for data, target in self.train_loader:
            for t in target:
                label = t.item()
                if label not in label_count:
                    label_count[label] = 0
                label_count[label] += 1
        return label_count

    def _exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """动态学习率调度器，与原论文保持一致"""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr
        
    def _compute_distillation_loss(self, student_logits, teacher_logits):
        """计算知识蒸馏损失（KL散度）"""
        # 计算 softmax 和 log_softmax
        student_log_softmax = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1).detach()
        
        # 计算 KL 散度并应用温度平方调整梯度
        return self.kl_div(student_log_softmax, soft_targets) * (self.temperature ** 2)
     
    def local_train(self, sync_round: int, weights=None, generator_weights=None):
        """
        FedGen本地训练过程，包括知识蒸馏
        
        Args:
            sync_round: 当前通信轮次
            weights: 全局模型权重
            generator_weights: 生成器模型权重
            
        Returns:
            tuple: (更新后的模型权重，样本数量，平均损失，标签计数)
        """
        # 1. 加载全局模型权重
        if weights is not None:
            self.update_weights(weights)
        
        # 2. 更新生成器模型
        if generator_weights is not None:
            self.generator.update_weights(generator_weights)
        
        # 开始本地训练
        self.model.train()
        self.generator.eval()  # 生成器处于评估模式
        total_loss = 0
        num_sample = len(self.train_loader.dataset)
        total_batches = len(self.train_loader) * self.epochs

        with tqdm(
            total=total_batches,
            desc=f"Client {self.client_id} Training Progress (FedGen)"
        ) as pbar:
            for epoch in range(self.epochs):
                epoch_loss = 0
                epoch_ce_loss = 0
                epoch_kd_loss = 0
                epoch_gen_loss = 0
                
                # 动态调整权重
                alpha = self._exp_lr_scheduler(sync_round, decay=0.98, init_lr=self.alpha)
                beta = self._exp_lr_scheduler(sync_round, decay=0.98, init_lr=self.beta)
                
                # 本地真实数据训练
                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    logits = self.model(data)
                    # 计算分类损失
                    ce_loss = self.loss(logits, target)

                    # 知识蒸馏部分 - 使用生成器
                    # 1. 生成当前批次对应标签的合成特征
                    with torch.no_grad():
                        _, gen_features = self.generator(target)
                    
                    # 2. 通过模型的分类层计算生成特征的预测
                    gen_logits = self.model(gen_features, start_layer="classify")
                    
                    # 3. 计算知识蒸馏损失
                    kd_loss = self._compute_distillation_loss(logits, gen_logits)
                    
                    # 随机生成样本进行知识传递
                    # 1. 随机采样标签
                    sampled_labels = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)
                    
                    # 2. 生成合成特征
                    with torch.no_grad():  
                        _, sampled_features = self.generator(sampled_labels)
                    
                    # 3. 计算生成样本的预测
                    sampled_logits = self.model(sampled_features, start_layer="classify")
                    
                    # 4. 计算生成样本的分类损失
                    gen_loss = self.loss(sampled_logits, sampled_labels)
                    
                    # 总损失 = 分类损失 + alpha * 知识蒸馏损失 + beta * 生成样本损失
                    loss = ce_loss + alpha * kd_loss + beta * gen_loss
                    
                    # 反向传播和优化
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # 记录损失
                    epoch_loss += loss.item()
                    epoch_ce_loss += ce_loss.item()
                    epoch_kd_loss += kd_loss.item()
                    epoch_gen_loss += gen_loss.item()
                    pbar.update(1)
                
                # 计算平均损失
                avg_loss = epoch_loss / len(self.train_loader)
                avg_ce_loss = epoch_ce_loss / len(self.train_loader)
                avg_kd_loss = epoch_kd_loss / len(self.train_loader)
                avg_gen_loss = epoch_gen_loss / len(self.train_loader)
                total_loss += epoch_loss
                
                # 更新进度条
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'ce_loss': f"{avg_ce_loss:.4f}",
                    'kd_loss': f"{avg_kd_loss:.4f}",
                    'gen_loss': f"{avg_gen_loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
        
        # 学习率调度
        if self.scheduler:
            self.scheduler.step()
        
        # 获取更新后的模型权重
        model_weights = self.get_weights(return_numpy=True)
        
        # 返回结果
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        return model_weights, num_sample, avg_loss, self.label_count
    
    
   
    
