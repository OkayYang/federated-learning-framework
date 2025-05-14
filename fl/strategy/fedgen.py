# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/10 11:07
# @Describe: Implementation of FedGen (Federated Learning with Generative Models)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from fl.fl_base import BaseClient
from fl.model.model import Generator

class FedGen(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 获取模型最后一层的输入维度作为特征维度
        self.feature_dim = kwargs.get('feature_dim', 256)
        self.num_classes = kwargs.get('num_classes', 62)  # 假设FEMNIST有62个类别
        
        # 生成器超参数
        self.latent_dim = kwargs.get('latent_dim', 64)
        self.hidden_dim = kwargs.get('hidden_dim', 256)
        self.alpha = kwargs.get('alpha', 10)  # 知识蒸馏损失的权重
        self.beta = kwargs.get('beta', 10)   # 生成器损失的权重
        
        # 初始化生成器
        self.generator = Generator(
            latent_dim=self.latent_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        )
        # 统计标签
        self.label_count = {}
        for data, target in self.train_loader:
            for t in target:
                label = t.item()
                if label not in self.label_count:
                    self.label_count[label] = 0
                self.label_count[label] += 1 

    def _exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr
     
    def local_train(self, sync_round: int, weights=None, generator_weights=None):
        """
        FedGen本地训练过程
        """
        # 1. 加载全局模型权重
        if weights is not None:
            self.update_weights(weights)
        
        # 2. 更新生成器模型
        if generator_weights is not None:
            self.generator.update_weights(generator_weights)
        
        # 4. 开始本地训练
        self.model.train()
        self.generator.eval()
        total_loss = 0
        num_sample = len(self.train_loader.dataset)
        total_batches = len(self.train_loader) * self.epochs

        
        with tqdm(
            total=total_batches,
            desc=f"Client {self.client_id} Training Progress (FedGen)"
        ) as pbar:
            for epoch in range(self.epochs):
                epoch_loss = 0
                epoch_teacher_loss = 0
                epoch_kd_loss = 0
                alpha = self._exp_lr_scheduler(sync_round*self.epochs + epoch, decay=0.98, init_lr=self.alpha)
                beta = self._exp_lr_scheduler(sync_round*self.epochs + epoch, decay=0.98, init_lr=self.beta)
                # 本地真实数据训练
                for data, target in self.train_loader:
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    logits = self.model(data)
                    # 计算分类损失
                    ce_loss = self.loss(logits, target)

                   
                    y_input = target.clone().detach()
                    batch_size = len(y_input)
                    # 生成噪声
                    z = torch.randn(batch_size, self.latent_dim)
                    # 生成特征
                    gen_features = self.generator(z, y_input)
                    
                    # 通过模型的分类层计算生成特征的预测
                    gen_logits = self.model(gen_features,start_layer=True)
                    
                    # 计算知识蒸馏损失
                    kd_loss = self._compute_kd_loss(gen_logits, y_input)
                    
                    # 随机生成样本
                    sampled_labels = np.random.choice(
                        self.num_classes, batch_size
                    )
                    sampled_labels = torch.LongTensor(sampled_labels)
                    sampled_features = self.generator(z, sampled_labels)
                    sampled_logits = self.model(sampled_features,start_layer=True)
                    
                    # 计算生成样本的分类损失
                    teacher_loss = self.loss(sampled_logits, sampled_labels)
                    
                    # 总损失
                    loss = ce_loss + alpha * kd_loss + beta * teacher_loss
                    
                    # 反向传播和优化
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_teacher_loss += teacher_loss.item()
                    epoch_kd_loss += kd_loss.item()
                    pbar.update(1)
                
                avg_loss = epoch_loss / (len(self.train_loader))
                total_loss += epoch_loss
                avg_teacher_loss = epoch_teacher_loss / (len(self.train_loader) )
                avg_kd_loss = epoch_kd_loss / (len(self.train_loader))

                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'teacher_loss': f"{avg_teacher_loss:.4f}",
                    'kd_loss': f"{avg_kd_loss:.4f}"
                })
        
        # 6. 获取更新后的模型权重
        model_weights = self.get_weights(return_numpy=True)
        
        # 7. 返回结果
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        return model_weights, num_sample, avg_loss,self.label_count
    
    
    def _compute_kd_loss(self, logits, labels, temperature=1.0):
        """
        计算知识蒸馏损失 - 从logits和标签
        
        Args:
            logits: 模型预测的logits
            labels: 目标标签（硬标签）
            temperature: 温度参数，控制软标签的平滑程度
            
        Returns:
            知识蒸馏损失
        """
        # 将硬标签转换为one-hot编码
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # 创建软目标 - 使用one-hot作为基础，但增加平滑度
        # 可以调整平滑度，这里使用0.1作为非目标类的概率
        smoothing = 0.1
        num_classes = logits.size(1)
        smooth_targets = one_hot * (1 - smoothing) + smoothing / num_classes
        
        # 应用温度缩放到学生logits
        student_log_softmax = F.log_softmax(logits / temperature, dim=1)
        
        # 计算KL散度损失
        kd_loss = -(smooth_targets * student_log_softmax).sum(dim=1).mean()
        
        # 应用温度平方调整梯度
        return kd_loss * (temperature ** 2)
    
