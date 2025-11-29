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
    FedGen客户端
    代码参考：https://github.com/zhuangdizhu/FedGen/
    论文复现：Data-Free Knowledge Distillation for Heterogeneous Federated Learning
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.feature_dim = kwargs.get('feature_dim')
        self.num_classes = kwargs.get('num_classes')  
        if self.num_classes is None or self.feature_dim is None:
            raise ValueError("num_classes and feature_dim must be provided")
        
        
        # 生成器超参数
        self.alpha = kwargs.get('alpha', 10)  # 知识蒸馏损失的权重
        self.beta = kwargs.get('beta', 10)   # 生成器损失的权重
        self.temperature = kwargs.get('temperature', 1.0)  # 温度用于软化概率分布
        self.init_generator = self.kwargs.get('generator_model')
        # 初始化生成器
        self.generator = FedGenGenerator(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
        ).to(self.device)
        self.generator.update_weights(self.init_generator.get_weights(return_numpy=True))

        # 初始化KL散度损失函数，使用batchmean模式避免警告
        self.kl_div = nn.KLDivLoss(reduction='batchmean')        
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
    def _compute_distillation_loss(self, student_logits, teacher_logits):
                # 计算 softmax 和 log_softmax
        student_log_softmax = F.log_softmax(student_logits, dim=-1)
        soft_targets=F.softmax(teacher_logits, dim=-1).clone().detach()
        
        # 计算 KL 散度并应用温度平方调整梯度
        return self.kl_div(student_log_softmax, soft_targets)
     
    def local_train(self, sync_round: int, weights=None, generator_weights=None):
        """
        FedGen本地训练过程
        """
        # 1. 加载预测层权重
        if weights is not None:
            self.update_weights(weights)
            # 获取当前模型的state_dict
            # state_dict = self.model.state_dict()
            # keys = list(state_dict.keys())
            # # 只更新最后一层(分类层)的权重和偏置
            # state_dict[keys[-2]] = torch.Tensor(weights[-2]).to(self.device)  # 分类层权重
            # state_dict[keys[-1]] = torch.Tensor(weights[-1]).to(self.device)  # 分类层偏置
            # # 更新模型权重
            # self.model.load_state_dict(state_dict)
        
        # 2. 更新生成器模型
        if generator_weights is not None:
            self.generator.update_weights(generator_weights)
        
        # 4. 开始本地训练
        self.model.train()
        self.generator.eval()
        total_loss = 0
        num_sample = len(self.train_loader.dataset)
        total_batches = len(self.train_loader) * self.epochs

        
        from fl.utils import ClientProgress
        
        with ClientProgress(
            progress_actor=self.progress_actor,
            client_id=self.client_id,
            total=total_batches,
            desc=f"Client {self.client_id} Training (FedGen)"
        ) as pbar:
            for epoch in range(self.epochs):
                epoch_loss = 0
                epoch_teacher_loss = 0
                epoch_kd_loss = 0
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

                    # 生成噪声
                    # 生成特征
                    with torch.no_grad():
                        eps,gen_features = self.generator(target)
                    
                    # 通过模型的分类层计算生成特征的预测
                    gen_logits = self.model(gen_features,start_layer="classify")
                    
                    # 计算知识蒸馏损失
                    kd_loss = self._compute_distillation_loss(logits, gen_logits)
                    
                    # 随机生成样本
                    sampled_labels = np.random.choice(
                        self.num_classes, self.generator.train_batch_size
                    )
                    sampled_labels = torch.LongTensor(sampled_labels).to(self.device)
                    with torch.no_grad():  
                        eps,sampled_features = self.generator(sampled_labels)
                    sampled_logits = self.model(sampled_features,start_layer="classify")
                    # 计算生成样本的分类损失
                    teacher_loss = self.loss(sampled_logits, sampled_labels)
                    
                    # 总损失
                    loss = ce_loss + alpha * kd_loss + beta * teacher_loss
                    
                    # 反向传播和优化
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_teacher_loss += teacher_loss.item()
                    epoch_kd_loss += kd_loss.item()
                    pbar.update(1)
                
                avg_loss = epoch_loss / (len(self.train_loader))
                total_loss += epoch_loss
                avg_teacher_loss = epoch_teacher_loss / (len(self.train_loader) )
                avg_kd_loss = epoch_kd_loss / (len(self.train_loader))
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'teacher_loss': f"{avg_teacher_loss:.4f}",
                    'kd_loss': f"{avg_kd_loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
        self.scheduler.step()
        
        # 6. 获取更新后的模型权重
        model_weights = self.get_weights(return_numpy=True)
        
        # 7. 返回结果
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        return model_weights, num_sample, avg_loss,self.label_count
    
    
   
    