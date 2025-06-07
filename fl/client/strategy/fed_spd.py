# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/5/16 11:07
# @Describe: FedSPD - 基于特征对齐和互信息最大化的联邦知识蒸馏

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from fl.client.fl_base import BaseClient
from scipy.spatial import distance

class FedSPD(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 核心知识蒸馏参数
        self.temperature = kwargs.get('temperature', 0.07)     # 对比学习温度参数
        self.proto_reg_weight = kwargs.get('proto_reg_weight', 0.5)  # 原型正则化权重
        self.contrastive_weight = kwargs.get('contrastive_weight', 0.3)  # 对比损失权重
        self.instance_weight = kwargs.get('instance_weight', 0.2)  # 实例对齐权重
        
        # 特征蒸馏和抽取参数
        self.feature_dim = kwargs.get('feature_dim', 128)      # 特征维度
        self.queue_size = kwargs.get('queue_size', 128)        # 负样本队列大小
        self.momentum_update = kwargs.get('momentum_update', 0.99)  # 动量更新系数
        
        # 训练参数
        self.adaptive_lr = kwargs.get('adaptive_lr', True)     # 自适应学习率
        self.min_epochs = 1
        self.max_epochs = 10
        self.patience = 3
        self.loss_threshold = 0.01
        
        # 初始化队列和特征银行
        self.feature_queue = None
        self.class_features = {}
        
        # 标准损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        
        print(f"🚀 FedSPD客户端 {self.client_id} 初始化完成")
        
    def _setup_feature_extractor(self):
        """
        设置特征提取器 - 使用模型的中间层作为特征提取器
        
        注意：现在使用模型的原生方法，不再需要动态添加方法
        """
        # 检查模型是否支持特征提取
        if not hasattr(self.model, 'get_features') and not hasattr(self.model, 'forward'):
            print(f"⚠️ 警告: 模型 {type(self.model).__name__} 不支持特征提取，将使用输出作为特征")

    def _compute_contrastive_loss(self, features, labels, global_prototypes=None):
        """
        计算对比学习损失 - 基于InfoNCE
        
        理论基础:
        对比学习通过最大化正样本之间的互信息和最小化负样本之间的互信息来学习表示
        
        数学公式:
        L_con = -log[ exp(sim(f_i, p_i)/τ) / (exp(sim(f_i, p_i)/τ) + Σ_j≠i exp(sim(f_i, p_j)/τ)) ]
        
        其中:
        - f_i是样本i的特征表示
        - p_i是样本i所属类别的原型
        - τ是温度参数
        - sim是余弦相似度函数
        """
        if global_prototypes is None or not global_prototypes:
            return torch.tensor(0.0, device=self.device)
        
        # 确保特征为二维张量
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # L2归一化
        features = F.normalize(features, dim=1)
        
        # 收集所有可用的原型
        prototype_vectors = []
        proto_labels = []
        
        for label in labels.unique():
            label_val = label.item()
            if label_val in global_prototypes:
                proto = torch.tensor(global_prototypes[label_val], 
                                    device=self.device, 
                                    dtype=torch.float32)
                prototype_vectors.append(proto)
                proto_labels.append(label_val)
        
        if not prototype_vectors:
            return torch.tensor(0.0, device=self.device)
            
        # 堆叠原型
        prototypes = torch.stack(prototype_vectors)
        prototypes = F.normalize(prototypes, dim=1)
        proto_labels = torch.tensor(proto_labels, device=self.device)
        
        # 计算特征与所有原型的相似度
        similarity = torch.matmul(features, prototypes.t()) / self.temperature
        
        # 计算对比损失
        losses = []
        for i, label in enumerate(labels):
            label_val = label.item()
            
            # 检查该类别是否有原型
            if label_val not in global_prototypes:
                continue
                
            # 找到正样本原型的索引
            positive_idx = (proto_labels == label_val).nonzero(as_tuple=True)[0]
            
            if len(positive_idx) == 0:
                continue
                
            # 计算对比损失
            logits = similarity[i]
            exp_logits = torch.exp(logits)
            pos_exp_logits = exp_logits[positive_idx]
            
            # InfoNCE公式
            loss_i = -torch.log(
                pos_exp_logits / (torch.sum(exp_logits) + 1e-8)
            )
            losses.append(loss_i.mean())
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=self.device)

    def _compute_prototype_reg_loss(self, features, labels, global_prototypes=None, global_covariances=None):
        """
        计算原型正则化损失 - 基于Wasserstein距离
        
        理论基础:
        鼓励局部特征与全局原型对齐，同时考虑协方差结构
        
        数学公式:
        L_proto = Σ_c W₂(μ_c^l, μ_c^g)
        
        其中:
        - μ_c^l是局部类别原型
        - μ_c^g是全局类别原型
        - W₂是简化的Wasserstein距离
        """
        if global_prototypes is None or not global_prototypes:
            return torch.tensor(0.0, device=self.device)
        
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # 计算类别原型
        losses = []
        for c in labels.unique():
            c_val = c.item()
            
            # 检查该类别是否有全局原型
            if c_val not in global_prototypes:
                continue
                
            # 获取该类别的特征
            c_mask = (labels == c)
            if not c_mask.any():
                continue
                
            c_features = features[c_mask]
            
            # 计算局部原型
            local_proto = c_features.mean(dim=0)
            
            # 获取全局原型
            global_proto = torch.tensor(
                global_prototypes[c_val], 
                device=self.device,
                dtype=torch.float32
            )
            
            # 计算原型之间的距离 (Wasserstein距离简化版)
            proto_dist = F.mse_loss(local_proto, global_proto)
            
            # 如果有协方差信息，考虑协方差对齐
            if global_covariances is not None and c_val in global_covariances:
                # 计算局部协方差
                if c_features.size(0) > 1:
                    c_centered = c_features - local_proto.unsqueeze(0)
                    local_cov = torch.matmul(c_centered.t(), c_centered) / (c_features.size(0) - 1)
                    
                    # 获取全局协方差
                    global_cov = torch.tensor(
                        global_covariances[c_val],
                        device=self.device,
                        dtype=torch.float32
                    )
                    
                    # Frobenius范数作为协方差差异
                    cov_dist = torch.norm(local_cov - global_cov, p='fro')
                    
                    # 综合原型和协方差距离
                    total_dist = proto_dist + 0.1 * cov_dist
                else:
                    total_dist = proto_dist
            else:
                total_dist = proto_dist
                
            losses.append(total_dist)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=self.device)

    def _compute_instance_alignment_loss(self, features, labels, global_prototypes=None):
        """
        计算实例对齐损失 - 基于软分配
        
        理论基础:
        通过软分配机制将局部实例与全局原型对齐
        
        数学公式:
        L_inst = -Σ_i Σ_c q_ic log p_ic
        
        其中:
        - q_ic是样本i对原型c的软分配
        - p_ic是模型预测的样本i对类别c的概率
        """
        if global_prototypes is None or not global_prototypes:
            return torch.tensor(0.0, device=self.device)
        
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # 收集所有原型
        prototypes = []
        proto_classes = []
        
        for c, proto in global_prototypes.items():
            prototypes.append(torch.tensor(proto, device=self.device, dtype=torch.float32))
            proto_classes.append(c)
            
        if not prototypes:
            return torch.tensor(0.0, device=self.device)
            
        # 堆叠所有原型
        prototypes = torch.stack(prototypes)
        proto_classes = torch.tensor(proto_classes, device=self.device)
        
        # L2归一化
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        
        # 计算每个样本与所有原型的余弦相似度
        similarity = torch.matmul(features, prototypes.t())
        
        # 使用softmax生成软分配
        soft_assign = F.softmax(similarity / self.temperature, dim=1)
        
        # 创建标签的one-hot编码
        num_classes = len(proto_classes)
        one_hot = torch.zeros(labels.size(0), num_classes, device=self.device)
        
        # 将标签映射到原型索引
        label_to_idx = {proto_classes[i].item(): i for i in range(len(proto_classes))}
        
        # 填充one-hot编码
        for i, label in enumerate(labels):
            label_val = label.item()
            if label_val in label_to_idx:
                one_hot[i, label_to_idx[label_val]] = 1.0
        
        # 计算交叉熵损失
        instance_loss = -torch.sum(one_hot * torch.log(soft_assign + 1e-8)) / labels.size(0)
        
        return instance_loss

    def _collect_features(self, features, labels):
        """收集特征用于后续分析"""
        # 确保特征是2D
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
            
        # 按类别收集特征
        for i, label in enumerate(labels):
            y = label.item()
            if y not in self.class_features:
                self.class_features[y] = []
            self.class_features[y].append(features[i].detach().cpu().numpy())

    def _adaptive_epochs(self, sync_round):
        """确定自适应训练轮数"""
        if sync_round <= 5:
            return min(self.max_epochs, max(self.min_epochs, 3))
        elif sync_round <= 15:
            return min(self.max_epochs, max(self.min_epochs, 2))
        else:
            return max(self.min_epochs, 1)

    def local_train(self, sync_round: int, weights=None, global_prototypes=None, global_covariances=None):
        """
        FedSPD训练 - 基于特征对齐和互信息最大化的联邦知识蒸馏
        
        核心步骤:
        1. 初始化特征提取器
        2. 更新模型权重
        3. 确定自适应训练轮数
        4. 进行特征对齐训练
        5. 收集特征用于后续聚合
        
        返回:
        - 模型权重
        - 样本数量
        - 训练损失
        - 类特征字典
        - 类标签
        """
        # 1. 确保特征提取器设置完成
        self._setup_feature_extractor()
        
        # 2. 更新模型权重
        if weights is not None:
            self.update_weights(weights)
        
        # 3. 自适应确定训练轮数
        adaptive_epochs = self.epochs
        
        
        
        # 5. 清空特征收集
        self.class_features = {}
        
        # 6. 开始训练
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_contrastive_loss = 0
        total_proto_loss = 0
        total_instance_loss = 0
        num_sample = len(self.train_loader.dataset)
        
        # 早停机制
        prev_loss = float('inf')
        patience_counter = 0
        
        print(f"📊 客户端 {self.client_id}: 自适应轮数={adaptive_epochs}")
        
        with tqdm(
            total=adaptive_epochs * len(self.train_loader),
            desc=f"Client {self.client_id} FedSPD Training"
        ) as pbar:
            
            for epoch in range(adaptive_epochs):
                epoch_loss = 0
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # 前向传播 - 获取特征和输出
                    if hasattr(self.model, 'forward') and 'return_all' in self.model.forward.__code__.co_varnames:
                        # 如果模型支持直接返回所有特征
                        hidden, features, output = self.model(data, return_all=True)
                    elif hasattr(self.model, 'get_features'):
                        # 使用get_features方法提取特征，然后获取输出
                        features = self.model.get_features(data)
                        if hasattr(self.model, 'get_hidden') and hasattr(self.model, 'classify'):
                            hidden = self.model.get_hidden(features)
                            output = self.model.classify(hidden)
                        else:
                            # 如果没有明确的hidden层，直接使用features
                            hidden = features
                            output = self.model(data)
                    else:
                        # 回退到标准前向传播
                        output = self.model(data)
                        features = output  # 回退
                        hidden = features
                    
                    # 1. 标准交叉熵损失
                    ce_loss = self.ce_loss(output, target)
                    
                    # 2. 特征对齐损失
                    contrastive_loss = self._compute_contrastive_loss(
                        features, target, global_prototypes
                    )
                    
                    # 3. 原型正则化损失
                    proto_loss = self._compute_prototype_reg_loss(
                        features, target, global_prototypes, global_covariances
                    )
                    
                    # 4. 实例对齐损失
                    instance_loss = self._compute_instance_alignment_loss(
                        features, target, global_prototypes
                    )
                    
                    # 5. 总损失
                    total_batch_loss = (
                        ce_loss + 
                        self.contrastive_weight * contrastive_loss +
                        self.proto_reg_weight * proto_loss +
                        self.instance_weight * instance_loss
                    )
                    
                    # 反向传播
                    total_batch_loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    # 收集特征（最后一个epoch）
                    if epoch == adaptive_epochs - 1:
                        self._collect_features(features.detach(), target)
                    
                    # 记录损失
                    epoch_loss += total_batch_loss.item()
                    total_ce_loss += ce_loss.item()
                    total_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0
                    total_proto_loss += proto_loss.item() if isinstance(proto_loss, torch.Tensor) else 0
                    total_instance_loss += instance_loss.item() if isinstance(instance_loss, torch.Tensor) else 0
                    
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.update(1)
                    pbar.set_postfix({
                        'epoch': f"{epoch+1}/{adaptive_epochs}",
                        'ce': f"{ce_loss.item():.4f}",
                        'con': f"{contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0:.4f}",
                        'proto': f"{proto_loss.item() if isinstance(proto_loss, torch.Tensor) else 0:.4f}",
                        'lr': f"{current_lr:.6f}"
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
        self.scheduler.step()
        # 7. 计算平均损失
        num_batches = adaptive_epochs * len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_proto_loss = total_proto_loss / num_batches
        avg_instance_loss = total_instance_loss / num_batches
        
        print(f"✅ 客户端 {self.client_id} 训练完成: 总损失={avg_loss:.4f}, CE={avg_ce_loss:.4f}, " +
              f"对比={avg_contrastive_loss:.4f}, 原型={avg_proto_loss:.4f}, 实例={avg_instance_loss:.4f}")
        
        # 8. 返回结果
        model_weights = self.get_weights(return_numpy=True)
        
        # 返回类别标签
        class_labels = list(self.class_features.keys())
        
        return model_weights, num_sample, avg_loss, self.class_features, class_labels