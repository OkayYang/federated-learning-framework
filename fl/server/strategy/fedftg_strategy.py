# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/19 16:30
# @Describe: FedFTG聚合策略实现

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from fl.server.strategy.strategy_base import AggregationStrategy
from fl.aggregation.aggregator import average_weight


class FedFTGStrategy(AggregationStrategy):
    """
    FedFTG聚合策略 - Fine-Tuning Global Model via Data-Free Knowledge Distillation
    
    参考论文: "Fine-Tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning"
    CVPR 2022, Zhang et al.
    
    FedFTG核心思想:
    1. 在服务器端使用生成器生成"困难样本"来蒸馏本地模型知识
    2. 使用硬样本挖掘(hard sample mining)策略有效探索输入空间
    3. 自定义标签采样和类级别集成来适应标签分布偏移
    """
    
    def __init__(self):
        """初始化FedFTG聚合策略"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.generator_optimizer = None
        
        # 超参数设置
        self.lambda_cls = 1.0       # 分类损失权重
        self.lambda_diversity = 1.0  # 多样性损失权重
        self.temperature = 1.0      # 温度系数
        
        # 训练参数
        self.generator_iter = 1     # 生成器内部迭代次数
        self.distill_iter = 5       # 蒸馏内部迭代次数
        self.server_epochs = 10     # 服务器端训练轮数
        self.batch_size = 32        # 生成数据的批次大小，默认较小值以避免内存问题
        self.z_dim = 100            # 随机噪声维度
        
    def initialize(self, server_kwargs):
        """
        从服务器参数初始化FedFTG特定数据
        
        Args:
            server_kwargs: 服务器初始化时传入的参数，必须包含generator和num_classes
        """
        # 初始化生成器
        self.generator = server_kwargs.get('generator')
        if self.generator is None:
            raise ValueError("FedFTG需要一个生成器模型，请在server_kwargs中提供'generator'参数")
        
        # 将生成器移到设备上
        self.generator = self.generator.to(self.device)
        
        # 初始化生成器优化器
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), 
            lr=server_kwargs.get('generator_lr', 0.001),
            betas=(0.5, 0.999)
        )
        
        # 获取类别数量
        self.num_classes = server_kwargs.get('num_classes')
        if self.num_classes is None:
            raise ValueError("FedFTG需要知道类别数量，请在server_kwargs中提供'num_classes'参数")
        
        # 设置超参数
        self.lambda_cls = server_kwargs.get('lambda_cls', 1.0)
        self.lambda_diversity = server_kwargs.get('lambda_diversity', 1.0)
        self.temperature = server_kwargs.get('temperature', 1.0)
        self.z_dim = server_kwargs.get('z_dim', 100)
        self.generator_iter = server_kwargs.get('generator_iter', 1)
        self.distill_iter = server_kwargs.get('distill_iter', 5)
        self.server_epochs = server_kwargs.get('server_epochs', 10)
        self.batch_size = server_kwargs.get('batch_size', 32)
        
        # 训练模式
        self.generator.train()
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """
        执行FedFTG聚合策略
        
        Args:
            server: 服务器实例
            selected_workers: 选中的工作节点
            round_num: 当前轮次
            global_weights: 全局模型权重
            
        Returns:
            tuple: (更新后的全局模型权重, 训练损失列表)
        """
        if not selected_workers:
            return global_weights, []
        
        print(f"\n=== 第{round_num}轮 FedFTG聚合 (客户端数: {len(selected_workers)}) ===")
        
        # 1. 收集客户端数据
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        label_count_list = []
        client_model_list = []
        
        futures_train = []
        futures_model = []
        
        generator_weights = self.generator.get_weights(return_numpy=True)
        
        for client_name, worker in selected_workers.items():
            # 使用自定义的客户端更新方法
            futures_train.append(worker.local_train.remote(
                sync_round=round_num,
                weights=global_weights,
                generator_weights=generator_weights
            ))
            # Get model copy after training
            futures_model.append(worker.get_model_copy.remote())
            
        results_train = ray.get(futures_train)
        results_model = ray.get(futures_model)
        
        for i, client_name in enumerate(selected_workers.keys()):
            client_weight, sample_num, train_loss, label_count = results_train[i]
            client_model_copy = results_model[i]
            
            # 处理结果
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            train_loss_list.append(train_loss)
            label_count_list.append(label_count)
            client_model_list.append(client_model_copy)
            
            server.history["workers"][client_name]["train_loss"].append(train_loss)
            print(f"   - 客户端 {client_name}: 损失={train_loss:.4f}, 样本数={sample_num}")
        
        # 2. 加权平均聚合全局模型 (FedAvg方式)
        print("   - 执行全局模型聚合...")
        global_weights = average_weight(client_weight_list, sample_num_list)
        
        # 3. 计算标签采样分布
        label_distribution = self._compute_label_distribution(label_count_list)
        
        # 4. 使用数据无关知识蒸馏微调全局模型
        global_weights = self._finetune_global_model(
            global_weights,
            client_model_list,
            label_count_list,
            label_distribution,
            round_num
        )
        
        # 5. 返回微调后的全局模型和训练损失
        avg_loss = np.mean(train_loss_list)
        print(f"   - 平均训练损失: {avg_loss:.4f}")
        
        return global_weights, train_loss_list
    
    def _compute_label_distribution(self, label_counts):
        """
        计算自定义标签采样分布 (对应论文中的公式9)
        
        Args:
            label_counts: 客户端标签计数列表
            
        Returns:
            numpy.ndarray: 标签采样概率分布
        """
        # 初始化计数数组
        class_counts = np.zeros(self.num_classes)
        
        # 累加每个客户端的标签计数
        for client_count in label_counts:
            for label, count in client_count.items():
                if 0 <= label < self.num_classes:
                    class_counts[label] += count
        
        # 计算采样概率 (防止除零)
        total_count = np.sum(class_counts)
        if total_count > 0:
            sampling_probs = class_counts / total_count
        else:
            # 如果没有样本，使用均匀分布
            sampling_probs = np.ones(self.num_classes) / self.num_classes
        
        return sampling_probs
    
    def _compute_class_level_weights(self, label_counts, label):
        """
        计算类级别集成权重 (对应论文中的公式10)
        
        Args:
            label_counts: 客户端标签计数列表
            label: 当前类别
            
        Returns:
            numpy.ndarray: 类级别集成权重
        """
        # 提取每个客户端该类别的样本数
        client_counts = np.zeros(len(label_counts))
        for i, client_count in enumerate(label_counts):
            client_counts[i] = client_count.get(label, 0)
        
        # 计算权重 (防止除零)
        total_count = np.sum(client_counts)
        if total_count > 0:
            weights = client_counts / total_count
        else:
            # 如果该类别没有样本，使用均匀权重
            weights = np.ones(len(label_counts)) / len(label_counts)
        
        return weights
    
    def _finetune_global_model(self, global_weights, client_models, label_counts, label_distribution, round_num):
        """
        使用数据无关知识蒸馏微调全局模型 (完整实现对应论文算法2)
        
        Args:
            global_weights: 聚合后的全局模型权重 (FedAvg结果)
            client_models: 客户端模型列表
            label_counts: 客户端标签计数列表
            label_distribution: 标签采样分布 (公式9)
            round_num: 当前轮次
            
        Returns:
            list: 微调后的全局模型权重
        """
        # 如果轮次过低或epochs设为0，跳过微调
        if round_num <= 0 or self.server_epochs <= 0:
            print(f"   - 跳过知识蒸馏微调 (轮次 {round_num}, 服务器轮数 {self.server_epochs})")
            return global_weights
            
        try:
            # 创建全局模型副本并加载权重
            print(f"   - 开始FedFTG知识蒸馏微调 ({self.server_epochs}轮)...")
            
            # 按照算法2第1步：FedAvg聚合已在aggregate方法中完成
            
            # 创建全局模型副本
            global_model = copy.deepcopy(client_models[0])
            global_model_state_dict = global_model.state_dict()
            
            # 加载全局权重
            for i, (key, _) in enumerate(global_model_state_dict.items()):
                if i < len(global_weights):
                    global_model_state_dict[key] = torch.tensor(global_weights[i])
            
            global_model.load_state_dict(global_model_state_dict)
            global_model = global_model.to(self.device)
            
            # 设置优化器
            g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
            d_optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001)
            
            # 设置为训练模式
            global_model.train()
            self.generator.train()
            
            # 按照算法2第3步：进行I轮服务器端训练
            for epoch in range(self.server_epochs):
                # 算法2第3.1步：采样噪声和标签
                z = torch.randn(self.batch_size, self.z_dim).to(self.device)
                
                # 使用概率分布采样标签
                labels_np = np.random.choice(
                    self.num_classes,
                    size=self.batch_size,
                    p=label_distribution
                )
                labels = torch.LongTensor(labels_np).to(self.device)
                
                # 算法2第3.2步：计算类级别权重
                class_weights = {}
                for label in np.unique(labels_np):
                    class_weights[label] = self._compute_class_level_weights(label_counts, label)
                
                # 算法2第3.3步：训练生成器I_g轮 - 挖掘难样本
                g_loss_val = 0
                for j in range(self.generator_iter):
                    # 生成合成样本
                    synthetic_data = self.generator(z, labels)
                    
                    # 计算多样性损失 (公式7)
                    diversity_loss = self._compute_diversity_loss(synthetic_data)
                    
                    # 计算分类损失
                    classification_loss = 0
                    for i, client_model in enumerate(client_models):
                        client_model = client_model.to(self.device)
                        client_model.eval()
                        
                        # 计算每个样本的权重
                        sample_weights = torch.zeros(self.batch_size, 1).to(self.device)
                        for j, label in enumerate(labels_np):
                            if label in class_weights:
                                sample_weights[j, 0] = class_weights[label][i]
                        
                        # 客户端模型前向传播
                        with torch.no_grad():
                            client_logits = client_model(synthetic_data)
                        
                        # 计算交叉熵损失
                        ce_loss = F.cross_entropy(client_logits, labels, reduction='none')
                        classification_loss += torch.mean(ce_loss.view(-1, 1) * sample_weights)
                    
                    # 计算KL散度损失 (最大化客户端和全局模型的差异)
                    kl_loss = 0
                    for client_model in client_models:
                        client_model = client_model.to(self.device)
                        client_model.eval()
                        
                        with torch.no_grad():
                            client_logits = client_model(synthetic_data)
                        
                        global_logits = global_model(synthetic_data)
                        
                        # 计算KL散度
                        client_probs = F.softmax(client_logits / self.temperature, dim=1)
                        global_log_probs = F.log_softmax(global_logits / self.temperature, dim=1)
                        kl = F.kl_div(global_log_probs, client_probs, reduction='batchmean') * (self.temperature ** 2)
                        
                        # 最大化KL散度 (负号)
                        kl_loss -= kl
                    
                    # 总生成器损失 (公式8)
                    g_loss = kl_loss - self.lambda_cls * classification_loss - self.lambda_diversity * diversity_loss
                    
                    # 反向传播和优化
                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()
                    
                    g_loss_val += g_loss.item()
                
                # 算法2第3.4步：训练全局模型I_d轮 - 知识蒸馏
                d_loss_val = 0
                for j in range(self.distill_iter):
                    # 使用当前生成器生成合成样本
                    with torch.no_grad():
                        synthetic_data = self.generator(z, labels)
                    
                    # 计算知识蒸馏损失
                    distill_loss = 0
                    for i, client_model in enumerate(client_models):
                        client_model = client_model.to(self.device)
                        client_model.eval()
                        
                        # 计算每个样本的权重
                        sample_weights = torch.zeros(self.batch_size, self.num_classes).to(self.device)
                        for j, label in enumerate(labels_np):
                            if label in class_weights:
                                weight = class_weights[label][i]
                                sample_weights[j, :] = weight
                        
                        # 客户端模型前向传播
                        with torch.no_grad():
                            client_logits = client_model(synthetic_data)
                        
                        # 全局模型前向传播
                        global_logits = global_model(synthetic_data)
                        
                        # 计算KL散度
                        client_probs = F.softmax(client_logits / self.temperature, dim=1)
                        global_log_probs = F.log_softmax(global_logits / self.temperature, dim=1)
                        
                        # 加权KL散度
                        kl = F.kl_div(
                            global_log_probs, 
                            client_probs, 
                            reduction='none'
                        ) * (self.temperature ** 2)
                        
                        # 应用样本权重
                        weighted_kl = kl * sample_weights.unsqueeze(1)
                        
                        # 求和
                        distill_loss += weighted_kl.sum() / self.batch_size
                    
                    # 反向传播和优化
                    d_optimizer.zero_grad()
                    distill_loss.backward()
                    d_optimizer.step()
                    
                    d_loss_val += distill_loss.item()
                
                # 每轮的平均损失
                avg_g_loss = g_loss_val / self.generator_iter if self.generator_iter > 0 else 0
                avg_d_loss = d_loss_val / self.distill_iter if self.distill_iter > 0 else 0
                
                if (epoch + 1) % 2 == 0:
                    print(f"      轮次 {epoch+1}/{self.server_epochs}: "
                          f"G损失={avg_g_loss:.4f}, D损失={avg_d_loss:.4f}")
            
            print("   - 知识蒸馏微调完成")
            
            # 获取微调后的全局模型权重，并确保数据类型和形状正确
            updated_weights = []
            for param in global_model.state_dict().values():
                # 转换为NumPy数组，确保数据类型是标准的float32
                param_np = param.cpu().detach().numpy().astype(np.float32)
                
                # 检查数组是否包含无效值
                if not np.all(np.isfinite(param_np)):
                    print("   - 警告：检测到无效值 (NaN 或 Inf)，使用原始权重")
                    return global_weights
                
                updated_weights.append(param_np)
            
            # 验证权重格式一致性
            if len(updated_weights) != len(global_weights):
                print(f"   - 警告：微调后的权重数量 ({len(updated_weights)}) 与原始权重 ({len(global_weights)}) 不匹配，使用原始权重")
                return global_weights
            
            for i, (new_w, old_w) in enumerate(zip(updated_weights, global_weights)):
                if new_w.shape != old_w.shape:
                    print(f"   - 警告：第 {i} 个权重形状不匹配 ({new_w.shape} vs {old_w.shape})，使用原始权重")
                    return global_weights
                
                # 额外检查：确保数据类型匹配，防止负维度问题
                if new_w.dtype != old_w.dtype:
                    print(f"   - 警告：第 {i} 个权重数据类型不匹配 ({new_w.dtype} vs {old_w.dtype})，进行转换")
                    new_w = new_w.astype(old_w.dtype)
                    updated_weights[i] = new_w
            
            return updated_weights
            
        except Exception as e:
            print(f"   - 微调过程中出错: {str(e)}")
            print(f"   - 使用FedAvg结果作为后备方案")
            import traceback
            traceback.print_exc()
            return global_weights
    
    def _compute_diversity_loss(self, synthetic_data):
        """
        计算多样性损失 (对应论文公式7)
        
        Args:
            synthetic_data: 生成的合成数据, 形状为 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 多样性损失
        """
        batch_size = synthetic_data.size(0)
        
        # 计算特征多样性
        # 将图像展平为向量
        flattened = synthetic_data.view(batch_size, -1)
        
        # 计算样本对之间的余弦相似度
        normalized = F.normalize(flattened, p=2, dim=1)
        cosine_sim = torch.mm(normalized, normalized.t())
        
        # 移除对角线上的自相似度
        mask = torch.ones_like(cosine_sim) - torch.eye(batch_size, device=synthetic_data.device)
        cosine_sim = cosine_sim * mask
        
        # 计算损失 (最小化相似度，即最大化多样性)
        diversity_loss = cosine_sim.sum() / (batch_size * (batch_size - 1))
        
        return diversity_loss 