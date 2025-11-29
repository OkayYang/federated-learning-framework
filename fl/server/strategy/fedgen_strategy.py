# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/10 11:07
# @Describe: FedGen聚合策略实现

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from fl.server.strategy.strategy_base import AggregationStrategy
from fl.aggregation.aggregator import average_weight
from fl.model.fedgen_generator import FedGenGenerator

class FedGenStrategy(AggregationStrategy):
    """
    FedGen聚合策略 - Data-Free Knowledge Distillation for Heterogeneous Federated Learning
    
    参考论文: "Data-Free Knowledge Distillation for Heterogeneous Federated Learning"
    ICML 2021, Zhu et al.
    
    论文链接: https://proceedings.mlr.press/v139/zhu21b.html
    原始实现: https://github.com/zhuangdizhu/FedGen
    
    FedGen核心思想:
    1. 使用服务器端的生成器模型来学习全局数据分布
    2. 利用客户端上传的模型和标签信息训练生成器
    3. 使用生成器生成的合成样本，在客户端通过知识蒸馏提高泛化能力
    """
    
    def __init__(self):
        """初始化FedGen聚合策略"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_generator = None
        self.dataset_name = "cifar10"  # 默认数据集
    
    def initialize(self, server_kwargs):
        """初始化生成器模型"""
        self.dataset_name = server_kwargs.get('dataset', 'cifar10').lower()
        
        generator_model = server_kwargs.get('generator_model')
        if generator_model is not None:
            self.global_generator = generator_model.to(self.device)
        else:
            # 如果没有提供生成器模型，检查是否提供了必要参数来创建一个
            feature_dim = server_kwargs.get('feature_dim')
            num_classes = server_kwargs.get('num_classes')
            if feature_dim is None or num_classes is None:
                raise ValueError("需要提供feature_dim和num_classes参数，或直接提供generator_model")
            
            self.global_generator = FedGenGenerator(
                feature_dim=feature_dim,
                num_classes=num_classes,
                dataset_name=self.dataset_name
            ).to(self.device)
        
        print(f"FedGen生成器初始化完成: feature_dim={self.global_generator.feature_dim}, "
              f"num_classes={self.global_generator.num_classes}, dataset={self.dataset_name}")
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """
        重写聚合方法，处理生成器训练
        
        Args:
            server: 服务器实例
            selected_workers: 选定的客户端
            round_num: 当前通信轮次
            global_weights: 全局模型权重
            
        Returns:
            tuple: (更新后的全局权重, 训练损失列表)
        """
        if not selected_workers:
            return global_weights, []
            
        print(f"\n=== 第{round_num}轮 FedGen聚合 (客户端数: {len(selected_workers)}) ===")
        
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        label_count_list = []
        client_model_list = []
        
        import ray
        
        futures_train = []
        futures_model = []
        
        generator_weights = self.global_generator.get_weights(return_numpy=True)
        
        for client_name, worker in selected_workers.items():
            # 使用自定义的客户端更新方法
            futures_train.append(worker.local_train.remote(
                sync_round=round_num,
                weights=global_weights,
                generator_weights=generator_weights
            ))
            # Get model copy after training (queued on actor)
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
        
        # 聚合模型权重
        print("   - 执行全局模型聚合...")
        global_weight = average_weight(client_weight_list, sample_num_list)
        
        # 训练生成器
        self._train_generator(
            client_model_list, 
            label_count_list, 
            global_weight, 
            round_num
        )
        
        # 返回更新后的全局模型和训练损失
        avg_loss = np.mean(train_loss_list)
        print(f"   - 平均训练损失: {avg_loss:.4f}")
        
        return global_weight, train_loss_list
    
    def _train_generator(self, client_model_list, label_count_list, global_weight, round_num):
        """
        训练生成器模型
        
        Args:
            client_model_list: 客户端模型列表
            label_count_list: 客户端标签统计列表
            global_weight: 全局模型权重
            round_num: 当前通信轮次
        """
        print(f"   - 训练生成器模型 (轮次 {round_num}, {self.global_generator.train_epochs}个epochs)...")
        
        # 计算标签权重
        label_weights = []
        for label in range(self.global_generator.num_classes):
            weights = [label_count.get(label, 0) for label_count in label_count_list]
            label_sum = np.sum(weights) + 1e-6  # 避免除零
            label_weights.append(np.array(weights) / label_sum)
        
        # 生成器训练
        self.global_generator.train()
        total_loss = 0
        
        # 设置训练batch数
        num_batches = max(1, self.global_generator.num_classes // self.global_generator.train_batch_size * 10)
        
        for epoch in range(self.global_generator.train_epochs):
            epoch_loss = 0
            
            for _ in range(num_batches):
                self.global_generator.optimizer.zero_grad()
                
                # 随机生成标签
                sampled_labels = np.random.choice(
                    self.global_generator.num_classes, 
                    self.global_generator.train_batch_size
                )
                sampled_labels = torch.LongTensor(sampled_labels).to(self.device)
                
                # 生成合成数据
                eps, synthetic_features = self.global_generator(sampled_labels)
                diversity_loss = self.global_generator.diversity_loss(eps, synthetic_features)
                
                # 教师损失（从客户端模型中学习）
                teacher_loss = 0
                teacher_logits_weighted = 0
                
                for idx, client_model in enumerate(client_model_list):
                    # 计算每个标签的权重
                    sampled_labels_np = sampled_labels.cpu().numpy()
                    batch_weights = np.zeros((len(sampled_labels_np), 1))
                    
                    for i, label in enumerate(sampled_labels_np):
                        batch_weights[i, 0] = label_weights[label][idx]
                    
                    weight_tensor = torch.tensor(batch_weights, dtype=torch.float32).to(self.device)
                    expanded_weights = torch.tensor(
                        np.tile(weight_tensor.cpu().numpy(), (1, self.global_generator.num_classes)),
                        dtype=torch.float32
                    ).to(self.device)
                    
                    # 使用客户端模型生成教师输出
                    with torch.no_grad():
                        client_logits = client_model(synthetic_features, start_layer="classify")
                    
                    # 计算加权的教师损失
                    teacher_loss_i = torch.mean(
                        self.global_generator.loss_fn(client_logits, sampled_labels) * weight_tensor
                    )
                    teacher_loss += teacher_loss_i
                    
                    # 累积加权的教师logits
                    teacher_logits_weighted += client_logits * expanded_weights
                
                # 学生损失（全局模型作为学生）
                global_model = copy.deepcopy(client_model_list[0])
                global_model.load_state_dict({
                    k: torch.tensor(v, dtype=torch.float32).to(self.device)
                    for k, v in zip(global_model.state_dict().keys(), global_weight)
                })
                global_model.eval()
                
                with torch.no_grad():
                    student_logits = global_model(synthetic_features, start_layer="classify")
                
                # KL散度损失（学生模型向教师模型学习）
                student_loss = F.kl_div(
                    F.log_softmax(student_logits, dim=1),
                    F.softmax(teacher_logits_weighted, dim=1),
                    reduction='batchmean'
                )
                
                # 总损失
                if self.global_generator.ensemble_beta > 0:
                    # 使用对抗训练（最大化学生损失）
                    loss = (
                        self.global_generator.ensemble_alpha * teacher_loss
                        - self.global_generator.ensemble_beta * student_loss
                        + self.global_generator.ensemble_eta * diversity_loss
                    )
                else:
                    # 不使用对抗训练
                    loss = (
                        self.global_generator.ensemble_alpha * teacher_loss
                        + self.global_generator.ensemble_eta * diversity_loss
                    )
                
                # 反向传播和优化
                loss.backward()
                self.global_generator.optimizer.step()
                epoch_loss += loss.item()
            
            # 打印每个epoch的损失
            avg_epoch_loss = epoch_loss / num_batches
            if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == self.global_generator.train_epochs - 1:
                print(f"      Epoch {epoch+1}/{self.global_generator.train_epochs}: 生成器损失={avg_epoch_loss:.4f}")
            
            total_loss += avg_epoch_loss
        
        # 打印平均损失
        avg_loss = total_loss / self.global_generator.train_epochs
        print(f"   - 生成器训练完成，平均损失: {avg_loss:.4f}")