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
    
    FedGen核心思想:
    1. 使用服务器端的生成器模型来学习全局数据分布
    2. 利用客户端上传的logits训练生成器，实现知识转移
    3. 使用生成器生成的合成样本，在客户端通过知识蒸馏提高泛化能力
    """
    
    """FedGen聚合策略"""
    
    def initialize(self, server_kwargs):
        """初始化生成器模型"""
        generator_model = server_kwargs.get('generator_model')
        if generator_model is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.global_generator = generator_model.to(self.device)
        else:
            raise ValueError("生成器模型未初始化")
    
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """重写聚合方法，处理生成器训练"""
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        label_count_list = []
        client_model_list = []
        
        for client_name, worker in selected_workers.items():
            # 使用自定义的客户端更新方法
            generator_weights = self.global_generator.get_weights(return_numpy=True)
            client_weight, sample_num, train_loss, label_count  =  worker.local_train(
                sync_round=round_num,
                weights=global_weights,
                generator_weights=generator_weights
            )
            
            # 处理结果
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            train_loss_list.append(train_loss)
            label_count_list.append(label_count)
            client_model_list.append(worker.get_model_copy())
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 聚合模型权重
        global_weight = average_weight(client_weight_list, sample_num_list)
        
        # 训练生成器
        self._train_generator(
            client_model_list, 
            label_count_list, 
            global_weight, 
            round_num
        )
        
        return global_weight, train_loss_list
    
    def _train_generator(self, client_model_list, label_count_list, global_weight, round_num):
        """训练生成器模型"""
        
        # 计算标签权重
        label_weights = []
        for label in range(self.global_generator.num_classes):
            weights = [label_count.get(label, 0) for label_count in label_count_list]
            label_sum = np.sum(weights) + 1e-6
            label_weights.append(np.array(weights) / label_sum)
        
        # 生成器训练
        self.global_generator.train()
        total_loss = 0
        
        for i in range(self.global_generator.train_epochs):
            self.global_generator.optimizer.zero_grad()
            
            # 随机生成标签
            sampled_labels = np.random.choice(
                self.global_generator.num_classes, self.global_generator.train_batch_size
            )
            sampled_labels = torch.LongTensor(sampled_labels).to(self.device)
            
            # 生成随机噪声
            
            # 生成合成数据
            eps,synthetic_features = self.global_generator(sampled_labels)
            diversity_loss = self.global_generator.diversity_loss(eps, synthetic_features)
            
            # 教师损失
            teacher_loss = 0
            teacher_logit = 0
            
            for idx in range(len(client_model_list)):
                # 计算每个标签的权重
                sampled_labels_np = sampled_labels.cpu().numpy()
                batch_weights = np.zeros((len(sampled_labels_np), 1))
                
                for i, label in enumerate(sampled_labels_np):
                    batch_weights[i, 0] = label_weights[label][idx]
                
                weight = torch.tensor(batch_weights, dtype=torch.float32).to(self.device)
                expand_weight = np.tile(weight.cpu().numpy(), (1, self.global_generator.num_classes))
                
                with torch.no_grad():
                    teacher_logits = client_model_list[idx](synthetic_features, start_layer="classify")
                
                teacher_loss_ = torch.mean(
                    self.global_generator.loss_fn(teacher_logits, sampled_labels) * weight
                )
                teacher_loss += teacher_loss_
                teacher_logit += teacher_logits * torch.tensor(
                    expand_weight, dtype=torch.float32
                ).to(self.device)
            
            # 学生损失
            global_model = copy.deepcopy(client_model_list[0])
            keys = global_model.state_dict().keys()
            weights_dict = {}
            
            for k, v in zip(keys, global_weight):
                weights_dict[k] = torch.Tensor(np.copy(v)).to(self.device)
                
            global_model.load_state_dict(weights_dict)
            global_model.eval()
            
            with torch.no_grad():
                student_output = global_model(synthetic_features, start_layer="classify").clone().detach()
            
            student_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_output, dim=1),
                torch.nn.functional.softmax(teacher_logit, dim=1),
                reduction='batchmean'
            )
            
            # 总损失
            if self.global_generator.ensemble_beta > 0:
                loss = (
                    self.global_generator.ensemble_alpha * teacher_loss
                    - self.global_generator.ensemble_beta * student_loss
                    + self.global_generator.ensemble_eta * diversity_loss
                )
            else:
                loss = (
                    self.global_generator.ensemble_alpha * teacher_loss
                    + self.global_generator.ensemble_eta * diversity_loss
                )
            
            loss.backward()
            self.global_generator.optimizer.step()
            total_loss += loss.item()
        
        print(f"FedGen 第{round_num}轮: 生成器模型训练, 总损失: {total_loss:.4f}")