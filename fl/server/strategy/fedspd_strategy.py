# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/19 16:30
# @Describe: FedSPD聚合策略实现

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from fl.server.strategy.strategy_base import AggregationStrategy
from fl.aggregation.aggregator import average_weight

class FedSPDStrategy(AggregationStrategy):
    """FedSmart聚合策略 - 简单而强大的质量感知聚合"""
    def __init__(self):
        self.global_reps = None
        self.global_logits = None
        
        # 历史性能追踪
        self.client_performance_history = {}
        self.performance_window = 5  # 追踪最近5轮的性能
        
        # 聚合参数
        self.quality_weight_factor = 2.0    # 质量权重因子
        self.stability_bonus = 0.2          # 稳定性奖励
        self.min_client_weight = 0.1        # 最小客户端权重
    
    def _compute_client_quality_score(self, train_loss, sample_num, client_name, round_num):
        """计算客户端质量分数 - 简单有效"""
        
        # 1. 基础质量分数（基于损失和样本数）
        loss_score = max(0.1, 1.0 / (1.0 + train_loss))  # 损失越低分数越高
        sample_score = min(1.0, sample_num / 1000.0)     # 样本数归一化
        base_quality = 0.7 * loss_score + 0.3 * sample_score
        
        # 2. 历史性能奖励
        if client_name not in self.client_performance_history:
            self.client_performance_history[client_name] = []
        
        # 记录当前性能
        self.client_performance_history[client_name].append(train_loss)
        if len(self.client_performance_history[client_name]) > self.performance_window:
            self.client_performance_history[client_name] = \
                self.client_performance_history[client_name][-self.performance_window:]
        
        # 计算性能稳定性奖励
        history = self.client_performance_history[client_name]
        if len(history) >= 3:
            # 损失下降趋势和稳定性
            recent_losses = np.array(history[-3:])
            if len(recent_losses) > 1:
                trend = -np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]  # 斜率为负表示下降
                stability = 1.0 / (1.0 + np.std(recent_losses))
                performance_bonus = self.stability_bonus * (max(0, trend) + stability) / 2
            else:
                performance_bonus = 0
        else:
            performance_bonus = 0
        
        final_quality = base_quality + performance_bonus
        return min(final_quality, 2.0)  # 限制最大分数

    def _smart_weight_computation(self, client_qualities, sample_nums):
        """智能权重计算 - 质量感知但保证公平性"""
        
        # 1. 基于样本数的基础权重
        total_samples = sum(sample_nums)
        base_weights = np.array(sample_nums) / total_samples
        
        # 2. 基于质量的调整权重
        quality_scores = np.array(client_qualities)
        
        # 归一化质量分数
        quality_normalized = quality_scores / (np.sum(quality_scores) + 1e-8)
        
        # 3. 混合权重：样本权重 + 质量权重
        # 高质量客户端获得更多权重，但不会完全忽略样本数
        mixed_weights = (
            0.6 * base_weights +                                    # 样本数权重
            0.4 * quality_normalized                                # 质量权重
        )
        
        # 4. 确保最小权重（避免某些客户端被完全忽略）
        min_weight = self.min_client_weight / len(mixed_weights)
        mixed_weights = np.maximum(mixed_weights, min_weight)
        
        # 5. 重新归一化
        mixed_weights = mixed_weights / np.sum(mixed_weights)
        
        return mixed_weights

    def _aggregate_knowledge_simple(self, class_data_list, weights):
        """简单高效的知识聚合"""
        aggregated = {}
        
        # 收集所有类别的数据
        all_class_data = {}
        for client_idx, client_data in enumerate(class_data_list):
            client_weight = weights[client_idx]
            
            for class_id, data in client_data.items():
                if class_id not in all_class_data:
                    all_class_data[class_id] = []
                
                all_class_data[class_id].append((np.array(data), client_weight))
        
        # 加权聚合每个类别
        for class_id, data_and_weights in all_class_data.items():
            if data_and_weights:
                data_arrays = [d for d, _ in data_and_weights]
                data_weights = [w for _, w in data_and_weights]
                
                # 归一化权重
                data_weights = np.array(data_weights)
                data_weights = data_weights / (np.sum(data_weights) + 1e-8)
                
                # 加权平均
                aggregated[class_id] = np.average(data_arrays, axis=0, weights=data_weights)
        
        return aggregated

    def aggregate(self, server, selected_workers, round_num, global_weights):
        """FedSmart聚合 - 智能但简单"""
        if not selected_workers:
            return global_weights, []
        
        print(f"\n🧠 第{round_num}轮 FedSmart聚合 (客户端数: {len(selected_workers)})")
        
        # 1. 收集客户端数据
        client_weights_list = []
        client_qualities = []
        sample_nums = []
        train_losses = []
        class_reps_list = []
        class_logits_list = []
        client_names = []
        
        for client_name, worker in selected_workers.items():
            # 调用客户端训练
            client_weight, sample_num, train_loss, class_reps, class_logits = worker.local_train(
                sync_round=round_num,
                weights=global_weights,
                global_reps=self.global_reps,
                global_logits=self.global_logits
            )
            
            # 计算客户端质量分数
            quality_score = self._compute_client_quality_score(
                train_loss, sample_num, client_name, round_num
            )
            
            # 收集数据
            client_weights_list.append(client_weight)
            client_qualities.append(quality_score)
            sample_nums.append(sample_num)
            train_losses.append(train_loss)
            class_reps_list.append(class_reps)
            class_logits_list.append(class_logits)
            client_names.append(client_name)
            
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 2. 智能权重计算
        aggregation_weights = self._smart_weight_computation(client_qualities, sample_nums)
        
        # 3. 聚合模型权重
        final_model_weights = average_weight(client_weights_list, aggregation_weights)
        
        # 4. 聚合知识
        final_reps = self._aggregate_knowledge_simple(class_reps_list, aggregation_weights)
        final_logits = self._aggregate_knowledge_simple(class_logits_list, aggregation_weights)
        
        # 5. 更新全局状态
        self.global_reps = final_reps
        self.global_logits = final_logits
        
        # 6. 打印统计信息
        avg_loss = np.mean(train_losses)
        avg_quality = np.mean(client_qualities)
        
        print(f"📊 聚合统计:")
        print(f"   - 平均训练损失: {avg_loss:.4f}")
        print(f"   - 平均质量分数: {avg_quality:.4f}")
        print(f"   - 权重分布: {[f'{w:.3f}' for w in aggregation_weights]}")
        print(f"   - 全局知识: 表征({len(final_reps)}类), logits({len(final_logits)}类)")
        
        # 显示性能最好的客户端
        best_client_idx = np.argmax(client_qualities)
        best_client = client_names[best_client_idx]
        print(f"   - 🏆 最佳客户端: {best_client} (质量: {client_qualities[best_client_idx]:.3f})")
        
        return final_model_weights, train_losses
