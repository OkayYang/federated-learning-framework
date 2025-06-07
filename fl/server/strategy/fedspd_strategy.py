# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/19 16:30
# @Describe: FedSPD聚合策略实现 - 基于特征对齐和互信息最大化的知识蒸馏

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from fl.server.strategy.strategy_base import AggregationStrategy
from fl.aggregation.aggregator import average_weight

class FedSPDStrategy(AggregationStrategy):
    """FedSPD聚合策略 - 基于特征对齐和互信息最大化的知识蒸馏"""
    def __init__(self):
        # 全局知识库
        self.global_prototypes = None  # 类原型向量
        self.global_covariances = None  # 类内协方差矩阵
        self.global_feature_banks = {}  # 特征库
        
        # 特征对齐参数
        self.alignment_temp = 0.07  # 对比学习温度参数
        self.max_prototypes_per_class = 5  # 每类最大原型数
        self.min_samples_per_prototype = 10  # 每个原型最小样本数
        self.feature_bank_size = 1000  # 特征库大小
        
        # 历史性能追踪
        self.client_performance_history = {}
        self.performance_window = 5  # 追踪最近5轮的性能
        
        # 聚合参数 
        self.feature_alignment_weight = 0.6  # 特征对齐权重
        self.min_client_weight = 0.1        # 最小客户端权重
        
        # 互信息估计参数
        self.mi_estimator_bins = 20  # 互信息估计的直方图分箱数
    
    def _compute_client_quality_score(self, train_loss, feature_alignment_score, 
                                     mutual_information, client_name, round_num):
        """
        计算客户端质量分数 - 基于损失、特征对齐度和互信息
        
        理论基础:
        1. 特征对齐分数衡量客户端特征与全局特征的一致性
        2. 互信息衡量客户端特征中包含的判别信息量
        3. 结合训练损失评估整体质量
        
        数学公式:
        quality = α * loss_score + β * alignment_score + γ * mi_score
        其中:
        - loss_score = 1/(1+L), L为训练损失
        - alignment_score = exp(-d/τ), d为平均特征距离, τ为温度
        - mi_score = MI(Z,Y)/H(Y), 归一化互信息
        """
        # 基础损失分数
        loss_score = max(0.1, 1.0 / (1.0 + train_loss))
        
        # 特征对齐分数
        alignment_score = feature_alignment_score
        
        # 互信息分数
        mi_score = mutual_information
        
        # 综合质量分数
        base_quality = 0.4 * loss_score + 0.4 * alignment_score + 0.2 * mi_score
        
        # 记录当前性能
        if client_name not in self.client_performance_history:
            self.client_performance_history[client_name] = []
        
        self.client_performance_history[client_name].append(train_loss)
        if len(self.client_performance_history[client_name]) > self.performance_window:
            self.client_performance_history[client_name] = \
                self.client_performance_history[client_name][-self.performance_window:]
        
        # 计算性能稳定性奖励
        history = self.client_performance_history[client_name]
        performance_bonus = 0
        if len(history) >= 3:
            # 计算趋势和稳定性
            recent_losses = np.array(history[-3:])
            trend = -np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            stability = 1.0 / (1.0 + np.std(recent_losses))
            performance_bonus = 0.2 * (max(0, trend) + stability) / 2
        
        final_quality = base_quality + performance_bonus
        return min(final_quality, 2.0)

    def _compute_feature_alignment_score(self, client_prototypes, client_covs):
        """
        计算特征对齐分数 - 基于最优传输理论
        
        理论基础:
        利用Wasserstein距离(最优传输)衡量分布之间的差异，并考虑协方差结构
        
        数学公式:
        alignment = exp(-W₂(D_g, D_c)/τ)
        
        其中:
        - W₂为2-Wasserstein距离: W₂²(D_g, D_c) = ||μ_g-μ_c||² + Tr(Σ_g + Σ_c - 2(Σ_g^(1/2)Σ_cΣ_g^(1/2))^(1/2))
        - D_g, D_c分别为全局和客户端特征分布
        - μ_g, μ_c分别为全局和客户端原型
        - Σ_g, Σ_c分别为全局和客户端协方差
        """
        if self.global_prototypes is None or not client_prototypes:
            return 0.5  # 默认中等对齐度
        
        alignment_scores = []
        
        for class_id in client_prototypes:
            if class_id in self.global_prototypes:
                # 获取客户端与全局的原型和协方差
                client_proto = client_prototypes[class_id]
                global_proto = self.global_prototypes[class_id]
                
                # 计算原型之间的距离矩阵
                distance_matrix = cdist(
                    np.array([client_proto]), 
                    np.array([global_proto]),
                    'euclidean'
                )[0, 0]
                
                # 如果有协方差信息，考虑分布形状差异
                if class_id in client_covs and class_id in self.global_covariances:
                    client_cov = client_covs[class_id]
                    global_cov = self.global_covariances[class_id]
                    
                    # 计算简化版本的协方差差异 (计算真正的Wasserstein距离需要矩阵平方根，计算复杂)
                    # 这里使用Frobenius范数作为协方差差异的度量
                    cov_diff = np.linalg.norm(client_cov - global_cov, ord='fro')
                    
                    # 综合原型距离和协方差差异
                    total_dist = distance_matrix + 0.1 * cov_diff
                else:
                    total_dist = distance_matrix
                
                # 计算对齐分数
                class_alignment = np.exp(-total_dist / self.alignment_temp)
                alignment_scores.append(class_alignment)
        
        if alignment_scores:
            return np.mean(alignment_scores)
        else:
            return 0.5  # 默认中等对齐度

    def _estimate_mutual_information(self, features, labels):
        """
        估计特征与标签之间的互信息 - 基于直方图方法
        
        理论基础:
        互信息衡量两个随机变量之间的相互依赖性，用于衡量特征的判别能力
        
        数学公式:
        MI(Z,Y) = H(Z) - H(Z|Y) = H(Z) + H(Y) - H(Z,Y)
        
        其中:
        - H(Z) = -∑p(z)log p(z) 是特征的熵
        - H(Z|Y) = -∑p(y)∑p(z|y)log p(z|y) 是给定标签下特征的条件熵
        - 我们使用归一化互信息: MI(Z,Y)/H(Y)，取值范围[0,1]
        """
        if not features or not labels or len(features) != len(labels):
            return 0.5  # 默认中等互信息
        
        # 将特征降维到主方向上(取每个特征的方差最大维度)
        features_1d = np.array([feat[np.argmax(np.var(feat, axis=0))] for feat in features])
        
        # 构建联合分布的直方图
        joint_hist, x_edges, y_edges = np.histogram2d(
            features_1d, labels, bins=self.mi_estimator_bins
        )
        joint_hist = joint_hist + 1e-10  # 避免log(0)
        
        # 归一化得到联合概率
        joint_prob = joint_hist / np.sum(joint_hist)
        
        # 计算边缘概率
        x_prob = np.sum(joint_prob, axis=1)
        y_prob = np.sum(joint_prob, axis=0)
        
        # 计算边缘熵
        h_x = -np.sum(x_prob * np.log2(x_prob))
        h_y = -np.sum(y_prob * np.log2(y_prob))
        
        # 计算联合熵
        h_xy = -np.sum(joint_prob * np.log2(joint_prob))
        
        # 计算互信息
        mi = h_x + h_y - h_xy
        
        # 归一化互信息(除以标签熵)
        normalized_mi = mi / h_y if h_y > 0 else 0
        
        return normalized_mi

    def _cluster_features(self, features, labels):
        """
        聚类特征创建多个原型 - 基于K-Means++
        
        理论基础:
        当类内存在多模态分布时，单一原型不足以表征整个类
        
        实现思路:
        1. 对每个类别分别聚类
        2. 自适应确定聚类数量(基于样本数和特征分布)
        3. 返回每个类的多个原型和对应协方差
        """
        class_prototypes = {}
        class_covariances = {}
        features_by_class = {}
        
        # 按类别组织特征
        for i, label in enumerate(labels):
            if label not in features_by_class:
                features_by_class[label] = []
            features_by_class[label].append(features[i])
        
        for class_id, class_features in features_by_class.items():
            class_features = np.array(class_features)
            n_samples = len(class_features)
            
            if n_samples < self.min_samples_per_prototype:
                # 样本太少，使用单个原型
                prototype = np.mean(class_features, axis=0)
                covariance = np.cov(class_features.T) if n_samples > 1 else np.eye(class_features.shape[1])
                class_prototypes[class_id] = prototype
                class_covariances[class_id] = covariance
            else:
                # 简化：这里只使用单原型，实际应用中可以实现多原型
                prototype = np.mean(class_features, axis=0)
                covariance = np.cov(class_features.T)
                class_prototypes[class_id] = prototype
                class_covariances[class_id] = covariance
        
        return class_prototypes, class_covariances

    def _smart_weight_computation(self, client_qualities, sample_nums, feature_scores):
        """
        智能权重计算 - 基于样本数量、质量分数和特征对齐度
        
        理论基础:
        综合考虑三个因素:
        1. 样本数量(统计效率)
        2. 训练质量(优化效果)
        3. 特征对齐度(知识一致性)
        
        数学公式:
        w_i = (1-λ)·(α·norm(n_i) + β·norm(q_i)) + λ·norm(f_i)
        
        其中:
        - norm()表示归一化操作
        - n_i是客户端样本数
        - q_i是客户端质量分数
        - f_i是特征对齐分数
        - λ是特征对齐权重(取值[0,1])
        - α,β是样本数和质量的权重(α+β=1)
        """
        # 1. 基于样本数的基础权重
        total_samples = sum(sample_nums)
        base_weights = np.array(sample_nums) / total_samples
        
        # 2. 质量分数归一化
        quality_scores = np.array(client_qualities)
        quality_normalized = quality_scores / (np.sum(quality_scores) + 1e-8)
        
        # 3. 特征对齐分数归一化
        feature_scores = np.array(feature_scores)
        feature_normalized = feature_scores / (np.sum(feature_scores) + 1e-8)
        
        # 4. 混合权重计算
        alpha = 0.3  # 样本数权重
        beta = 0.7   # 质量分数权重
        
        # 两阶段混合
        base_mixed = alpha * base_weights + beta * quality_normalized
        final_weights = (1 - self.feature_alignment_weight) * base_mixed + \
                        self.feature_alignment_weight * feature_normalized
        
        # 5. 确保最小权重
        min_weight = self.min_client_weight / len(final_weights)
        final_weights = np.maximum(final_weights, min_weight)
        
        # 6. 重新归一化
        final_weights = final_weights / np.sum(final_weights)
        
        return final_weights

    def _update_feature_banks(self, new_class_features):
        """
        更新全局特征库 - 实现特征动量更新
        
        理论基础:
        特征库维护一个移动平均的特征集合，以更好表示全局分布
        
        数学公式:
        F_t = (1-m)·F_{t-1} + m·F_new
        
        其中:
        - F_t是当前轮次的特征库
        - F_{t-1}是上一轮次的特征库
        - F_new是新收集的特征
        - m是动量参数(取值[0,1])
        """
        momentum = 0.8  # 特征动量参数
        
        for class_id, features in new_class_features.items():
            if not features:
                continue
                
            features = np.array(features)
            
            if class_id not in self.global_feature_banks:
                # 初始化特征库
                self.global_feature_banks[class_id] = features[:min(len(features), self.feature_bank_size)]
            else:
                # 动量更新特征库
                old_features = self.global_feature_banks[class_id]
                
                # 计算保留的旧特征数量
                keep_size = int(min(len(old_features), self.feature_bank_size * momentum))
                
                # 计算添加的新特征数量
                add_size = min(len(features), self.feature_bank_size - keep_size)
                
                # 更新特征库
                if keep_size > 0:
                    # 随机选择保留的旧特征
                    keep_indices = np.random.choice(len(old_features), keep_size, replace=False)
                    kept_features = old_features[keep_indices]
                    
                    # 随机选择添加的新特征
                    if add_size > 0 and len(features) > 0:
                        add_indices = np.random.choice(len(features), add_size, replace=False)
                        added_features = features[add_indices]
                        self.global_feature_banks[class_id] = np.vstack([kept_features, added_features])
                    else:
                        self.global_feature_banks[class_id] = kept_features
                else:
                    # 只使用新特征
                    add_indices = np.random.choice(len(features), add_size, replace=False)
                    self.global_feature_banks[class_id] = features[add_indices]

    def _aggregate_global_prototypes(self, class_prototypes_list, class_covs_list, weights):
        """
        聚合全局原型 - 基于加权平均
        
        理论基础:
        通过对客户端原型进行加权平均，生成更具代表性的全局原型
        
        数学公式:
        P_g = ∑w_i·P_i
        Σ_g = ∑w_i·Σ_i
        
        其中:
        - P_g是全局原型
        - Σ_g是全局协方差
        - P_i, Σ_i是客户端i的原型和协方差
        - w_i是客户端i的权重
        """
        global_prototypes = {}
        global_covariances = {}
        
        # 收集所有类别的原型
        all_class_protos = {}
        all_class_covs = {}
        
        for client_idx, client_protos in enumerate(class_prototypes_list):
            client_covs = class_covs_list[client_idx]
            client_weight = weights[client_idx]
            
            for class_id, proto in client_protos.items():
                if class_id not in all_class_protos:
                    all_class_protos[class_id] = []
                    all_class_covs[class_id] = []
                
                cov = client_covs.get(class_id, None)
                if cov is not None:
                    all_class_protos[class_id].append((proto, client_weight))
                    all_class_covs[class_id].append((cov, client_weight))
        
        # 加权聚合每个类别的原型和协方差
        for class_id, proto_and_weights in all_class_protos.items():
            if proto_and_weights:
                # 加权平均原型
                proto_arrays = [p for p, _ in proto_and_weights]
                proto_weights = [w for _, w in proto_and_weights]
                proto_weights = np.array(proto_weights) / np.sum(proto_weights)
                global_prototypes[class_id] = np.average(proto_arrays, axis=0, weights=proto_weights)
                
                # 加权平均协方差
                if class_id in all_class_covs and all_class_covs[class_id]:
                    cov_arrays = [c for c, _ in all_class_covs[class_id]]
                    cov_weights = [w for _, w in all_class_covs[class_id]]
                    cov_weights = np.array(cov_weights) / np.sum(cov_weights)
                    global_covariances[class_id] = np.average(cov_arrays, axis=0, weights=cov_weights)
        
        return global_prototypes, global_covariances

    def aggregate(self, server, selected_workers, round_num, global_weights):
        """
        FedSPD聚合 - 基于特征对齐和互信息最大化的知识蒸馏
        
        聚合流程:
        1. 收集客户端特征、原型和训练信息
        2. 计算特征对齐分数和互信息
        3. 基于多因素确定聚合权重
        4. 聚合模型参数和知识库
        5. 更新全局知识用于下一轮训练
        """
        if not selected_workers:
            return global_weights, []
        
        print(f"\n🧠 第{round_num}轮 FedSPD聚合 (客户端数: {len(selected_workers)})")
        
        # 1. 收集客户端数据
        client_weights_list = []
        client_qualities = []
        sample_nums = []
        train_losses = []
        client_names = []
        
        # 特征相关数据
        class_features_list = []
        class_prototypes_list = []
        class_covs_list = []
        feature_alignment_scores = []
        mutual_information_scores = []
        
        for client_name, worker in selected_workers.items():
            # 调用客户端训练
            client_weight, sample_num, train_loss, class_features, class_labels = worker.local_train(
                sync_round=round_num,
                weights=global_weights,
                global_prototypes=self.global_prototypes,
                global_covariances=self.global_covariances
            )
            
            # 处理特征数据
            flattened_features = []
            flattened_labels = []
            for class_id, features in class_features.items():
                flattened_features.extend(features)
                flattened_labels.extend([class_id] * len(features))
            
            # 计算类原型和协方差
            class_protos, class_covs = self._cluster_features(flattened_features, flattened_labels)
            
            # 计算特征对齐分数
            feature_alignment = self._compute_feature_alignment_score(class_protos, class_covs)
            
            # 计算互信息
            mutual_info = self._estimate_mutual_information(flattened_features, flattened_labels)
            
            # 计算客户端质量分数
            quality_score = self._compute_client_quality_score(
                train_loss, feature_alignment, mutual_info, client_name, round_num
            )
            
            # 收集数据
            client_weights_list.append(client_weight)
            client_qualities.append(quality_score)
            sample_nums.append(sample_num)
            train_losses.append(train_loss)
            client_names.append(client_name)
            
            # 收集特征数据
            class_features_list.append(class_features)
            class_prototypes_list.append(class_protos)
            class_covs_list.append(class_covs)
            feature_alignment_scores.append(feature_alignment)
            mutual_information_scores.append(mutual_info)
            
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 2. 智能权重计算
        aggregation_weights = self._smart_weight_computation(
            client_qualities, sample_nums, feature_alignment_scores
        )
        
        # 3. 聚合模型权重
        final_model_weights = average_weight(client_weights_list, aggregation_weights)
        
        # 4. 聚合原型和协方差
        final_prototypes, final_covariances = self._aggregate_global_prototypes(
            class_prototypes_list, class_covs_list, aggregation_weights
        )
        
        # 5. 更新特征库
        for class_features in class_features_list:
            self._update_feature_banks(class_features)
        
        # 6. 更新全局状态
        self.global_prototypes = final_prototypes
        self.global_covariances = final_covariances
        
        # 7. 打印统计信息
        avg_loss = np.mean(train_losses)
        avg_quality = np.mean(client_qualities)
        avg_alignment = np.mean(feature_alignment_scores)
        avg_mi = np.mean(mutual_information_scores)
        
        print(f"📊 聚合统计:")
        print(f"   - 平均训练损失: {avg_loss:.4f}")
        print(f"   - 平均质量分数: {avg_quality:.4f}")
        print(f"   - 平均特征对齐度: {avg_alignment:.4f}")
        print(f"   - 平均互信息分数: {avg_mi:.4f}")
        print(f"   - 权重分布: {[f'{w:.3f}' for w in aggregation_weights]}")
        print(f"   - 全局知识: 原型({len(final_prototypes)}类), 协方差({len(final_covariances)}类)")
        
        # 显示性能最好的客户端
        best_client_idx = np.argmax(client_qualities)
        best_client = client_names[best_client_idx]
        print(f"   - 🏆 最佳客户端: {best_client} (质量: {client_qualities[best_client_idx]:.3f})")
        
        return final_model_weights, train_losses
