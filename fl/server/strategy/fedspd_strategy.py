import torch
import numpy as np
import copy

from fl.aggregation.aggregator import average_weight
from fl.server.strategy.strategy_base import AggregationStrategy


class FedSPDStrategy(AggregationStrategy):
    """FedSPD聚合策略 - 简化版本"""
    def __init__(self):
        # 核心状态
        self.global_reps = None  # 全局表征
        self.global_logits = None  # 全局logits
        self.eps = 1e-10  # 添加一个小的常数避免除零
        
        # 历史记录 - 简化版
        self.global_reps_history = {}  # 每个类别的历史表征 {class_id: [rep1, rep2, ...]}
        self.global_logits_history = {}  # 每个类别的历史logits {class_id: [logit1, logit2, ...]}
        self.history_length = 3  # 只保留最近几轮的历史数据
        self.history_weight = 0.2  # 历史数据的权重
    
    def _compute_confidence_scores(self, logits):
        """基于预测熵计算置信度分数"""
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=-1)
        confidence = 1 - (entropy / (np.log(probs.shape[-1]) + self.eps))  # 归一化熵
        return confidence.numpy()
    
    def _update_history_and_apply(self, global_class_reps, global_logits):
        """更新历史记录并应用历史平均"""
        # 1. 应用历史平均
        if self.global_reps_history:  # 如果有历史数据
            for class_id, rep in global_class_reps.items():
                if class_id in self.global_reps_history and self.global_reps_history[class_id]:
                    # 计算历史平均表征
                    history_reps = np.array(self.global_reps_history[class_id], dtype=np.float32)
                    avg_history = np.mean(history_reps, axis=0).astype(np.float32)
                    # 应用加权平均
                    global_class_reps[class_id] = ((1 - self.history_weight) * rep + self.history_weight * avg_history).astype(np.float32)
        
        if self.global_logits_history:  # 如果有历史数据
            for class_id, logit in global_logits.items():
                if class_id in self.global_logits_history and self.global_logits_history[class_id]:
                    # 计算历史平均logits
                    history_logits = np.array(self.global_logits_history[class_id], dtype=np.float32)
                    avg_history = np.mean(history_logits, axis=0).astype(np.float32)
                    # 应用加权平均
                    global_logits[class_id] = ((1 - self.history_weight) * logit + self.history_weight * avg_history).astype(np.float32)
        
        # 2. 更新历史记录
        for class_id, rep in global_class_reps.items():
            if class_id not in self.global_reps_history:
                self.global_reps_history[class_id] = []
            # 确保保存为float32类型
            self.global_reps_history[class_id].append(rep.astype(np.float32))
            # 保持历史长度
            if len(self.global_reps_history[class_id]) > self.history_length:
                self.global_reps_history[class_id].pop(0)
        
        for class_id, logit in global_logits.items():
            if class_id not in self.global_logits_history:
                self.global_logits_history[class_id] = []
            # 确保保存为float32类型
            self.global_logits_history[class_id].append(logit.astype(np.float32))
            # 保持历史长度
            if len(self.global_logits_history[class_id]) > self.history_length:
                self.global_logits_history[class_id].pop(0)
        
        return global_class_reps, global_logits
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """重写聚合方法，处理类别表征和logits - 简化版"""
        if not selected_workers:  # 如果没有选中的工作节点
            return global_weights, []  # 返回原始权重和空的损失列表
            
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        client_class_reps = []
        client_class_logits = []
        
        # 1. 收集客户端训练结果
        for client_name, worker in selected_workers.items():
            client_weight, sample_num, train_loss, class_reps, class_logits = worker.local_train(
                sync_round=round_num,
                weights=global_weights,
                global_reps=self.global_reps,
                global_logits=self.global_logits
            )
            
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            train_loss_list.append(train_loss)
            client_class_reps.append(class_reps)
            client_class_logits.append(class_logits)
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 2. 计算样本权重
        sample_weights = np.array([max(float(w), self.eps) for w in sample_num_list], dtype=np.float32)
        total_samples = np.sum(sample_weights)
        if total_samples < self.eps:  # 如果总样本数接近零
            sample_weights = np.ones_like(sample_weights, dtype=np.float32) / len(sample_weights)
        else:
            sample_weights = sample_weights / total_samples
            
        # 3. 聚合模型权重
        global_weight = average_weight(client_weight_list, sample_weights.tolist())
        
        # 4. 聚合表征和logits
        all_class_reps = {}
        all_class_logits = {}
        
        # 收集所有客户端的表征和logits
        for client_idx, (client_rep, client_logit) in enumerate(zip(client_class_reps, client_class_logits)):
            client_weight = float(sample_weights[client_idx])  # 使用样本数量作为权重
            
            # 收集表征
            for class_id, rep in client_rep.items():
                if class_id not in all_class_reps:
                    all_class_reps[class_id] = []
                # 确保rep是float32类型
                all_class_reps[class_id].append((np.array(rep, dtype=np.float32), client_weight))
            
            # 收集logits
            for class_id, logit in client_logit.items():
                if class_id not in all_class_logits:
                    all_class_logits[class_id] = []
                # 确保logit是float32类型
                all_class_logits[class_id].append((np.array(logit, dtype=np.float32), client_weight))
        
        # 简单加权平均聚合
        global_class_reps = {}
        global_logits = {}
        
        # 聚合表征
        for class_id, reps_and_weights in all_class_reps.items():
            reps = np.array([rep for rep, _ in reps_and_weights], dtype=np.float32)
            weights = np.array([weight for _, weight in reps_and_weights], dtype=np.float32)
            weights = weights / (np.sum(weights) + self.eps)  # 归一化权重
            global_class_reps[class_id] = np.average(reps, axis=0, weights=weights).astype(np.float32)
        
        # 聚合logits
        for class_id, logits_and_weights in all_class_logits.items():
            logits = np.array([logit for logit, _ in logits_and_weights], dtype=np.float32)
            weights = np.array([weight for _, weight in logits_and_weights], dtype=np.float32)
            weights = weights / (np.sum(weights) + self.eps)  # 归一化权重
            global_logits[class_id] = np.average(logits, axis=0, weights=weights).astype(np.float32)
        
        # 5. 应用历史平均并更新历史记录
        global_class_reps, global_logits = self._update_history_and_apply(global_class_reps, global_logits)
        
        # 6. 更新全局表征和logits
        self.global_reps = global_class_reps
        self.global_logits = global_logits
        
        return global_weight, train_loss_list