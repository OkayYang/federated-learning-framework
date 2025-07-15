import torch
import numpy as np
import copy

from fl.aggregation.aggregator import average_weight
from fl.server.strategy.strategy_base import AggregationStrategy


class FedSPDStrategy(AggregationStrategy):
    """FedSPD历史教师策略
    
    核心设计：
    - 维护历史全局模型缓冲区
    - 计算历史模型平均作为教师
    - 下发历史平均教师给客户端进行蒸馏
    
    优势：
    - 稳定的历史知识（避免单轮波动）
    - 简化接口（无需类别级知识传递）
    - 双重蒸馏（客户端做reps+logits蒸馏）
    """
    def __init__(self):
        # 历史模型管理
        self.model_buffer = []  # 历史全局模型缓冲区
        self.buffer_size = 5    # 缓冲区大小
        self.eps = 1e-10        # 避免除零

    def _update_model_buffer(self, global_weights):
        """更新历史模型缓冲区"""
        import copy
        # 保存当前全局模型的深拷贝到缓冲区
        self.model_buffer.append(copy.deepcopy(global_weights))
        
        # 如果缓冲区大小超过限制，移除最旧的模型
        if len(self.model_buffer) > self.buffer_size:
            self.model_buffer.pop(0)
    
    def _build_ensemble_teacher(self):
        """构建历史模型平均作为教师"""
        if not self.model_buffer:
            return None
        
        # 简单平均所有缓冲区中的模型权重
        from fl.aggregation.aggregator import average_weight
        return average_weight(self.model_buffer)

    def aggregate(self, server, selected_workers, round_num, global_weights):
        """FedSPD历史教师聚合方法"""
        if not selected_workers:
            return global_weights, []
        
        # 1. 更新历史模型缓冲区
        self._update_model_buffer(global_weights)
        
        # 2. 构建历史平均教师
        ensemble_teacher = self._build_ensemble_teacher()
            
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        
        # 3. 收集客户端训练结果（传递历史平均教师）
        for client_name, worker in selected_workers.items():
            client_weight, sample_num, train_loss = worker.local_train(
                sync_round=round_num,
                weights=global_weights,
                ensemble_weights=ensemble_teacher  # 🚀 传递历史平均教师
            )
            
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            train_loss_list.append(train_loss)
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 4. 聚合全局模型权重
        sample_weights = np.array([max(float(w), self.eps) for w in sample_num_list], dtype=np.float32)
        total_samples = np.sum(sample_weights)
        if total_samples < self.eps:
            sample_weights = np.ones_like(sample_weights, dtype=np.float32) / len(sample_weights)
        else:
            sample_weights = sample_weights / total_samples
            
        global_weight = average_weight(client_weight_list, sample_weights.tolist())
        
        return global_weight, train_loss_list