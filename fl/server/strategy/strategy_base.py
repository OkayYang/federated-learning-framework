# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/19 16:30
# @Describe: 聚合策略接口和实现类

from abc import ABC

from fl.aggregation.aggregator import average_weight


class AggregationStrategy(ABC):
    """聚合策略基类"""
    
    def initialize(self, server_kwargs):
        """
        从服务器参数初始化算法特定数据
        
        Args:
            server_kwargs: 服务器初始化时传入的参数
        """
        pass
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """
        执行聚合逻辑的默认实现 - 简单的加权平均
        
        Args:
            server: 服务器实例
            selected_workers: 选中的工作节点
            round_num: 当前轮次
            global_weights: 全局模型权重
            
        Returns:
            tuple: (更新后的全局状态, 训练损失列表)
        """
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        
        if not selected_workers:
            return global_weights, []
        
        import ray
        
        # Launch all local training tasks in parallel
        futures = []
        for client_name, worker in selected_workers.items():
            futures.append(worker.local_train.remote(
                sync_round=round_num, 
                weights=global_weights
            ))
            
        # Wait for all results
        results = ray.get(futures)
        
        for i, client_name in enumerate(selected_workers.keys()):
            model_weights, num_sample, avg_loss = results[i]
            
            # 处理客户端返回结果
            client_weight_list.append(model_weights)
            sample_num_list.append(num_sample)
            train_loss_list.append(avg_loss)
            server.history["workers"][client_name]["train_loss"].append(avg_loss)
        
        # 执行加权平均聚合
        global_weight = average_weight(client_weight_list, sample_num_list)
        
        return global_weight, train_loss_list
    

class FedAloneStrategy(AggregationStrategy):
    """FedAlone聚合策略 - 使用基础聚合实现"""
    pass

class FedAvgStrategy(AggregationStrategy):
    """FedAvg聚合策略 - 使用基础聚合实现"""
    pass


class FedProxStrategy(AggregationStrategy):
    """FedProx聚合策略 - 使用基础聚合实现"""
    pass


class MoonStrategy(AggregationStrategy):
    """Moon聚合策略 - 使用基础聚合实现"""
    pass


