# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/20 10:30
# @Describe: FEDGKD (Federated Global Knowledge Distillation) 聚合策略

import torch
import numpy as np
import copy

from fl.aggregation.aggregator import average_weight
from fl.server.strategy.strategy_base import AggregationStrategy


class FedGKDStrategy(AggregationStrategy):
    """
    FEDGKD (Federated Global Knowledge Distillation) 聚合策略
    
    通过历史全局模型的知识蒸馏引导客户端本地训练，缓解客户端漂移问题
    """
    
    def __init__(self):
        # 模型缓冲区，用于存储历史全局模型
        self.model_buffer = []
        # 集成模型权重
        self.ensemble_weights = None
        # 配置参数
        self.buffer_size = 5  # 默认缓冲区大小M
        self.use_vote_mode = False  # 是否使用FEDGKD-VOTE模式
    
    def initialize(self, server_kwargs):
        """
        从服务器参数初始化算法特定数据
        
        Args:
            server_kwargs: 服务器初始化时传入的参数
        """
        # 从kwargs中获取配置参数
        self.buffer_size = server_kwargs.get('buffer_size', 5)
        self.use_vote_mode = server_kwargs.get('use_vote_mode', False)
        
        # 初始化其他参数
        self.model_buffer = []
        self.ensemble_weights = None
    
    def _update_model_buffer(self, global_weights):
        """
        更新模型缓冲区，保存最近的M个全局模型
        
        Args:
            global_weights: 当前全局模型权重
        """
        # 保存当前全局模型的深拷贝到缓冲区
        self.model_buffer.append(copy.deepcopy(global_weights))
        
        # 如果缓冲区大小超过限制，移除最旧的模型
        if len(self.model_buffer) > self.buffer_size:
            self.model_buffer.pop(0)
    
    def _build_ensemble_model(self):
        """
        构建集成模型，通过平均最近M个模型得到
        
        Returns:
            ensemble_weights: 集成模型权重
        """
        if not self.model_buffer:
            return None
        
        # 简单平均所有缓冲区中的模型权重
        return average_weight(self.model_buffer)
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """
        执行FEDGKD聚合逻辑
        
        Args:
            server: 服务器实例
            selected_workers: 选中的工作节点
            round_num: 当前轮次
            global_weights: 全局模型权重
            
        Returns:
            tuple: (更新后的全局权重, 训练损失列表)
        """
        if not selected_workers:
            return global_weights, []
        
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        
        # 更新模型缓冲区
        self._update_model_buffer(global_weights)
        
        # 构建集成模型
        self.ensemble_weights = self._build_ensemble_model()
        
        import ray
        
        futures = []
        for client_name, worker in selected_workers.items():
            # 根据模式决定发送什么模型到客户端
            if self.use_vote_mode and len(self.model_buffer) >= 2:
                # FEDGKD-VOTE模式: 发送最近M个历史全局模型
                futures.append(worker.local_train.remote(
                    sync_round=round_num,
                    weights=global_weights,
                    historical_models=self.model_buffer[-self.buffer_size:]  # 最近M个模型
                ))
            else:
                # 默认FEDGKD模式: 发送当前模型和集成模型
                futures.append(worker.local_train.remote(
                    sync_round=round_num,
                    weights=global_weights,
                    ensemble_weights=self.ensemble_weights
                ))
            
        results = ray.get(futures)
        
        for i, client_name in enumerate(selected_workers.keys()):
            model_weights, num_sample, avg_loss = results[i]
            
            # 处理客户端返回结果
            client_weight_list.append(model_weights)
            sample_num_list.append(num_sample)
            train_loss_list.append(avg_loss)
            server.history["workers"][client_name]["train_loss"].append(avg_loss)
        
        # 执行加权平均聚合
        new_global_weights = average_weight(client_weight_list, sample_num_list)
        
        return new_global_weights, train_loss_list
