# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/19 16:30
# @Describe: Scaffold聚合策略实现

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from fl.server.strategy.strategy_base import AggregationStrategy
from fl.aggregation.aggregator import average_weight
from fl.aggregation.aggregator import average_scaffold_parameter_c

class ScaffoldStrategy(AggregationStrategy):
    """Scaffold聚合策略"""
    def __init__(self):
        # 初始化全局控制变量为None，将在第一轮聚合时创建
        self.global_c = None
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """
        重写聚合方法，处理控制变量
        :param server: 服务器实例
        :param selected_workers: 被选中的客户端字典 {client_name: worker}
        :param round_num: 当前通信轮次
        :param global_weights: 全局模型权重
        :return: (更新后的全局权重, 训练损失列表)
        """
        client_weight_list = []   # 客户端模型权重列表
        client_c_list = []        # 客户端控制变量列表
        sample_num_list = []      # 客户端样本数量列表
        train_loss_list = []      # 客户端训练损失列表
        client_delta_y = []       # 客户端模型更新量(yi - x)
        
        if not selected_workers:
            return global_weights, []
            
        # 遍历所有选中的客户端，收集更新
        import ray
        
        # 遍历所有选中的客户端，收集更新
        futures = []
        for client_name, worker in selected_workers.items():
            # 使用自定义的客户端更新方法，传入全局控制变量
            futures.append(worker.local_train.remote(
                sync_round=round_num,
                weights=global_weights,
                cg=self.global_c
            ))
            
        results = ray.get(futures)
        
        for i, client_name in enumerate(selected_workers.keys()):
            client_weight, sample_num, train_loss, c = results[i]
            
            # 收集每个客户端的结果
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            client_c_list.append(c)
            train_loss_list.append(train_loss)
            # 将训练损失记录到服务器历史中
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 聚合模型权重 - 使用加权平均
        global_weight = average_weight(client_weight_list, sample_num_list)
        
        # 聚合控制变量 - 参考GitHub实现方式
        # 如果是第一轮，初始化全局控制变量
        if self.global_c is None:
            # 第一轮时，直接使用客户端控制变量的加权平均作为全局控制变量
            self.global_c = average_scaffold_parameter_c(client_c_list, sample_num_list)
        else:
            # 后续轮次，按照SCAFFOLD论文更新全局控制变量
            # 计算客户端控制变量的加权平均
            avg_client_c = average_scaffold_parameter_c(client_c_list, sample_num_list)
            
            # 更新全局控制变量: c = c + |S|/N * (1/|S| * sum(delta_ci) - 0)
            # 这里|S|是选中的客户端数量，N是总客户端数量
            participation_rate = len(selected_workers) / len(server._workers)
            
            # 更新全局控制变量
            self.global_c = [
                global_c + participation_rate * (client_c - global_c)
                for global_c, client_c in zip(self.global_c, avg_client_c)
            ]
        
        return global_weight, train_loss_list