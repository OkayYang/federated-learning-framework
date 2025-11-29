# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/19 16:30
# @Describe: FedDistill聚合策略实现

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from fl.server.strategy.strategy_base import AggregationStrategy
from fl.aggregation.aggregator import average_weight
from fl.aggregation.aggregator import average_logits


class FedDistillStrategy(AggregationStrategy):
    """FedDistill聚合策略"""
    def __init__(self):
        self.global_logits = None
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """重写聚合方法，处理logits"""
        client_weight_list = []
        client_logits_list = []
        sample_num_list = []
        train_loss_list = []
        
        if not selected_workers:
            return global_weights, []
        
        import ray
        
        futures = []
        for client_name, worker in selected_workers.items():
            # 使用自定义的客户端更新方法
            futures.append(worker.local_train.remote(
                sync_round=round_num,
                weights=global_weights,
                global_logits=self.global_logits
            ))
            
        results = ray.get(futures)
        
        for i, client_name in enumerate(selected_workers.keys()):
            client_weight, sample_num, train_loss, client_logits = results[i]
            
            # 处理结果
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            client_logits_list.append((client_logits, sample_num))
            train_loss_list.append(train_loss)
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 聚合模型权重
        global_weight = average_weight(client_weight_list, sample_num_list)

        # 聚合logits
        global_logits = average_logits(client_logits_list, sample_num_list)
        self.global_logits = global_logits
        
        return global_weight, train_loss_list