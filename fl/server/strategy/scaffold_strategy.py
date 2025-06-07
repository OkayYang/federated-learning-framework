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
        self.global_c = None
    
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """重写聚合方法，处理控制变量"""
        client_weight_list = []
        client_c_list = []
        sample_num_list = []
        train_loss_list = []
        
        for client_name, worker in selected_workers.items():
            # 使用自定义的客户端更新方法
            client_weight, sample_num, train_loss, c =  worker.local_train(
                sync_round=round_num,
                weights=global_weights,
                cg=self.global_c
            )
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            client_c_list.append(c)
            train_loss_list.append(train_loss)
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 聚合模型权重
        global_weight = average_weight(client_weight_list, sample_num_list)
        self.global_c = average_scaffold_parameter_c(client_c_list, sample_num_list)
        return global_weight, train_loss_list