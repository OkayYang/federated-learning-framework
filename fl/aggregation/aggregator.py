# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/8 11:18
# @Describe:
from typing import List

import numpy as np


def average_weight(data: List, weights: List[float] = None):
    if weights is None:
        weights = [1] * len(data)

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Initialize weighted sum as None
    weighted_sum = None
    # Iterate over each model's weights
    for i, model_weights in enumerate(data):
        # Scale model weights by their corresponding normalized weight
        scaled_weights = [
            np.array(layer) * normalized_weights[i] for layer in model_weights
        ]
        if weighted_sum is None:
            weighted_sum = scaled_weights
        else:
            weighted_sum = [w + sw for w, sw in zip(weighted_sum, scaled_weights)]

    return weighted_sum

def average_scaffold_parameter_c(data: List, weights: List[float] = None):
    """
    SCAFFOLD算法中控制变量c的聚合函数
    
    Args:
        data: 客户端控制变量c的列表，每个元素是一个客户端的c列表
        weights: 聚合权重，默认为等权重
        
    Returns:
        聚合后的控制变量c
    """
    if weights is None:
        weights = [1] * len(data)

    # 归一化权重
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # 初始化加权和为None
    aggregated_c = None
    
    # 遍历每个客户端的控制变量c
    for i, client_c in enumerate(data):
        # 按权重缩放控制变量
        scaled_c = [c * normalized_weights[i] for c in client_c]
        
        if aggregated_c is None:
            aggregated_c = scaled_c
        else:
            # 元素级别相加
            aggregated_c = [c1 + c2 for c1, c2 in zip(aggregated_c, scaled_c)]

    return aggregated_c

def average_logits(client_logits_list, sample_num_list):
    """
    Aggregate logits from multiple clients using weighted averaging
    Args:
        client_logits_list: List of tuples (client_logits, sample_num)
        sample_num_list: List of sample numbers for each client
    """
    total_samples = sum(sample_num_list)
    normalized_weights = [num / total_samples for num in sample_num_list]
    
    # 获取所有可能的标签
    all_labels = set()
    for client_logits, _ in client_logits_list:
        all_labels.update(client_logits.keys())
    
    # 初始化全局logits字典
    global_logits = {}
    
    # 对每个标签进行加权平均
    for label in all_labels:
        weighted_sum = None
        for i, (client_logits, _) in enumerate(client_logits_list):
            if label in client_logits:
                if weighted_sum is None:
                    weighted_sum = client_logits[label] * normalized_weights[i]
                else:
                    weighted_sum += client_logits[label] * normalized_weights[i]
        
        if weighted_sum is not None:
            global_logits[label] = weighted_sum
    
    return global_logits
