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