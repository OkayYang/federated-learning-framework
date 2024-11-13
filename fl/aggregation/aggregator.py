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
