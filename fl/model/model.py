# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/13 14:49
# @Describe:
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    生成器模型，用于生成合成数据样本
    """
    def __init__(self, latent_dim, feature_dim, hidden_dim=256, num_classes=62):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # 嵌入层，将类别标签转换为嵌入向量
        self.label_embedding = nn.Embedding(num_classes, hidden_dim)
        
        # 生成器网络
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        # 训练信息
        self.train_epochs = 20
        self.train_batch_size = 64
        self.train_lr = 0.001
        self.ensemble_alpha=1
        self.ensemble_beta=0
        self.ensemble_eta=1
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.train_lr)
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, z, labels):
        """
        输入噪声向量和标签，输出合成特征
        """
        # 将标签转换为嵌入向量
        label_embedding = self.label_embedding(labels)
        # 将噪声向量和标签嵌入连接起来
        concat_input = torch.cat([z, label_embedding], dim=1)
        # 生成特征
        features = self.generator(concat_input)
        return features
    def get_weights(self, return_numpy=False):
        if not return_numpy:
            return {k: v.cpu() for k, v in self.state_dict().items()}
        else:
            weights_list = []
            for v in self.state_dict().values():
                weights_list.append(v.cpu().numpy())
            return [e.copy() for e in weights_list]
    def update_weights(self, weights:np.ndarray):
        if len(weights) != len(self.state_dict()):
            raise ValueError("传入的权重数组数量与模型参数数量不匹配。")
        keys = self.state_dict().keys()
        weights_dict = {}
        for k, v in zip(keys, weights):
            weights_dict[k] = torch.Tensor(np.copy(v))
        self.load_state_dict(weights_dict) 
    
    def diversity_loss(self, eps, features):
        """
        简化的多样性损失，使用L1距离
        
        Args:
            eps (torch.Tensor): 随机噪声输入
            features (torch.Tensor): 生成的特征
        
        Returns:
            torch.Tensor: 多样性损失
        """
        # 特征标准化
        features_normalized = F.normalize(features, p=2, dim=1)
        
        # 计算所有特征对之间的相似度矩阵
        batch_size = features.size(0)
        similarity_matrix = torch.matmul(features_normalized, features_normalized.t())
        
        # 移除自身相似度（对角线元素）
        mask = torch.eye(batch_size, device=features.device)
        similarity_matrix = similarity_matrix * (1 - mask)
        
        # 计算平均相似度作为多样性损失（越相似，损失越大）
        diversity_loss = torch.mean(similarity_matrix)
        
        return diversity_loss


class FeMNISTNet(nn.Module):
    """Small ConvNet for FeMNIST."""

    def __init__(self):
        super(FeMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1个输入通道，6个输出通道，5x5的卷积核
        self.pool = nn.MaxPool2d(2, 2)  # 2x2的池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6个输入通道，16个输出通道，5x5的卷积核
        self.fc1 = nn.Linear(16 * 4 * 4, 128)  # 4x4是通过两次2x2的池化层得到的
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x, return_all=False,start_layer=False):
        if start_layer:
            x = self.fc1(x)
            x = self.fc2(F.relu(x))
            return x
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            h = x.view(-1, 16 * 4 * 4)
            x = self.fc1(h)
            y = self.fc2(F.relu(x))
            if return_all:
                return h, x, y
            return y


# 定义简单的 CNN 模型
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 第一层卷积，输入通道为1（MNIST图像是单通道），输出通道为32，卷积核大小为3
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积，输入通道为32，输出通道为64，卷积核大小为3
        self.conv2 = nn.Conv2d(32, 64, 3)
        # 全连接层输入通道数量 = 64 * 5 * 5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)  # 输出10个类别

    def forward(self, x, return_all=False):
        # 第一层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 将二维数据展平为一维
        h = x.view(-1, 64 * 5 * 5)
        # 全连接层
        x = self.fc1(h)
        y = self.fc2(F.relu(x))
        if return_all:
            return h, x, y
        return y
