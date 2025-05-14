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
    def __init__(self, latent_dim, feature_dim,num_classes,hidden_dim=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
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

    def forward(self, x, return_all=False,start_layer=False):
        if start_layer:
            x = self.fc1(x)
            x = self.fc2(F.relu(x))
            return x
        else:
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

class CIFAR10Net(nn.Module):
    """
    CIFAR10网络模型，模块化设计为三个部分：
    1. 头部特征提取层 (Backbone)：卷积层，提取图像特征
    2. 映射层 (Mapping Layer)：将特征映射到隐藏空间
    3. 输出层 (Output Layer)：分类层，输出类别概率
    """
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        # 头部特征提取层 - 使用更深的卷积网络处理CIFAR10的32x32彩色图像
        self.backbone = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 计算特征维度
        self.feature_dim = 128 * 4 * 4  # 3次下采样后，32x32 -> 4x4
        
        # 映射层 - 将卷积特征映射到较低维的隐藏空间
        self.mapping = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 输出层 - 将隐藏空间映射到类别空间
        self.classifier = nn.Linear(256, 10)  # CIFAR10有10个类别
    
    def forward(self, x, start_layer=False, return_features=False):
        """
        前向传播函数
        
        Args:
            x: 输入数据
            start_layer: 是否从映射层开始（用于fedgen等）
            return_features: 是否返回特征（用于对比学习等）
            
        Returns:
            模型输出
        """
        if start_layer:
            # 从映射层开始，适用于生成的特征输入
            features = x
        else:
            # 从头部开始，提取特征
            x = self.backbone(x)
            features = x.view(x.size(0), -1)
        
        # 通过映射层
        hidden = self.mapping(features)
        
        # 通过输出层获得类别预测
        logits = self.classifier(hidden)
        
        if return_features:
            # 返回中间特征表示（用于对比学习、特征可视化等）
            return features, hidden, logits
        
        return logits
        
    def get_features(self, x):
        """提取输入的特征表示"""
        x = self.backbone(x)
        return x.view(x.size(0), -1)
    
    def get_hidden(self, features):
        """将特征映射到隐藏空间"""
        return self.mapping(features)
    
    def classify(self, hidden):
        """将隐藏表示分类到各个类别"""
        return self.classifier(hidden)

class CIFAR100Net(nn.Module):
    """
    CIFAR100网络模型，模块化设计为三个部分：
    1. 头部特征提取层 (Backbone)：卷积层，提取图像特征
    2. 映射层 (Mapping Layer)：将特征映射到隐藏空间
    3. 输出层 (Output Layer)：分类层，输出类别概率
    """
    def __init__(self):
        super(CIFAR100Net, self).__init__()
        # 头部特征提取层 - 使用更深的卷积网络处理CIFAR100的32x32彩色图像
        self.backbone = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 计算特征维度
        self.feature_dim = 256 * 4 * 4  # 3次下采样后，32x32 -> 4x4
        
        # 映射层 - 将卷积特征映射到较低维的隐藏空间
        self.mapping = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 输出层 - 将隐藏空间映射到类别空间
        self.classifier = nn.Linear(512, 100)  # CIFAR100有100个类别
    
    def forward(self, x, start_layer=False, return_features=False):
        """
        前向传播函数
        
        Args:
            x: 输入数据
            start_layer: 是否从映射层开始（用于fedgen等）
            return_features: 是否返回特征（用于对比学习等）
            
        Returns:
            模型输出
        """
        if start_layer:
            # 从映射层开始，适用于生成的特征输入
            features = x
        else:
            # 从头部开始，提取特征
            x = self.backbone(x)
            features = x.view(x.size(0), -1)
        
        # 通过映射层
        hidden = self.mapping(features)
        
        # 通过输出层获得类别预测
        logits = self.classifier(hidden)
        
        if return_features:
            # 返回中间特征表示（用于对比学习、特征可视化等）
            return features, hidden, logits
        
        return logits
        
    def get_features(self, x):
        """提取输入的特征表示"""
        x = self.backbone(x)
        return x.view(x.size(0), -1)
    
    def get_hidden(self, features):
        """将特征映射到隐藏空间"""
        return self.mapping(features)
    
    def classify(self, hidden):
        """将隐藏表示分类到各个类别"""
        return self.classifier(hidden)
        