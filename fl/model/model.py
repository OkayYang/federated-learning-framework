# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/13 14:49
# @Describe:
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeMNISTNet(nn.Module):
    """Small ConvNet for FeMNIST."""

    def __init__(self):
        super(FeMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1个输入通道，6个输出通道，5x5的卷积核
        self.pool = nn.MaxPool2d(2, 2)  # 2x2的池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6个输入通道，16个输出通道，5x5的卷积核
        self.fc1 = nn.Linear(16 * 4 * 4, 128)  # 4x4是通过两次2x2的池化层得到的
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x, start_layer="raw", return_all=False):
        if start_layer == "hidden":
            x = self.fc1(h)
            x = self.fc2(F.relu(x))
            return x
        elif start_layer == "classify":
            x = self.fc2(x)
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

    def forward(self, x, start_layer="raw", return_all=False):
        if start_layer == "hidden":
            x = self.fc1(x)
            x = self.fc2(F.relu(x))
            return x
        elif start_layer == "classify":
            x = self.fc2(x)
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
    
    def forward(self, x, start_layer="raw", return_all=False):
        """
        前向传播函数
        
        Args:
            x: 输入数据
            start_layer: 是否从映射层开始（用于fedgen等）
            return_all: 是否返回特征（用于对比学习等）
            
        Returns:
            模型输出
        """
        if start_layer == "hidden":
            # 从映射层开始，适用于生成的特征输入
            hidden = self.mapping(x)
            # 通过输出层获得类别预测
            logits = self.classifier(hidden)
            return logits
        elif start_layer == "classify":
            logits = self.classifier(x)
            return logits
        else:
            # 从头部开始，提取特征
            x = self.backbone(x)
            features = x.view(x.size(0), -1)
        
            # 通过映射层
            hidden = self.mapping(features)
            
            # 通过输出层获得类别预测
            logits = self.classifier(hidden)
            
            if return_all:
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

class SVHNNet(nn.Module):
    """
    SVHN网络模型，专为Street View House Numbers数据集设计
    类似CIFAR10Net，但针对SVHN数据集特点进行了优化
    """
    def __init__(self):
        super(SVHNNet, self).__init__()
        # 头部特征提取层 - 处理SVHN的32x32彩色图像
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
        self.classifier = nn.Linear(256, 10)  # SVHN有10个类别（数字0-9）
    
    def forward(self, x, start_layer="raw", return_all=False):
        """
        前向传播函数
        
        Args:
            x: 输入数据
            start_layer: 是否从映射层开始（用于fedgen等）
            return_all: 是否返回特征（用于对比学习等）
            
        Returns:
            模型输出
        """
        if start_layer == "hidden":
            # 从映射层开始，适用于生成的特征输入
            hidden = self.mapping(x)
            # 通过输出层获得类别预测
            logits = self.classifier(hidden)
            return logits
        elif start_layer == "classify":
            logits = self.classifier(x)
            return logits
        else:
            # 从头部开始，提取特征
            x = self.backbone(x)
            features = x.view(x.size(0), -1)
        
            # 通过映射层
            hidden = self.mapping(features)
            
            # 通过输出层获得类别预测
            logits = self.classifier(hidden)
            
            if return_all:
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
    
    def forward(self, x, start_layer="raw", return_all=False):
        """
        前向传播函数
        
        Args:
            x: 输入数据
            start_layer: 是否从映射层开始（用于fedgen等）
            return_all: 是否返回特征（用于对比学习等）
            
        Returns:
            模型输出
        """
        if start_layer == "hidden":
            # 从映射层开始，适用于生成的特征输入
            hidden = self.mapping(x)
            # 通过输出层获得类别预测
            logits = self.classifier(hidden)
            return logits
        elif start_layer == "classify":
            logits = self.classifier(x)
            return logits
        else:
            # 从头部开始，提取特征
            x = self.backbone(x)
            features = x.view(x.size(0), -1)
        
            # 通过映射层
            hidden = self.mapping(features)
            
            # 通过输出层获得类别预测
            logits = self.classifier(hidden)
            
            if return_all:
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

class TinyImageNetNet(nn.Module):
    """
    TinyImageNet网络模型，模块化设计为三个部分：
    1. 头部特征提取层 (Backbone)：卷积层，提取图像特征
    2. 映射层 (Mapping Layer)：将特征映射到隐藏空间
    3. 输出层 (Output Layer)：分类层，输出类别概率
    
    TinyImageNet包含200个类别，图像尺寸为64x64
    """
    def __init__(self):
        super(TinyImageNetNet, self).__init__()
        # 头部特征提取层 - 使用更深的卷积网络处理TinyImageNet的64x64彩色图像
        self.backbone = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            
            # 第四个卷积块 - 添加额外层以处理更大的输入尺寸
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
        )
        
        # 计算特征维度
        self.feature_dim = 512 * 4 * 4  # 4次下采样后，64x64 -> 4x4
        
        # 映射层 - 将卷积特征映射到较低维的隐藏空间
        self.mapping = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        # 输出层 - 将隐藏空间映射到类别空间
        self.classifier = nn.Linear(512, 200)  # TinyImageNet有200个类别
    
    def forward(self, x, start_layer="raw", return_all=False):
        """
        前向传播函数
        
        Args:
            x: 输入数据
            start_layer: 是否从映射层开始（用于fedgen等）
            return_all: 是否返回特征（用于对比学习等）
            
        Returns:
            模型输出
        """
        if start_layer == "hidden":
            # 从映射层开始，适用于生成的特征输入
            hidden = self.mapping(x)
            # 通过输出层获得类别预测
            logits = self.classifier(hidden)
            return logits
        elif start_layer == "classify":
            logits = self.classifier(x)
            return logits
        else:
            # 从头部开始，提取特征
            x = self.backbone(x)
            features = x.view(x.size(0), -1)
        
            # 通过映射层
            hidden = self.mapping(features)
            
            # 通过输出层获得类别预测
            logits = self.classifier(hidden)
            
            if return_all:
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
        
# ResNet基本模块
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 全连接分类层
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, start_layer="raw", return_all=False):
        if start_layer == "hidden":
            # 假设输入是最后一个卷积层的特征
            x = self.linear(x)
            return x
        elif start_layer == "classify":
            # 直接进行分类
            x = self.linear(x)
            return x
        else:
            # 完整的前向传播
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.avg_pool2d(x, 4)
            features = x.view(x.size(0), -1)
            logits = self.linear(features)
            
            if return_all:
                return features, features, logits  # 为了保持一致性，返回相同的特征两次
            
            return logits
    
    def get_features(self, x):
        """提取输入的特征表示"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        return x.view(x.size(0), -1)
    
    def get_hidden(self, features):
        """将特征映射到隐藏空间 - 在ResNet中可以直接返回特征"""
        return features
    
    def classify(self, hidden):
        """将隐藏表示分类到各个类别"""
        return self.linear(hidden)

# ResNet18实现 - CIFAR10
class ResNet18_CIFAR10(nn.Module):
    def __init__(self):
        super(ResNet18_CIFAR10, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        
    def forward(self, x, start_layer="raw", return_all=False):
        return self.model(x, start_layer, return_all)
        
    def get_features(self, x):
        return self.model.get_features(x)
    
    def get_hidden(self, features):
        return self.model.get_hidden(features)
    
    def classify(self, hidden):
        return self.model.classify(hidden)

# ResNet18实现 - CIFAR100
class ResNet18_CIFAR100(nn.Module):
    def __init__(self):
        super(ResNet18_CIFAR100, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)
        
    def forward(self, x, start_layer="raw", return_all=False):
        return self.model(x, start_layer, return_all)
        
    def get_features(self, x):
        return self.model.get_features(x)
    
    def get_hidden(self, features):
        return self.model.get_hidden(features)
    
    def classify(self, hidden):
        return self.model.classify(hidden)

# ResNet18实现 - TinyImageNet
class ResNet18_TinyImageNet(nn.Module):
    def __init__(self):
        super(ResNet18_TinyImageNet, self).__init__()
        # ResNet模型基础架构与CIFAR类似，但需要调整输入大小和输出类别数
        # Tiny ImageNet有200个类别，图像大小为64x64
        
        # 创建基础ResNet模型
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=200)
        
        # 由于Tiny ImageNet的图像尺寸是64x64，需要修改平均池化的大小
        # 在前向传播中调整
        
    def forward(self, x, start_layer="raw", return_all=False):
        if start_layer == "hidden":
            # 从中间层开始
            x = self.model.linear(x)
            return x
        elif start_layer == "classify":
            # 直接分类
            x = self.model.linear(x)
            return x
        else:
            # 完整的前向传播
            x = F.relu(self.model.bn1(self.model.conv1(x)))
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            
            # 对于64x64的输入，经过多次下采样后尺寸为8x8
            # 所以这里使用8x8的平均池化
            x = F.avg_pool2d(x, 8)
            
            features = x.view(x.size(0), -1)
            logits = self.model.linear(features)
            
            if return_all:
                return features, features, logits
            
            return logits
        
    def get_features(self, x):
        """提取输入的特征表示"""
        x = F.relu(self.model.bn1(self.model.conv1(x)))
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = F.avg_pool2d(x, 8)  # 对于64x64的输入使用8x8池化
        return x.view(x.size(0), -1)
    
    def get_hidden(self, features):
        """将特征映射到隐藏空间 - 在ResNet中可以直接返回特征"""
        return features
    
    def classify(self, hidden):
        """将隐藏表示分类到各个类别"""
        return self.model.linear(hidden)

