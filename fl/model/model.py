# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/13 14:49
# @Describe:
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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义简单的 CNN 模型
class MNISTNetCNN(nn.Module):
    def __init__(self):
        super(MNISTNetCNN, self).__init__()
        # 第一层卷积，输入通道为1（MNIST图像是单通道），输出通道为32，卷积核大小为3
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积，输入通道为32，输出通道为64，卷积核大小为3
        self.conv2 = nn.Conv2d(32, 64, 3)
        # 全连接层输入通道数量 = 64 * 5 * 5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)  # 输出10个类别

    def forward(self, x):
        # 第一层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 将二维数据展平为一维
        x = x.view(-1, 64 * 5 * 5)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
