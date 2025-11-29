# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/19 16:30
# @Describe: FedFTG算法专用生成器实现

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FedFTGGenerator(nn.Module):
    """
    FedFTG生成器 - 基于论文"Fine-Tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning"
    
    用于生成用于知识蒸馏的合成图像样本
    """
    def __init__(self, z_dim=100, num_classes=10, img_channels=3, img_size=32):
        super(FedFTGGenerator, self).__init__()
        
        self.z_dim = z_dim  # 噪声向量维度
        self.num_classes = num_classes  # 类别数量
        self.img_channels = img_channels  # 图像通道数
        self.img_size = img_size  # 图像大小
        self.latent_dim = z_dim + num_classes  # 潜在空间维度
        
        # 初始尺寸
        self.init_size = self.img_size // 4  # 初始特征图大小
        self.l1 = nn.Sequential(
            nn.Linear(self.latent_dim, 128 * self.init_size ** 2)
        )
        
        # 上采样和卷积生成器
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            
            nn.Upsample(scale_factor=2),  # 初始尺寸*2
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),  # 初始尺寸*4
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, self.img_channels, 3, stride=1, padding=1),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
        # 初始化权重
        self.apply(self._weights_init)
        
    def _weights_init(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    def forward(self, z, labels):
        """
        前向传播
        
        Args:
            z: 随机噪声向量 [batch_size, z_dim]
            labels: 类别标签 [batch_size]
            
        Returns:
            torch.Tensor: 生成的图像 [batch_size, channels, height, width]
        """
        # 创建一热编码
        batch_size = z.size(0)
        onehot = torch.zeros(batch_size, self.num_classes).to(z.device)
        onehot = onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        # 连接噪声和标签
        z = torch.cat([z, onehot], dim=1)
        
        # 生成初始特征图
        out = self.l1(z)
        out = out.view(batch_size, 128, self.init_size, self.init_size)
        
        # 上采样和卷积生成最终图像
        img = self.conv_blocks(out)
        
        return img
    
    def diversity_loss(self, generated_data):
        """
        计算生成样本的多样性损失 (论文公式7)
        
        Args:
            generated_data: 生成的数据 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 多样性损失
        """
        batch_size = generated_data.size(0)
        
        # 如果批次大小太小，直接返回零损失
        if batch_size <= 1:
            return torch.tensor(0.0).to(generated_data.device)
        
        # 平铺图像数据
        generated_data_flat = generated_data.view(batch_size, -1)
        
        # 计算样本间的成对距离
        dist_matrix = torch.cdist(generated_data_flat, generated_data_flat, p=2)
        
        # 对角线为0（自身与自身的距离），需要排除
        mask = torch.ones_like(dist_matrix) - torch.eye(batch_size, device=generated_data.device)
        
        # 计算平均距离的负指数
        num_valid_pairs = batch_size * (batch_size - 1)
        sum_dist = torch.sum(dist_matrix * mask)
        mean_dist = sum_dist / num_valid_pairs
        diversity_loss = torch.exp(-mean_dist / 1.0)  # 缩放因子可调整
        
        return diversity_loss


class FedFTGGeneratorDeepInversion(nn.Module):
    """
    FedFTG生成器替代实现 - 使用DeepInversion方法
    
    不使用标准生成器网络，而是直接优化输入图像（参考DeepInversion方法）
    """
    def __init__(self, img_channels=3, img_size=32, num_classes=10):
        super(FedFTGGeneratorDeepInversion, self).__init__()
        
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_classes = num_classes
        
        # 初始化随机图像参数 - 这将是可训练的参数
        self.synthetic_data = nn.Parameter(
            torch.randn(1, img_channels, img_size, img_size) * 0.1
        )
        
        # 正则化超参数
        self.tv_weight = 0.01  # 总变差正则化权重
        self.l2_weight = 0.01  # L2正则化权重
        
    def forward(self, batch_size=1, labels=None):
        """
        生成合成图像
        
        Args:
            batch_size: 批次大小
            labels: 类别标签（在这个实现中不直接使用）
            
        Returns:
            torch.Tensor: 生成的图像 [batch_size, channels, height, width]
        """
        # 克隆当前的合成数据以保持梯度流
        img = self.synthetic_data.repeat(batch_size, 1, 1, 1)
        
        # 应用Tanh确保输出范围在[-1, 1]之间
        img = torch.tanh(img)
        
        return img
    
    def compute_image_regularization(self, img):
        """
        计算图像正则化损失
        
        Args:
            img: 生成的图像
            
        Returns:
            torch.Tensor: 正则化损失
        """
        # 总变差正则化 - 促进图像平滑
        diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
        diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
        tv_loss = torch.sum(torch.abs(diff_h)) + torch.sum(torch.abs(diff_w))
        
        # L2正则化 - 防止像素值过大
        l2_loss = torch.sum(img ** 2)
        
        # 总正则化损失
        reg_loss = self.tv_weight * tv_loss + self.l2_weight * l2_loss
        
        return reg_loss
    
    def diversity_loss(self, img_batch):
        """计算多样性损失"""
        batch_size = img_batch.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0).to(img_batch.device)
        
        # 平铺图像
        flat_imgs = img_batch.view(batch_size, -1)
        
        # 计算批次中图像间的L2距离
        dist_matrix = torch.cdist(flat_imgs, flat_imgs, p=2)
        
        # 移除对角线（自身与自身的距离）
        mask = 1.0 - torch.eye(batch_size, device=img_batch.device)
        
        # 计算平均距离的负指数
        mean_dist = torch.sum(dist_matrix * mask) / (batch_size * (batch_size - 1))
        div_loss = torch.exp(-mean_dist)
        
        return div_loss


def create_fedftg_generator(model_type='standard', **kwargs):
    """
    创建FedFTG生成器
    
    Args:
        model_type: 生成器类型，'standard'或'deepinversion'
        **kwargs: 其他参数
        
    Returns:
        nn.Module: FedFTG生成器模型
    """
    if model_type.lower() == 'standard':
        return FedFTGGenerator(
            z_dim=kwargs.get('z_dim', 100),
            num_classes=kwargs.get('num_classes', 10),
            img_channels=kwargs.get('img_channels', 3),
            img_size=kwargs.get('img_size', 32)
        )
    elif model_type.lower() == 'deepinversion':
        return FedFTGGeneratorDeepInversion(
            img_channels=kwargs.get('img_channels', 3),
            img_size=kwargs.get('img_size', 32),
            num_classes=kwargs.get('num_classes', 10)
        )
    else:
        raise ValueError(f"不支持的生成器类型: {model_type}, 支持的类型: ['standard', 'deepinversion']") 