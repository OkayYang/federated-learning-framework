# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 15:15
# @Describe: 非联邦学习的分布式训练示例 - 每个客户端独立训练自己的模型

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from fl.data import datasets
from fl.model.model import FeMNISTNet


def train_client_model(client_name, train_dataset, test_dataset, epochs=10, batch_size=64):
    """为单个客户端训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = FeMNISTNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    # 用于记录训练过程的指标
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": []
    }
    
    print(f"\n{'='*50}")
    print(f"Training model for client: {client_name}")
    print(f"{'='*50}")
    
    # 开始训练
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"{'-'*30}")
        
        # 训练阶段
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Training {client_name}", unit="batch", leave=False)):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f"Testing {client_name}", unit="batch", leave=False):
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        # 计算测试指标
        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total
        
        history["test_loss"].append(avg_test_loss)
        history["test_accuracy"].append(accuracy)
        
        # 输出当前轮次结果
        print(f"├─ Train Loss: {avg_train_loss:.4f}")
        print(f"├─ Test Loss: {avg_test_loss:.4f}")
        print(f"└─ Test Accuracy: {accuracy:.2%}")
    
    # 保存模型
    torch.save(model.state_dict(), f"{client_name}_model.pth")
    
    return history, model


def setup_and_train_standalone_models():
    """
    设置并独立训练每个客户端的模型（非联邦方式）。

    每个客户端只使用自己的数据集进行训练，没有模型聚合。
    这个函数执行以下步骤：
    - 加载每个客户端的数据集
    - 为每个客户端独立训练一个模型
    - 比较不同客户端的模型性能
    - 可视化每个客户端的训练结果
    """
    # 加载 FEMNIST 数据集
    client_list, dataset_dict = datasets.load_feminist_dataset()
    
    # 存储所有客户端的训练历史
    all_clients_history = {}
    
    # 训练参数
    epochs = 50
    batch_size = 64
    
    # 为每个客户端独立训练模型
    for client_name in client_list:
        train_dataset = dataset_dict[client_name]["train_dataset"]
        test_dataset = dataset_dict[client_name]["test_dataset"]
        
        # 训练此客户端的模型
        history, _ = train_client_model(
            client_name, 
            train_dataset, 
            test_dataset, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # 保存此客户端的历史记录
        all_clients_history[client_name] = history
    
    # 绘制比较结果
    plot_all_clients_metrics(all_clients_history, client_list)
    
    return all_clients_history


def plot_all_clients_metrics(all_history, client_list):
    """绘制所有客户端的训练指标比较"""
    # 使用默认样式而不是seaborn
    plt.figure(figsize=(12, 18))
    
    # 为每个客户端绘制训练损失
    plt.subplot(3, 1, 1)
    for client in client_list:
        plt.plot(all_history[client]["train_loss"], label=f'{client}', marker='o')
    
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 为每个客户端绘制测试损失
    plt.subplot(3, 1, 2)
    for client in client_list:
        plt.plot(all_history[client]["test_loss"], label=f'{client}', marker='s')
    
    plt.title('Test Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 为每个客户端绘制准确率
    plt.subplot(3, 1, 3)
    for client in client_list:
        plt.plot(all_history[client]["test_accuracy"], label=f'{client}', marker='^')
    
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./plots/standalone_clients_comparison.png')
    plt.show()


# 运行独立客户端训练
if __name__ == "__main__":
    setup_and_train_standalone_models() 