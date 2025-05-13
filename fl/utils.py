# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 19:55
# @Describe:
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import glob
import seaborn as sns


def optim_wrapper(func, *args, **kwargs):
    def wrapped_func(params):
        return func(params, *args, **kwargs)

    return wrapped_func


def plot_global_metrics(history: dict, strategy: str):
    # 绘制全局训练损失、测试准确率和测试损失
    global_history = history["global"]
    epochs = range(1, len(global_history["train_loss"]) + 1)

    plt.figure(figsize=(15, 5))

    # 绘制全局训练损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, global_history["train_loss"], label="Train Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Global Train Loss")
    plt.grid(True)

    # 绘制全局测试准确率
    plt.subplot(1, 3, 2)
    plt.plot(
        epochs, global_history["test_accuracy"], label="Test Accuracy", color="green"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{strategy} Global Test Accuracy")
    plt.grid(True)

    # 绘制全局测试损失
    plt.subplot(1, 3, 3)
    plt.plot(epochs, global_history["test_loss"], label="Test Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Global Test Loss")
    plt.grid(True)

    # 调整布局，显示所有子图
    plt.tight_layout()
    
    # 创建保存目录
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/fl_global_metrics_{strategy}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_worker_metrics(history: dict, strategy: str):
    """
    将所有worker的训练损失、测试准确率和测试损失绘制在同一个图表上
    """
    workers_history = history["workers"]
    epochs = range(1, len(next(iter(workers_history.values()))["train_loss"]) + 1)

    # 创建保存目录
    os.makedirs("plots", exist_ok=True)

    # 创建一个大的图表，包含三个子图
    plt.figure(figsize=(15, 15))

    # 绘制训练损失比较
    plt.subplot(3, 1, 1)
    for client_name, metrics in workers_history.items():
        plt.plot(
            epochs,
            metrics["train_loss"],
            label=f"{client_name}",
            marker='o',
            markersize=3
        )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Training Loss Comparison")
    plt.legend()
    plt.grid(True)

    # 绘制测试准确率比较
    plt.subplot(3, 1, 2)
    for client_name, metrics in workers_history.items():
        plt.plot(
            epochs,
            metrics["accuracy"],
            label=f"{client_name}",
            marker='^',
            markersize=3
        )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{strategy} Test Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    # 绘制测试损失比较
    plt.subplot(3, 1, 3)
    for client_name, metrics in workers_history.items():
        plt.plot(
            epochs,
            metrics["test_loss"],
            label=f"{client_name}",
            marker='s',
            markersize=3
        )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Test Loss Comparison")
    plt.legend()
    plt.grid(True)

    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f"plots/fl_clients_comparison_{strategy}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_client_label_distribution(datasets_dict):
    """
    绘制两个图表：
    1. 热图：横坐标是客户端ID，纵坐标是标签，颜色表示每个客户端每个标签的样本数量
    2. 条形图：横坐标是客户端ID，纵坐标是每个客户端的样本总数
    
    :param datasets_dict: 每个客户端的数据字典，包含 'train_dataset' 和 'test_dataset'
    """
    # 创建保存目录
    os.makedirs("plots", exist_ok=True)
    
    # 获取客户端ID列表
    client_ids = list(datasets_dict.keys())
    num_clients = len(client_ids)
    
    # 确定标签数量范围
    max_label = 0
    for client, data in datasets_dict.items():
        train_labels = data["train_dataset"].Y
        max_label = max(max_label, train_labels.max().item())
    
    num_labels = max_label + 1
    
    # 创建一个多维数组来存储每个客户端的标签数量
    label_distribution = np.zeros((num_labels, num_clients))
    
    # 统计每个客户端的标签分布
    for i, client in enumerate(client_ids):
        train_labels = datasets_dict[client]["train_dataset"].Y
        counts = np.bincount(train_labels, minlength=num_labels)
        # 转置矩阵，使客户端在横轴，标签在纵轴
        label_distribution[:, i] = counts
    
    # 1. 热图：客户端为横坐标，标签为纵坐标
    plt.figure(figsize=(10, 8))
    
    # 设置标签步长，根据标签数量自动调整
    if num_labels > 20:
        step = max(1, num_labels // 20)  # 如果标签超过20个，则设置步长
        y_labels = [f'{i}' if i % step == 0 else '' for i in range(num_labels)]
    else:
        y_labels = [f'{i}' for i in range(num_labels)]
    
    # 绘制热图，不显示具体数值(annot=False)
    sns.heatmap(label_distribution, annot=False, fmt='g', 
                yticklabels=y_labels,
                xticklabels=client_ids, cmap='YlGnBu')
    plt.title('Label Distribution Across Clients')
    plt.xlabel('Client ID')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.savefig("plots/client_label_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 条形图：显示每个客户端的样本总数
    plt.figure(figsize=(12, 6))
    
    # 计算每个客户端的样本总数
    client_sample_counts = np.sum(label_distribution, axis=0)
    
    # 绘制条形图
    plt.bar(client_ids, client_sample_counts, color='skyblue')
    plt.title('Total Sample Count per Client')
    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在每个条形上方显示具体数值
    for i, count in enumerate(client_sample_counts):
        plt.text(i, count + 0.5, str(int(count)), ha='center')
    
    plt.tight_layout()
    plt.savefig("plots/client_sample_count.png", dpi=300, bbox_inches='tight')
    plt.show()


