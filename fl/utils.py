# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 19:55
# @Describe:
import matplotlib.pyplot as plt
import numpy as np


def optim_wrapper(func, *args, **kwargs):
    def wrapped_func(params):
        return func(params, *args, **kwargs)

    return wrapped_func


def plot_global_metrics(history: dict):
    # 绘制全局训练损失、测试准确率和测试损失
    global_history = history["global"]
    epochs = range(1, len(global_history["train_loss"]) + 1)

    plt.figure(figsize=(15, 5))

    # 绘制全局训练损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, global_history["train_loss"], label="Train Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Global Train Loss")
    plt.grid(True)

    # 绘制全局测试准确率
    plt.subplot(1, 3, 2)
    plt.plot(
        epochs, global_history["test_accuracy"], label="Test Accuracy", color="green"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Global Test Accuracy")
    plt.grid(True)

    # 绘制全局测试损失
    plt.subplot(1, 3, 3)
    plt.plot(epochs, global_history["test_loss"], label="Test Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Global Test Loss")
    plt.grid(True)

    # 调整布局，显示所有子图
    plt.tight_layout()
    plt.show()


def plot_worker_metrics(history: dict):
    """
    为每个worker创建一个独立的画板，绘制其训练损失、测试准确率和测试损失
    """
    workers_history = history["workers"]
    epochs = range(1, len(next(iter(workers_history.values()))["train_loss"]) + 1)

    # 遍历每个worker，为每个worker创建一个独立的画板
    for client_name, metrics in workers_history.items():
        # 创建一个新的图表（画板）
        plt.figure(figsize=(12, 5))

        # 绘制训练损失
        plt.subplot(1, 3, 1)
        plt.plot(
            epochs,
            metrics["train_loss"],
            label=f"{client_name} Train Loss",
            color="blue",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{client_name} - Train Loss")
        plt.grid(True)

        # 绘制测试准确率
        plt.subplot(1, 3, 2)
        plt.plot(
            epochs,
            metrics["accuracy"],
            label=f"{client_name} Test Accuracy",
            color="green",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{client_name} - Test Accuracy")
        plt.grid(True)

        # 绘制测试损失
        plt.subplot(1, 3, 3)
        plt.plot(
            epochs, metrics["test_loss"], label=f"{client_name} Test Loss", color="red"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{client_name} - Test Loss")
        plt.grid(True)

        # 调整布局，确保图表不重叠
        plt.tight_layout()

        # 显示该worker的图表
        plt.show()


def plot_client_label_distribution(datasets_dict):
    """
    绘制每个客户端的标签分布。
    :param datasets_dict: 每个客户端的数据字典，包含 'train_dataset' 和 'test_dataset'
    """
    # 遍历客户端的数据字典
    for client, data in datasets_dict.items():
        # 提取训练集标签
        train_labels = data["train_dataset"].Y  # 访问标签时使用 self.Y

        # 统计每个标签的数量
        label_counts = np.bincount(train_labels)

        # 绘制标签分布图
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(10),
            label_counts,
            tick_label=[str(i) for i in range(10)],
            color="skyblue",
        )
        plt.title(f"Client {client} Label Distribution")
        plt.xlabel("Label")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
