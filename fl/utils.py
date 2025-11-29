# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 19:55
# @Describe:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import glob
import seaborn as sns
import torch
import time


# 联邦学习算法颜色字典
ALGORITHM_COLORS = {
    'fedavg': '#1f77b4',     # 蓝色
    'fedprox': '#ff7f0e',    # 橙色
    'moon': '#2ca02c',       # 绿色
    'scaffold': '#e377c2',   # 粉色
    'feddistill': '#ff9896', # 紫色
    'fedgen': '#8c564b',     # 棕色
    'fedspd': '#d62728',     # 红色
    'fedspd-lc': '#9467bd',  # 浅红色
    'fedalone': '#7f7f7f',   # 灰色
    'fedftg': '#17becf',     # 青色
    'fedgkd': '#bcbd22'      # 橄榄绿
}


def get_algorithm_color(algorithm_name):
    """
    根据算法名称获取对应的颜色
    
    Args:
        algorithm_name (str): 算法名称
        
    Returns:
        str: 对应的颜色代码，如果算法不存在则返回黑色
    """
    return ALGORITHM_COLORS.get(algorithm_name.lower(), '#000000')  # 默认黑色


def get_all_algorithm_colors():
    """
    获取所有算法的颜色字典
    
    Returns:
        dict: 算法颜色字典的副本
    """
    return ALGORITHM_COLORS.copy()


def update_model_weights(model, weights):
    """
    更新模型权重，支持numpy数组和PyTorch张量
    
    Args:
        model: PyTorch模型
        weights: 权重（可以是numpy数组或PyTorch张量）
    """
    if isinstance(weights, list):
        # 如果是列表，则假设是numpy数组列表
        state_dict = model.state_dict()
        for k, v in zip(state_dict.keys(), weights):
            if isinstance(v, np.ndarray):
                state_dict[k] = torch.from_numpy(v).to(state_dict[k].device)
            elif isinstance(v, (float, np.float32, np.float64)):
                # 处理标量值
                state_dict[k] = torch.tensor(v, dtype=state_dict[k].dtype, device=state_dict[k].device)
            else:
                # 处理已经是tensor的情况
                state_dict[k] = v.to(state_dict[k].device)
        model.load_state_dict(state_dict)
    elif isinstance(weights, dict):
        # 如果是字典，则直接加载
        model.load_state_dict(weights)
    else:
        raise TypeError("权重必须是列表或字典")


def optim_wrapper(func, *args, **kwargs):
    def wrapped_func(params):
        return func(params, *args, **kwargs)

    return wrapped_func


def scheduler_wrapper(scheduler_type='step', **kwargs):
    """
    学习率调度器包装器
    
    Args:
        scheduler_type: 调度器类型 'step', 'exp', 'cosine'
        **kwargs: 调度器参数
    """
    from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
    
    def wrapped_func(optimizer):
        step_size = kwargs.get('step_size', 5)
        gamma = kwargs.get('gamma', 0.8)
        patience = kwargs.get('patience', 3)
        comm_rounds = kwargs.get('comm_rounds', 50)
        
        if scheduler_type == 'step':
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'exp':
            return ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=comm_rounds, eta_min=1e-6)
        else:
            # 默认使用StepLR
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    return wrapped_func


def plot_global_metrics(history: dict, experiment_name: str):
    # 从experiment_name中提取策略和数据集名称
    parts = experiment_name.split('_')
    strategy = parts[0]
    dataset = parts[1]
    
    # 绘制全局训练损失、测试准确率和测试损失
    global_history = history["global"]
    epochs = range(1, len(global_history["train_loss"]) + 1)

    plt.figure(figsize=(15, 5))

    # 获取策略对应的颜色
    strategy_color = get_algorithm_color(strategy)

    # 绘制全局训练损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, global_history["train_loss"], label="Train Loss", color=strategy_color)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Global Train Loss")
    plt.grid(True)

    # 绘制全局测试准确率
    plt.subplot(1, 3, 2)
    plt.plot(
        epochs, global_history["test_accuracy"], label="Test Accuracy", color=strategy_color
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{strategy} Global Test Accuracy")
    plt.grid(True)

    # 绘制全局测试损失
    plt.subplot(1, 3, 3)
    plt.plot(epochs, global_history["test_loss"], label="Test Loss", color=strategy_color)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Global Test Loss")
    plt.grid(True)

    # 调整布局，显示所有子图
    plt.tight_layout()
    
    # 创建按数据集分类的保存目录
    plots_dir = f"plots/{dataset}"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/fl_global_metrics_{experiment_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_worker_metrics(history: dict, experiment_name: str):
    """
    将所有worker的训练损失、测试准确率和测试损失绘制在同一个图表上，使用插值法使曲线平滑
    额外显示workers测试准确率和测试损失
    """
    # 从experiment_name中提取策略和数据集名称
    parts = experiment_name.split('_')
    strategy = parts[0]
    dataset = parts[1]
    
    workers_history = history["workers"]
    
    # 创建按数据集分类的保存目录
    plots_dir = f"plots/{dataset}"
    os.makedirs(plots_dir, exist_ok=True)

    # 创建一个大的图表，包含五个子图
    plt.figure(figsize=(15, 25))
    
    # 获取全局轮次数量作为x轴
    total_rounds = len(history["global"]["train_loss"])
    global_epochs = np.arange(1, total_rounds + 1)

    # 获取策略对应的颜色
    strategy_color = get_algorithm_color(strategy)

    # 绘制训练损失比较
    plt.subplot(5, 1, 1)
    for client_name, metrics in workers_history.items():
        # 记录客户端参与的轮次和对应的数据
        rounds_participated = []
        values = []
        
        # 收集客户端参与的轮次和对应的训练损失
        for i, loss in enumerate(metrics["train_loss"]):
            rounds_participated.append(i + 1)  # 轮次从1开始
            values.append(loss)
        
        # 如果客户端至少参与了一轮训练
        if rounds_participated:
            # 使用scipy的插值函数
            from scipy import interpolate
            
            # 如果只有一个数据点，则复制该点作为第二个点
            if len(rounds_participated) == 1:
                rounds_participated.append(rounds_participated[0] + 0.1)
                values.append(values[0])
            
            # 创建插值函数
            f = interpolate.interp1d(rounds_participated, values, kind='linear', 
                                     bounds_error=False, fill_value=(values[0], values[-1]))
            
            # 生成插值后的数据
            smooth_values = f(global_epochs)
            
            # 绘制平滑曲线
            plt.plot(
                global_epochs,
                smooth_values,
                label=f"{client_name}",
                color=strategy_color,
                marker='o',
                markersize=3
            )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Training Loss Comparison")
    plt.legend()
    plt.grid(True)

    # 绘制测试准确率比较
    plt.subplot(5, 1, 2)
    for client_name, metrics in workers_history.items():
        # 记录客户端参与的轮次和对应的数据
        rounds_participated = []
        values = []
        
        # 收集客户端参与的轮次和对应的准确率
        for i, acc in enumerate(metrics["local_test_accuracy"]):
            rounds_participated.append(i + 1)  # 轮次从1开始
            values.append(acc)
        
        # 如果客户端至少参与了一轮训练
        if rounds_participated:
            # 使用scipy的插值函数
            from scipy import interpolate
            
            # 如果只有一个数据点，则复制该点作为第二个点
            if len(rounds_participated) == 1:
                rounds_participated.append(rounds_participated[0] + 0.1)
                values.append(values[0])
            
            # 创建插值函数
            f = interpolate.interp1d(rounds_participated, values, kind='linear', 
                                     bounds_error=False, fill_value=(values[0], values[-1]))
            
            # 生成插值后的数据
            smooth_values = f(global_epochs)
            
            # 绘制平滑曲线
            plt.plot(
                global_epochs,
                smooth_values,
                label=f"{client_name}",
                color=strategy_color,
                marker='^',
                markersize=3
            )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{strategy} Local Test Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    # 绘制测试损失比较
    plt.subplot(5, 1, 3)
    for client_name, metrics in workers_history.items():
        # 记录客户端参与的轮次和对应的数据
        rounds_participated = []
        values = []
        
        # 收集客户端参与的轮次和对应的测试损失
        for i, loss in enumerate(metrics["local_test_loss"]):
            rounds_participated.append(i + 1)  # 轮次从1开始
            values.append(loss)
        
        # 如果客户端至少参与了一轮训练
        if rounds_participated:
            # 使用scipy的插值函数
            from scipy import interpolate
            
            # 如果只有一个数据点，则复制该点作为第二个点
            if len(rounds_participated) == 1:
                rounds_participated.append(rounds_participated[0] + 0.1)
                values.append(values[0])
            
            # 创建插值函数
            f = interpolate.interp1d(rounds_participated, values, kind='linear', 
                                     bounds_error=False, fill_value=(values[0], values[-1]))
            
            # 生成插值后的数据
            smooth_values = f(global_epochs)
            
            # 绘制平滑曲线
            plt.plot(
                global_epochs,
                smooth_values,
                label=f"{client_name}",
                color=strategy_color,
                marker='s',
                markersize=3
            )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Local Test Loss Comparison")
    plt.legend()
    plt.grid(True)
    
    # 绘制workers的测试准确率平均值
    plt.subplot(5, 1, 4)
    for client_name, metrics in workers_history.items():
        # 记录客户端参与的轮次和对应的数据
        rounds_participated = []
        values = []
        
        # 收集客户端参与的轮次和对应的测试损失
        for i, loss in enumerate(metrics["global_test_loss"]):
            rounds_participated.append(i + 1)  # 轮次从1开始
            values.append(loss)
        
        # 如果客户端至少参与了一轮训练
        if rounds_participated:
            # 使用scipy的插值函数
            from scipy import interpolate
            
            # 如果只有一个数据点，则复制该点作为第二个点
            if len(rounds_participated) == 1:
                rounds_participated.append(rounds_participated[0] + 0.1)
                values.append(values[0])
            
            # 创建插值函数
            f = interpolate.interp1d(rounds_participated, values, kind='linear', 
                                     bounds_error=False, fill_value=(values[0], values[-1]))
            
            # 生成插值后的数据
            smooth_values = f(global_epochs)
            
            # 绘制平滑曲线
            plt.plot(
                global_epochs,
                smooth_values,
                label=f"{client_name}",
                color=strategy_color,
                marker='s',
                markersize=3
            )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Global Test Loss Comparison")
    plt.legend()
    plt.grid(True)
    
    # 绘制workers的测试损失平均值
    plt.subplot(5, 1, 5)
    for client_name, metrics in workers_history.items():
        # 记录客户端参与的轮次和对应的数据
        rounds_participated = []
        values = []
        
        # 收集客户端参与的轮次和对应的测试损失
        for i, loss in enumerate(metrics["global_test_loss"]):
            rounds_participated.append(i + 1)  # 轮次从1开始
            values.append(loss)
        
        # 如果客户端至少参与了一轮训练
        if rounds_participated:
            # 使用scipy的插值函数
            from scipy import interpolate
            
            # 如果只有一个数据点，则复制该点作为第二个点
            if len(rounds_participated) == 1:
                rounds_participated.append(rounds_participated[0] + 0.1)
                values.append(values[0])
            
            # 创建插值函数
            f = interpolate.interp1d(rounds_participated, values, kind='linear', 
                                     bounds_error=False, fill_value=(values[0], values[-1]))
            
            # 生成插值后的数据
            smooth_values = f(global_epochs)
            
            # 绘制平滑曲线
            plt.plot(
                global_epochs,
                smooth_values,
                label=f"{client_name}",
                color=strategy_color,
                marker='s',
                markersize=3
            )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Global Test Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f"{plots_dir}/fl_clients_comparison_{experiment_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_client_label_distribution(datasets_dict, dataset_name=None):
    """
    绘制两个图表：
    1. 热图：横坐标是客户端ID，纵坐标是标签，颜色表示每个客户端每个标签的样本数量
    2. 条形图：横坐标是客户端ID，纵坐标是每个客户端的样本总数
    
    :param datasets_dict: 每个客户端的数据字典，包含 'train_dataset' 和 'test_dataset'
    :param dataset_name: 数据集名称，用于保存到对应文件夹
    """
    # 创建按数据集分类的保存目录
    if dataset_name:
        plots_dir = f"plots/{dataset_name}"
    else:
        plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 获取客户端ID列表，排除global数据集
    client_ids = [client for client in datasets_dict.keys() if client != "global"]
    num_clients = len(client_ids)
    
    # 确定标签数量范围
    max_label = 0
    for client in client_ids:
        train_labels = datasets_dict[client]["train_dataset"].Y
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
    plt.savefig(f"{plots_dir}/client_label_heatmap.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{plots_dir}/client_sample_count.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_global_test_metrics(history: dict, experiment_name: str):
    """
    绘制在全局测试数据集上的评估结果
    """
    # 从experiment_name中提取策略和数据集名称
    parts = experiment_name.split('_')
    strategy = parts[0]
    dataset = parts[1]
    
    # 检查是否有全局测试数据
    if "global_test" not in history:
        print("没有全局测试数据可供绘图")
        return
    
    global_test = history["global_test"]
    epochs = range(1, len(global_test["accuracy"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制全局测试准确率
    plt.subplot(1, 2, 1)
    plt.plot(epochs, global_test["accuracy"], 'b-o', label="Global Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{strategy} Global Test Accuracy")
    plt.grid(True)
    
    # 绘制全局测试损失
    plt.subplot(1, 2, 2)
    plt.plot(epochs, global_test["loss"], 'r-o', label="Global Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{strategy} Global Test Loss")
    plt.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 创建按数据集分类的保存目录
    plots_dir = f"plots/{dataset}"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/global_test_metrics_{experiment_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

class ClientProgress:
    """
    Context manager for tracking progress, compatible with tqdm.
    If progress_actor is provided, sends updates to Ray actor.
    Otherwise, uses local tqdm.
    """
    def __init__(self, progress_actor=None, client_id=None, total=0, desc=""):
        self.progress_actor = progress_actor
        self.client_id = client_id
        self.total = total
        self.desc = desc
        self.tqdm_bar = None
        self.task_id = f"{client_id}_{time.time()}" if client_id else str(time.time())

    def __enter__(self):
        if self.progress_actor:
            self.progress_actor.add_task.remote(self.task_id, self.total, self.desc)
        else:
            from tqdm import tqdm
            self.tqdm_bar = tqdm(total=self.total, desc=self.desc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tqdm_bar:
            self.tqdm_bar.close()

    def update(self, n=1):
        if self.progress_actor:
            self.progress_actor.update.remote(self.task_id, advance=n)
        elif self.tqdm_bar:
            self.tqdm_bar.update(n)

    def set_postfix(self, metrics):
        if self.progress_actor:
            self.progress_actor.update.remote(self.task_id, metrics=metrics)
        elif self.tqdm_bar:
            self.tqdm_bar.set_postfix(metrics)


