# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 19:10
# @Describe:
import json
import os
import random

import numpy as np
import requests
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def download_data(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded data from {url} to {save_path}")
    else:
        print(
            f"Failed to download data from {url}. Status code: {response.status_code}"
        )


def load_feminist_dataset():
    """
    加载FEMNIST数据集，如果本地不存在则从远程下载

    Returns:
        tuple: (client_list, datasets_dict) - 客户端列表和包含训练测试数据的字典
    """
    import os
    import json

    # 本地存储数据集的路径
    train_json_dir = "data/Femnist/train_data_niid.json"
    test_json_dir = "data/Femnist/test_data_niid.json"

    # 远程服务器上数据集的URL
    train_data_url = "https://cos.ywenrou.cn/dataset/FEMNIST/train_data_niid.json"
    test_data_url = "https://cos.ywenrou.cn/dataset/FEMNIST/test_data_niid.json"

    # 确保数据目录存在
    os.makedirs(os.path.dirname(train_json_dir), exist_ok=True)

    # 检查训练数据集是否在本地存在，如果不存在，则下载
    if not os.path.exists(train_json_dir):
        download_data(train_data_url, train_json_dir)

    # 检查测试数据集是否在本地存在，如果不存在，则下载
    if not os.path.exists(test_json_dir):
        download_data(test_data_url, test_json_dir)

    # 从本地JSON文件加载数据
    with open(train_json_dir, "r") as f:
        train_data = json.load(f)
    with open(test_json_dir, "r") as f:
        test_data = json.load(f)

    datasets_dict = {}
    client_list = []  # 初始化客户端名称列表
    
    # 用于收集全局数据的列表
    all_train_data = []
    all_train_labels = []
    all_test_data = []
    all_test_labels = []

    for user in train_data["users"]:
        user_train_data = train_data["user_data"][user]
        user_test_data = test_data["user_data"][user]

        # 创建训练集和测试集
        train_dataset = FemnistDataset(user_train_data["x"], user_train_data["y"])
        test_dataset = FemnistDataset(user_test_data["x"], user_test_data["y"])

        # 将数据添加到字典中
        datasets_dict[user] = {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
        }
        client_list.append(user)  # 添加用户到客户端列表
        
        # 收集全局数据
        all_train_data.extend(user_train_data["x"])
        all_train_labels.extend(user_train_data["y"])
        all_test_data.extend(user_test_data["x"])
        all_test_labels.extend(user_test_data["y"])
    
    # 创建全局数据集
    global_train_dataset = FemnistDataset(all_train_data, all_train_labels)
    global_test_dataset = FemnistDataset(all_test_data, all_test_labels)
    
    # 将全局数据集添加到字典中
    datasets_dict["global"] = {
        "train_dataset": global_train_dataset,
        "test_dataset": global_test_dataset,
    }
    
    print(f"全局训练数据集大小: {len(global_train_dataset)}")
    print(f"全局测试数据集大小: {len(global_test_dataset)}")

    return client_list, datasets_dict  # 返回客户端列表和数据集字典


class BaseDataset(Dataset):
    """
    基础数据集类，所有自定义数据集类都继承自这个类
    """
    def __init__(self, X, Y, normalize=True):
        """
        初始化数据集
        
        Args:
            X: 特征数据
            Y: 标签数据
            normalize: 是否对数据进行归一化
        """
        self.X = X
        self.Y = Y
        
        # 确保X和Y是torch张量
        if not isinstance(self.X, torch.Tensor):
            # 先确保是numpy数组，避免从列表创建张量导致的警告
            if isinstance(self.X, list):
                self.X = np.array(self.X)
            self.X = torch.tensor(self.X, dtype=torch.float32)
            
        if not isinstance(self.Y, torch.Tensor):
            # 先确保是numpy数组，避免从列表创建张量导致的警告
            if isinstance(self.Y, list):
                self.Y = np.array(self.Y)
            self.Y = torch.tensor(self.Y, dtype=torch.long)
            
        # 数据归一化 - 将像素值从[0,255]归一化到[-1,1]
        if normalize and self.X.max() > 1.0:
            self.X = (self.X / 127.5) - 1.0

    def __len__(self):
        """返回数据集中的样本数"""
        return len(self.X)

    def __getitem__(self, idx):
        """根据给定的索引idx返回一个样本"""
        return self.X[idx], self.Y[idx]


class FemnistDataset(BaseDataset):
    """FEMNIST数据集类"""
    def __init__(self, X, Y):
        super(FemnistDataset, self).__init__(X, Y)
        
        # 确保图像数据形状正确 (N x C x H x W)
        if self.X.dim() == 2:  # 假设X是二维的，每行是一个平铺的图像
            self.X = self.X.view(-1, 1, 28, 28)  # 转换为N x 1 x 28 x 28的张量
        elif self.X.dim() == 3:  # 假设X已经是N x 28 x 28
            self.X = self.X.unsqueeze(1)  # 添加通道维，使其成为N x 1 x 28 x 28


class MNISTDataset(BaseDataset):
    """MNIST数据集类"""
    def __init__(self, X, Y):
        super(MNISTDataset, self).__init__(X, Y)
        
        # 确保图像数据形状正确 (N x C x H x W)
        if self.X.dim() == 3:  # 如果X是N x H x W
            self.X = self.X.unsqueeze(1)  # 添加通道维，使其成为N x 1 x H x W


class CIFAR10Dataset(BaseDataset):
    """CIFAR10数据集类"""
    def __init__(self, X, Y):
        super(CIFAR10Dataset, self).__init__(X, Y)
        # CIFAR10数据已经是正确的格式 (N x 3 x 32 x 32)，无需额外处理


class CIFAR100Dataset(BaseDataset):
    """CIFAR100数据集类"""
    def __init__(self, X, Y):
        super(CIFAR100Dataset, self).__init__(X, Y)
        
        # CIFAR100数据已经是正确的格式 (N x 3 x 32 x 32)，无需额外处理


class TinyImageNetDataset(BaseDataset):
    """Tiny ImageNet数据集类"""
    def __init__(self, X, Y):
        super(TinyImageNetDataset, self).__init__(X, Y)
        
        # Tiny ImageNet数据已经是正确的格式 (N x 3 x 64 x 64)，无需额外处理


class SVHNDataset(BaseDataset):
    """SVHN数据集类"""
    def __init__(self, X, Y):
        super(SVHNDataset, self).__init__(X, Y)
        
        # SVHN数据已经是正确的格式 (N x 3 x 32 x 32)，无需额外处理


def partition_data_by_dirichlet(train_data, train_labels, test_data, test_labels, client_num, num_classes, beta=0.4, seed=42):
    """
    使用狄利克雷分布创建非IID训练数据分区，测试数据按IID方式划分。
    
    Args:
        train_data: 训练数据
        train_labels: 训练标签
        test_data: 测试数据
        test_labels: 测试标签
        client_num: 客户端数量
        num_classes: 类别数量
        beta: 狄利克雷分布参数，控制异质性程度（较小的值表示更高的异质性）
        seed: 随机种子
    
    Returns:
        tuple: 客户端训练数据、标签、测试数据、标签的列表
    """
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    # 创建每个类的样本索引列表
    label_distribution = {}
    for k in range(num_classes):
        label_distribution[k] = []
    for i, label in enumerate(train_labels):
        label_distribution[label].append(i)
    
    # 每个客户端的类别分布由狄利克雷分布确定
    class_priors = np.random.dirichlet(alpha=[beta] * num_classes, size=client_num)
    
    # 初始化每个客户端的训练数据
    client_train_data = [[] for _ in range(client_num)]
    client_train_labels = [[] for _ in range(client_num)]
    
    # 记录每个客户端获得的样本总数
    client_sample_count = np.zeros(client_num)
    
    # 为每个类别分配训练数据
    for k in range(num_classes):
        # 得到该类别的所有样本索引
        idx_k = label_distribution[k]
        
        if len(idx_k) == 0:
            continue  # 跳过没有样本的类别
            
        # 按照狄利克雷分布比例分配
        proportions = class_priors[:, k]
        proportions = proportions / proportions.sum()
        
        # 确保每个客户端至少分配到最小数量的样本
        min_samples = 5  # 每个客户端在每个类别至少分配5个样本(如果有足够样本)
        if len(idx_k) >= client_num * min_samples:
            # 计算剩余样本的比例分配
            remaining_samples = len(idx_k) - client_num * min_samples
            # 先计算按比例分配的样本数
            extra_samples = (proportions * remaining_samples).astype(int)
            # 处理因舍入导致的总数不匹配问题
            extra_samples[-1] = remaining_samples - extra_samples[:-1].sum()
            # 最终每个客户端获得的样本数 = 最小保证样本数 + 按比例分配的额外样本数
            num_samples_per_client = min_samples + extra_samples
        else:
            # 如果样本不足以保证每个客户端的最小数量，则按原比例分配
            num_samples_per_client = (proportions * len(idx_k)).astype(int)
            # 处理因舍入导致的样本总数不匹配问题
            remainder = len(idx_k) - num_samples_per_client.sum()
            if remainder > 0:
                # 将剩余样本分配给获得样本最少的客户端
                idx_min = np.argmin(num_samples_per_client)
                num_samples_per_client[idx_min] += remainder
        
        # 随机打乱该类别的样本
        np.random.shuffle(idx_k)
        
        # 分配样本给客户端
        idx_begin = 0
        for client_id in range(client_num):
            samples_to_take = num_samples_per_client[client_id]
            if samples_to_take > 0:  # 确保样本数量为正
                idx_end = idx_begin + samples_to_take
                if idx_end > len(idx_k):  # 防止索引越界
                    idx_end = len(idx_k)
                
                client_train_data[client_id].extend([train_data[idx] for idx in idx_k[idx_begin:idx_end]])
                client_train_labels[client_id].extend([train_labels[idx] for idx in idx_k[idx_begin:idx_end]])
                client_sample_count[client_id] += idx_end - idx_begin
                idx_begin = idx_end
    
    # 确保每个客户端至少有一定数量的样本
    min_total_samples = 100  # 每个客户端至少需要100个样本
    for client_id in range(client_num):
        if client_sample_count[client_id] < min_total_samples:
            # 找出样本数量最多的客户端
            max_client_id = np.argmax(client_sample_count)
            # 从样本最多的客户端转移样本
            samples_to_transfer = min(int(client_sample_count[max_client_id] * 0.1), min_total_samples - int(client_sample_count[client_id]))
            
            if samples_to_transfer > 0 and samples_to_transfer < len(client_train_data[max_client_id]):
                # 随机选择要转移的样本索引
                transfer_indices = np.random.choice(len(client_train_data[max_client_id]), samples_to_transfer, replace=False)
                
                # 转移样本
                for idx in transfer_indices:
                    client_train_data[client_id].append(client_train_data[max_client_id][idx])
                    client_train_labels[client_id].append(client_train_labels[max_client_id][idx])
                
                # 从原客户端删除已转移的样本
                # 创建一个布尔掩码，标记要保留的样本(不在transfer_indices中的)
                mask = np.ones(len(client_train_data[max_client_id]), dtype=bool)
                mask[transfer_indices] = False
                
                client_train_data[max_client_id] = [client_train_data[max_client_id][i] for i in range(len(mask)) if mask[i]]
                client_train_labels[max_client_id] = [client_train_labels[max_client_id][i] for i in range(len(mask)) if mask[i]]
                
                # 更新样本计数
                client_sample_count[client_id] += samples_to_transfer
                client_sample_count[max_client_id] -= samples_to_transfer
    
    # 测试数据按IID方式划分，确保公平评估
    client_test_data, client_test_labels = partition_test_data_iid(test_data, test_labels, client_num, seed)
    
    # 打印样本分配统计信息
    print("\n客户端样本数量统计:")
    for client_id in range(client_num):
        print(f"客户端 {client_id}: 训练样本数 = {len(client_train_data[client_id])}, 测试样本数 = {len(client_test_data[client_id])}")
    
    return client_train_data, client_train_labels, client_test_data, client_test_labels


def init_dataset_loading(data_dir, transform=None, seed=42):
    """
    初始化数据集加载的公共部分
    
    Args:
        data_dir: 数据目录路径
        transform: 数据转换函数
        seed: 随机种子
        
    Returns:
        创建好的目录路径
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 定义数据的保存路径
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    return data_dir


def partition_test_data_iid(test_data, test_labels, client_num, seed=42):
    """
    按独立同分布(IID)方式划分测试数据
    
    Args:
        test_data: 测试数据
        test_labels: 测试标签
        client_num: 客户端数量
        seed: 随机种子
        
    Returns:
        tuple: 客户端测试数据、标签的列表
    """
    # 设置随机种子
    np.random.seed(seed + 1000)  # 使用不同的种子避免与训练数据冲突
    random.seed(seed + 1000)
    
    # 确保数据是numpy数组
    if not isinstance(test_data, np.ndarray):
        test_data = np.array(test_data)
    if not isinstance(test_labels, np.ndarray):
        test_labels = np.array(test_labels)
    
    # 随机打乱测试数据
    test_indices = np.random.permutation(len(test_data))
    test_data = test_data[test_indices]
    test_labels = test_labels[test_indices]
    
    # 初始化客户端数据
    client_test_data = []
    client_test_labels = []
    
    # 均匀划分测试数据给每个客户端
    test_samples_per_client = len(test_data) // client_num
    for i in range(client_num):
        start_idx = i * test_samples_per_client
        end_idx = start_idx + test_samples_per_client if i < client_num - 1 else len(test_data)
        
        client_test_data.append(test_data[start_idx:end_idx])
        client_test_labels.append(test_labels[start_idx:end_idx])
    
    return client_test_data, client_test_labels


def partition_data_iid(train_data, train_labels, test_data, test_labels, client_num, seed=42):
    """
    按独立同分布(IID)方式划分数据
    
    Args:
        train_data: 训练数据
        train_labels: 训练标签
        test_data: 测试数据
        test_labels: 测试标签
        client_num: 客户端数量
        seed: 随机种子
        
    Returns:
        tuple: 客户端训练数据、标签、测试数据、标签的列表
    """
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    # 确保数据是numpy数组
    if not isinstance(train_data, np.ndarray):
        train_data = np.array(train_data)
    if not isinstance(train_labels, np.ndarray):
        train_labels = np.array(train_labels)
    
    # 随机打乱训练数据
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    
    # 初始化客户端数据
    client_train_data = []
    client_train_labels = []
    
    # 均匀划分训练数据给每个客户端
    samples_per_client = len(train_data) // client_num
    for i in range(client_num):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < client_num - 1 else len(train_data)
        
        client_train_data.append(train_data[start_idx:end_idx])
        client_train_labels.append(train_labels[start_idx:end_idx])
    
    # 测试数据按IID方式划分
    client_test_data, client_test_labels = partition_test_data_iid(test_data, test_labels, client_num, seed)
    
    return client_train_data, client_train_labels, client_test_data, client_test_labels


def partition_data_noiid(train_data, train_labels, test_data, test_labels, client_num, num_classes, seed=42):
    """
    按非独立同分布(Non-IID)方式划分训练数据，测试数据按IID方式划分
    
    Args:
        train_data: 训练数据
        train_labels: 训练标签
        test_data: 测试数据
        test_labels: 测试标签
        client_num: 客户端数量
        num_classes: 类别数量
        seed: 随机种子
        
    Returns:
        tuple: 客户端训练数据、标签、测试数据、标签的列表
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 确保数据是numpy数组
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    
    # 按类别整理数据索引，而不是实际数据
    label_to_indices = {i: [] for i in range(num_classes)}
    for i, label in enumerate(train_labels):
        label_to_indices[label].append(i)
    
    # 初始化客户端数据
    client_train_indices = [[] for _ in range(client_num)]
    client_train_data = []
    client_train_labels = []
    
    # 为每个客户端分配不均衡的类别样本
    for client_id in range(client_num):
        for label, indices in label_to_indices.items():
            if len(indices) > 0:
                num_samples_per_label = random.randint(1, max(1, len(indices) // 2))
                selected_indices = random.sample(indices, min(num_samples_per_label, len(indices)))
                client_train_indices[client_id].extend(selected_indices)
    
    # 根据索引提取实际数据
    for client_id in range(client_num):
        indices = client_train_indices[client_id]
        client_train_data.append(train_data[indices])
        client_train_labels.append(train_labels[indices])
    
    # 测试数据按IID方式划分，确保公平评估
    client_test_data, client_test_labels = partition_test_data_iid(test_data, test_labels, client_num, seed)
    
    return client_train_data, client_train_labels, client_test_data, client_test_labels


def partition_dataset(train_data, train_labels, test_data, test_labels, client_list, partition, dataset_class, num_classes, beta=0.4, seed=42):
    """
    统一数据集划分函数，根据指定的划分方式划分数据
    
    Args:
        train_data: 训练数据
        train_labels: 训练标签
        test_data: 测试数据
        test_labels: 测试标签
        client_list: 客户端列表
        partition: 划分方式 ("iid", "noiid", "dirichlet")
        dataset_class: 数据集类 (MNISTDataset, CIFAR10Dataset等)
        num_classes: 类别数量
        beta: 狄利克雷分布参数 (仅用于dirichlet划分)
        seed: 随机种子
        
    Returns:
        dict: 客户端数据集字典，包含全局训练和测试数据集
    """
    # 确保数据是numpy数组格式
    if isinstance(train_data, list):
        train_data = np.array(train_data)
    if isinstance(train_labels, list):
        train_labels = np.array(train_labels)
    if isinstance(test_data, list):
        test_data = np.array(test_data)
    if isinstance(test_labels, list):
        test_labels = np.array(test_labels)
        
    num_clients = len(client_list)
    datasets_dict = {}
    
    # 根据划分方式选择合适的划分函数
    if partition == "iid":
        client_train_data, client_train_labels, client_test_data, client_test_labels = partition_data_iid(
            train_data, train_labels, test_data, test_labels, num_clients, seed
        )
    elif partition == "noiid":
        client_train_data, client_train_labels, client_test_data, client_test_labels = partition_data_noiid(
            train_data, train_labels, test_data, test_labels, num_clients, num_classes, seed
        )
    elif partition == "dirichlet":
        client_train_data, client_train_labels, client_test_data, client_test_labels = partition_data_by_dirichlet(
            train_data, train_labels, test_data, test_labels, num_clients, num_classes, beta, seed
        )
    else:
        raise ValueError(f"不支持的划分方式: {partition}，请使用 'iid', 'noiid' 或 'dirichlet'")
    
    # 为每个客户端创建数据集
    for i, client in enumerate(client_list):
        # 创建训练集和测试集
        train_dataset = dataset_class(client_train_data[i], client_train_labels[i])
        test_dataset = dataset_class(client_test_data[i], client_test_labels[i])
        
        # 添加到数据集字典
        datasets_dict[client] = {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
        }
    
    # 添加全局训练和测试数据集
    global_train_dataset = dataset_class(train_data, train_labels)
    global_test_dataset = dataset_class(test_data, test_labels)
    
    # 将全局数据集添加到字典中
    datasets_dict["global"] = {
        "train_dataset": global_train_dataset,
        "test_dataset": global_test_dataset,
    }
    
    return datasets_dict


def load_mnist_dataset(client_list, transform=None, partition="noiid", beta=0.4, seed=42, data_fraction=1.0):
    """
    加载 MNIST 数据集，并根据指定的划分方式分发给客户端。
    
    Args:
        client_list: 客户端列表
        transform: 数据预处理转换
        partition: 划分方式，"iid"、"noiid"或"dirichlet"
        beta: 狄利克雷分布的参数，控制非IID程度（仅当partition="dirichlet"时使用）
        seed: 随机种子
        data_fraction: 使用的数据比例，范围(0,1]，1表示全量数据
        
    Returns:
        按客户端划分的训练集和测试集字典
    """
    # 初始化数据集加载
    data_dir = init_dataset_loading("./data/MNIST/", transform, seed)

    # 预处理：转换为张量 - 仅用于参考，我们将使用原始数据
    if transform is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    # 下载 MNIST 数据集
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=None
    )

    # 直接使用numpy数组
    train_data = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()
    test_data = test_dataset.data.numpy()
    test_labels = test_dataset.targets.numpy()
    
    # 控制数据量
    if data_fraction < 1.0:
        # 设置随机种子确保可重复性
        np.random.seed(seed)
        
        # 计算保留的样本数
        train_samples = int(len(train_data) * data_fraction)
        test_samples = int(len(test_data) * data_fraction)
        
        # 随机选择样本
        train_indices = np.random.choice(len(train_data), train_samples, replace=False)
        test_indices = np.random.choice(len(test_data), test_samples, replace=False)
        
        # 筛选数据
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]
        
        print(f"使用 {data_fraction:.2f} 比例的MNIST数据: {len(train_data)}个训练样本, {len(test_data)}个测试样本")
    
    # 添加通道维度并转换为float32
    train_data = train_data.reshape(-1, 1, 28, 28).astype(np.float32)
    test_data = test_data.reshape(-1, 1, 28, 28).astype(np.float32)
    
    # 归一化
    train_data = train_data / 127.5 - 1.0
    test_data = test_data / 127.5 - 1.0
    
    # 使用统一的划分函数处理数据
    return partition_dataset(
        train_data, train_labels, test_data, test_labels, 
        client_list, partition, MNISTDataset, 
        num_classes=10, beta=beta, seed=seed
    )


def load_cifar10_dataset(client_list, transform=None, partition="noiid", beta=0.4, seed=42, data_fraction=1.0):
    """
    加载 CIFAR10 数据集，并根据指定的划分方式分发给客户端。
    
    Args:
        client_list: 客户端列表
        transform: 数据预处理转换
        partition: 划分方式，"iid"、"noiid"或"dirichlet"
        beta: 狄利克雷分布的参数，控制非IID程度（仅当partition="dirichlet"时使用）
        seed: 随机种子
        data_fraction: 使用的数据比例，范围(0,1]，1表示全量数据
        
    Returns:
        按客户端划分的训练集和测试集字典
    """
    # 初始化数据集加载
    data_dir = init_dataset_loading("./data/CIFAR10/", transform, seed)

    # 预处理：转换为张量
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # 下载 CIFAR10 数据集
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=None
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=None
    )

    # 直接使用numpy数组而不是列表收集数据
    train_data = train_dataset.data  # 已经是numpy数组格式 (N, 32, 32, 3)
    train_labels = np.array(train_dataset.targets)
    test_data = test_dataset.data  # 已经是numpy数组格式 (N, 32, 32, 3)
    test_labels = np.array(test_dataset.targets)
    
    # 控制数据量
    if data_fraction < 1.0:
        # 设置随机种子确保可重复性
        np.random.seed(seed)
        
        # 计算保留的样本数
        train_samples = int(len(train_data) * data_fraction)
        test_samples = int(len(test_data) * data_fraction)
        
        # 随机选择样本
        train_indices = np.random.choice(len(train_data), train_samples, replace=False)
        test_indices = np.random.choice(len(test_data), test_samples, replace=False)
        
        # 筛选数据
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]
        
        print(f"使用 {data_fraction:.2f} 比例的CIFAR10数据: {len(train_data)}个训练样本, {len(test_data)}个测试样本")
    
    # CIFAR10图像转换 - 从(N, 32, 32, 3)变换为(N, 3, 32, 32)
    train_data = np.transpose(train_data, (0, 3, 1, 2)).astype(np.float32)
    test_data = np.transpose(test_data, (0, 3, 1, 2)).astype(np.float32)
    
    # 归一化
    train_data = train_data / 127.5 - 1.0
    test_data = test_data / 127.5 - 1.0
    
    # 使用统一的划分函数处理数据
    return partition_dataset(
        train_data, train_labels, test_data, test_labels, 
        client_list, partition, CIFAR10Dataset, 
        num_classes=10, beta=beta, seed=seed
    )


def load_cifar100_dataset(client_list, transform=None, partition="noiid", beta=0.4, seed=42, data_fraction=1.0):
    """
    加载 CIFAR100 数据集，并根据指定的划分方式分发给客户端。
    
    Args:
        client_list: 客户端列表
        transform: 数据预处理转换
        partition: 划分方式，"iid"、"noiid"或"dirichlet"
        beta: 狄利克雷分布的参数，控制非IID程度（仅当partition="dirichlet"时使用）
        seed: 随机种子
        data_fraction: 使用的数据比例，范围(0,1]，1表示全量数据
        
    Returns:
        按客户端划分的训练集和测试集字典
    """
    # 初始化数据集加载
    data_dir = init_dataset_loading("./data/CIFAR100/", transform, seed)

    # 预处理：转换为张量 - 仅用于参考，我们将使用原始数据
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # 下载 CIFAR100 数据集
    train_dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=None
    )
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=None
    )

    # 直接使用numpy数组
    train_data = train_dataset.data  # 已经是numpy数组格式 (N, 32, 32, 3)
    train_labels = np.array(train_dataset.targets)
    test_data = test_dataset.data  # 已经是numpy数组格式 (N, 32, 32, 3)
    test_labels = np.array(test_dataset.targets)
    
    # 控制数据量
    if data_fraction < 1.0:
        # 设置随机种子确保可重复性
        np.random.seed(seed)
        
        # 计算保留的样本数
        train_samples = int(len(train_data) * data_fraction)
        test_samples = int(len(test_data) * data_fraction)
        
        # 随机选择样本
        train_indices = np.random.choice(len(train_data), train_samples, replace=False)
        test_indices = np.random.choice(len(test_data), test_samples, replace=False)
        
        # 筛选数据
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]
        
        print(f"使用 {data_fraction:.2f} 比例的CIFAR100数据: {len(train_data)}个训练样本, {len(test_data)}个测试样本")
    
    # CIFAR100图像转换 - 从(N, 32, 32, 3)变换为(N, 3, 32, 32)
    train_data = np.transpose(train_data, (0, 3, 1, 2)).astype(np.float32)
    test_data = np.transpose(test_data, (0, 3, 1, 2)).astype(np.float32)
    
    # 归一化
    train_data = train_data / 127.5 - 1.0
    test_data = test_data / 127.5 - 1.0
    
    # 使用统一的划分函数处理数据
    return partition_dataset(
        train_data, train_labels, test_data, test_labels, 
        client_list, partition, CIFAR100Dataset, 
        num_classes=100, beta=beta, seed=seed
    )


def load_tinyimagenet_dataset(client_list, transform=None, partition="noiid", beta=0.4, seed=42, data_fraction=1.0):
    """
    加载 Tiny ImageNet 数据集，并根据指定的划分方式分发给客户端。
    
    Tiny ImageNet 包含200个类别，每个类别500个训练图像和50个验证图像，
    图像尺寸为64x64x3。
    
    Args:
        client_list: 客户端列表
        transform: 数据预处理转换
        partition: 划分方式，"iid"、"noiid"或"dirichlet"
        beta: 狄利克雷分布的参数，控制非IID程度（仅当partition="dirichlet"时使用）
        seed: 随机种子
        data_fraction: 使用的数据比例，范围(0,1]，1表示全量数据
        
    Returns:
        按客户端划分的训练集和测试集字典
    """
    # 初始化数据集加载
    data_dir = init_dataset_loading("./data/tiny-imagenet-200/", transform, seed)
    
    # 检查数据集是否已下载，如果没有则自动下载
    if not os.path.exists(os.path.join(data_dir, 'train')):
        import subprocess
        import sys
        
        print("Tiny ImageNet 数据集未找到，正在自动下载...")
        
        # 确保data目录存在
        os.makedirs("./data", exist_ok=True)
        
        # 临时zip文件路径
        zip_path = "./data/tiny-imagenet-200.zip"
        
        # 下载数据集
        try:
            print("正在下载 Tiny ImageNet 数据集...")
            download_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            
            # 使用requests下载
            import requests
            response = requests.get(download_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with open(zip_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    # 简单的进度显示
                    sys.stdout.write(f"\r下载进度: {os.path.getsize(zip_path)/total_size*100:.1f}% " if total_size > 0 else "\r下载中...")
                    sys.stdout.flush()
            
            print("\n下载完成，正在解压...")
            
            # 解压数据集
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("./data/")
            
            print("解压完成")
            
            # 可选：删除zip文件
            os.remove(zip_path)
            
        except Exception as e:
            print(f"下载或解压过程中出错: {e}")
            print("请手动下载并解压数据集:")
            print("1. 下载: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            print("2. 解压: unzip tiny-imagenet-200.zip -d ./data/")
            return None
    
    # 预处理：转换为张量
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    # 加载数据集
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    # 类别名称到索引的映射
    class_to_idx = {}
    for i, class_dir in enumerate(sorted(os.listdir(os.path.join(data_dir, 'train')))):
        class_to_idx[class_dir] = i
    
    # 加载训练数据
    class_samples = {}  # 记录每个类别加载的样本数
    
    # 控制每个类别的样本数
    max_samples_per_class = None
    if data_fraction < 1.0:
        # 每个类别默认有500个训练样本
        max_samples_per_class = int(500 * data_fraction)
        print(f"每个类别最多加载 {max_samples_per_class} 个训练样本 (总共 {max_samples_per_class * len(class_to_idx)} 个)")
    
    for class_dir in os.listdir(os.path.join(data_dir, 'train')):
        class_idx = class_to_idx[class_dir]
        img_dir = os.path.join(data_dir, 'train', class_dir, 'images')
        class_samples[class_idx] = 0
        
        if os.path.isdir(img_dir):
            # 获取所有图像文件
            img_files = [f for f in os.listdir(img_dir) if f.endswith('.JPEG')]
            
            # 如果需要限制数据量，随机选择部分图像
            if max_samples_per_class is not None and len(img_files) > max_samples_per_class:
                np.random.seed(seed + class_idx)  # 为每个类别使用不同的种子
                img_files = np.random.choice(img_files, max_samples_per_class, replace=False)
            
            for img_file in img_files:
                img_path = os.path.join(img_dir, img_file)
                try:
                    # 使用PIL读取图像并转换为numpy数组
                    from PIL import Image
                    img = Image.open(img_path).convert('RGB')
                    img = np.array(img.resize((64, 64)), dtype=np.float32)
                    
                    # 转换为CHW格式并归一化
                    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                    img = img / 127.5 - 1.0
                    
                    train_data.append(img)
                    train_labels.append(class_idx)
                    class_samples[class_idx] += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
    
    # 加载验证数据 (用作测试集)
    val_annotations_path = os.path.join(data_dir, 'val', 'val_annotations.txt')
    if os.path.exists(val_annotations_path):
        # 为每个类别记录测试样本数
        test_class_samples = {k: 0 for k in class_to_idx.values()}
        max_test_samples_per_class = None
        
        if data_fraction < 1.0:
            # 每个类别默认有50个验证样本
            max_test_samples_per_class = int(50 * data_fraction)
        
        # 读取验证集标注
        val_img_to_class = {}
        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                val_img_to_class[parts[0]] = parts[1]
        
        # 按类别组织验证图像
        class_val_images = {k: [] for k in class_to_idx.keys()}
        for img_file, class_name in val_img_to_class.items():
            if class_name in class_to_idx:
                class_val_images[class_name].append(img_file)
        
        # 加载每个类别的部分验证图像
        val_img_dir = os.path.join(data_dir, 'val', 'images')
        for class_name, img_files in class_val_images.items():
            class_idx = class_to_idx[class_name]
            
            # 如果需要限制数据量，随机选择部分图像
            if max_test_samples_per_class is not None and len(img_files) > max_test_samples_per_class:
                np.random.seed(seed + class_idx + 10000)  # 为每个类别使用不同的种子
                img_files = np.random.choice(img_files, max_test_samples_per_class, replace=False)
            
            for img_file in img_files:
                img_path = os.path.join(val_img_dir, img_file)
                try:
                    # 使用PIL读取图像并转换为numpy数组
                    from PIL import Image
                    img = Image.open(img_path).convert('RGB')
                    img = np.array(img.resize((64, 64)), dtype=np.float32)
                    
                    # 转换为CHW格式并归一化
                    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                    img = img / 127.5 - 1.0
                    
                    test_data.append(img)
                    test_labels.append(class_idx)
                    test_class_samples[class_idx] += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
    
    # 转换为numpy数组
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_data = np.array(test_data, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int64)
    
    print(f"加载了 {len(train_data)} 个训练样本，{len(test_data)} 个测试样本")
    if data_fraction < 1.0:
        print(f"使用 {data_fraction:.2f} 比例的TinyImageNet数据")
    
    # 使用统一的划分函数处理数据
    return partition_dataset(
        train_data, train_labels, test_data, test_labels, 
        client_list, partition, TinyImageNetDataset, 
        num_classes=200, beta=beta, seed=seed
    )


def load_svhn_dataset(client_list, transform=None, partition="noiid", beta=0.4, seed=42, data_fraction=1.0):
    """
    加载 SVHN (Street View House Numbers) 数据集，并根据指定的划分方式分发给客户端。
    
    SVHN 包含10个类别的彩色数字图像，图像尺寸为32x32x3。
    
    Args:
        client_list: 客户端列表
        transform: 数据预处理转换
        partition: 划分方式，"iid"、"noiid"或"dirichlet"
        beta: 狄利克雷分布的参数，控制非IID程度（仅当partition="dirichlet"时使用）
        seed: 随机种子
        data_fraction: 使用的数据比例，范围(0,1]，1表示全量数据
        
    Returns:
        按客户端划分的训练集和测试集字典
    """
    # 初始化数据集加载
    data_dir = init_dataset_loading("./data/SVHN/", transform, seed)

    # 预处理：转换为张量 - 仅用于参考，我们将使用原始数据
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # 下载 SVHN 数据集
    from torchvision.datasets import SVHN
    train_dataset = SVHN(
        root="./data", split='train', download=True, transform=None
    )
    test_dataset = SVHN(
        root="./data", split='test', download=True, transform=None
    )

    # 获取数据
    train_data = train_dataset.data  # 已经是numpy数组格式 (N, 3, 32, 32)
    train_labels = train_dataset.labels
    test_data = test_dataset.data  # 已经是numpy数组格式 (N, 3, 32, 32)
    test_labels = test_dataset.labels
    
    # 控制数据量
    if data_fraction < 1.0:
        # 设置随机种子确保可重复性
        np.random.seed(seed)
        
        # 计算保留的样本数
        train_samples = int(len(train_data) * data_fraction)
        test_samples = int(len(test_data) * data_fraction)
        
        # 随机选择样本
        train_indices = np.random.choice(len(train_data), train_samples, replace=False)
        test_indices = np.random.choice(len(test_data), test_samples, replace=False)
        
        # 筛选数据
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]
        
        print(f"使用 {data_fraction:.2f} 比例的SVHN数据: {len(train_data)}个训练样本, {len(test_data)}个测试样本")
    
    # 确保数据类型是float32
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    
    # 归一化
    train_data = train_data / 127.5 - 1.0
    test_data = test_data / 127.5 - 1.0
    
    # 使用统一的划分函数处理数据
    return partition_dataset(
        train_data, train_labels, test_data, test_labels, 
        client_list, partition, SVHNDataset, 
        num_classes=10, beta=beta, seed=seed
    )