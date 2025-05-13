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

    return client_list, datasets_dict  # 返回客户端列表和数据集字典


class FemnistDataset(Dataset):
    """
    一个自定义数据集，继承自PyTorch的Dataset类。
    """

    def __init__(self, X, Y):
        """
        初始化数据集。
        :param X: 包含图像数据的数组，每个图像数据应为28x28的二维数组。
        :param Y: 包含图像标签的数组。
        """
        self.X = X
        self.Y = Y

        # 确保X和Y是torch张量
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

        # 如果图像数据是一维平铺的或者二维的，需要转换为四维张量：N x C x H x W
        if self.X.dim() == 2:  # 假设X是二维的，每行是一个平铺的图像
            self.X = self.X.view(-1, 1, 28, 28)  # 转换为N x 1 x 28 x 28的张量
        elif self.X.dim() == 3:  # 假设X已经是N x 28 x 28
            self.X = self.X.unsqueeze(1)  # 添加通道维，使其成为N x 1 x 28 x 28
        
        # 数据归一化 - 将像素值从[0,255]归一化到[-1,1]
        if self.X.max() > 1.0:
            self.X = (self.X / 127.5) - 1.0

    def __len__(self):
        """
        返回数据集中的样本数。
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        根据给定的索引idx返回一个样本。
        """
        return self.X[idx], self.Y[idx]


class MNISTDataset(Dataset):
    """
    自定义数据集类，用于处理 MNIST 数据集。
    """

    def __init__(self, X, Y):
        """
        初始化数据集。
        :param X: 图像数据，N x C x H x W。
        :param Y: 标签数据，N。
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def partition_data_by_dirichlet(train_data, train_labels, test_data, test_labels, client_num, num_classes, beta=0.4, seed=42):
    """
    使用狄利克雷分布创建非IID数据分区。
    
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
    
    # 为每个类别分配数据
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
    
    # 同样处理测试数据，使用相同的类别分布
    test_label_distribution = {}
    for k in range(num_classes):
        test_label_distribution[k] = []
    
    for i, label in enumerate(test_labels):
        test_label_distribution[label].append(i)
    
    client_test_data = [[] for _ in range(client_num)]
    client_test_labels = [[] for _ in range(client_num)]
    
    # 为测试数据使用相同的分布，确保测试数据和训练数据匹配
    for k in range(num_classes):
        idx_k = test_label_distribution[k]
        if len(idx_k) == 0:
            continue
            
        proportions = class_priors[:, k]
        proportions = proportions / proportions.sum()
        
        # 计算每个客户端应该获得的测试样本数量
        num_samples_per_client = (proportions * len(idx_k)).astype(int)
        # 处理因舍入导致的总数不匹配问题
        remainder = len(idx_k) - num_samples_per_client.sum()
        if remainder > 0:
            # 将剩余样本分配给获得样本最少的客户端
            idx_min = np.argmin(num_samples_per_client)
            num_samples_per_client[idx_min] += remainder
        
        # 随机打乱该类别的测试样本
        np.random.shuffle(idx_k)
        
        # 分配测试样本给客户端
        idx_begin = 0
        for client_id in range(client_num):
            samples_to_take = num_samples_per_client[client_id]
            if samples_to_take > 0:  # 确保样本数量为正
                idx_end = idx_begin + samples_to_take
                if idx_end > len(idx_k):  # 防止索引越界
                    idx_end = len(idx_k)
                
                client_test_data[client_id].extend([test_data[idx] for idx in idx_k[idx_begin:idx_end]])
                client_test_labels[client_id].extend([test_labels[idx] for idx in idx_k[idx_begin:idx_end]])
                idx_begin = idx_end
    
    # 打印样本分配统计信息
    print("\n客户端样本数量统计:")
    for client_id in range(client_num):
        print(f"客户端 {client_id}: 训练样本数 = {len(client_train_data[client_id])}, 测试样本数 = {len(client_test_data[client_id])}")
    
    return client_train_data, client_train_labels, client_test_data, client_test_labels


def load_mnist_dataset(client_list, transform=None, partition="noiid", beta=0.4, seed=42):
    """
    加载 MNIST 数据集，并根据指定的划分方式保存数据。
    :param client_list: 客户端列表
    :param transform: 数据预处理转换
    :param partition: 划分方式，"iid"、"noiid"或"dirichlet"
    :param beta: 狄利克雷分布的参数，控制非IID程度（仅当partition="dirichlet"时使用）
    :param seed: 随机种子
    :return: 按客户端划分的训练集和测试集，并保存为 .json 格式
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 定义数据的保存路径
    data_dir = "./data/MNIST/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 预处理：转换为张量
    if transform is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    # 下载 MNIST 数据集
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_data, train_labels = train_dataset.data.numpy(), train_dataset.targets.numpy()
    test_data, test_labels = test_dataset.data.numpy(), test_dataset.targets.numpy()

    # 初始化数据字典
    datasets_dict = {}
    num_clients = len(client_list)

    if partition == "iid":
        # 独立同分布 (IID) 划分
        # 随机打乱训练数据
        indices = np.random.permutation(len(train_data))
        train_data = train_data[indices]
        train_labels = train_labels[indices]
        
        # 均匀划分给每个客户端
        samples_per_client = len(train_data) // num_clients
        
        for i, client in enumerate(client_list):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(train_data)
            
            client_train_data = train_data[start_idx:end_idx]
            client_train_labels = train_labels[start_idx:end_idx]
            
            # 同样处理测试数据
            test_indices = np.random.permutation(len(test_data))
            test_samples_per_client = len(test_data) // num_clients
            test_start_idx = i * test_samples_per_client
            test_end_idx = test_start_idx + test_samples_per_client if i < num_clients - 1 else len(test_data)
            
            client_test_data = test_data[test_indices[test_start_idx:test_end_idx]]
            client_test_labels = test_labels[test_indices[test_start_idx:test_end_idx]]
            
            # 转换为合适的格式
            client_train_data = torch.tensor(client_train_data, dtype=torch.float32).unsqueeze(1)
            client_test_data = torch.tensor(client_test_data, dtype=torch.float32).unsqueeze(1)
            
            datasets_dict[client] = {
                "train_dataset": MNISTDataset(
                    client_train_data.numpy(), client_train_labels
                ),
                "test_dataset": MNISTDataset(
                    client_test_data.numpy(), client_test_labels
                ),
            }
            
    elif partition == "dirichlet":
        # 基于狄利克雷分布的非IID划分
        client_train_data, client_train_labels, client_test_data, client_test_labels = partition_data_by_dirichlet(
            train_data, train_labels, test_data, test_labels, num_clients, 10, beta, seed
        )
        
        for i, client in enumerate(client_list):
            # 转换为合适的格式
            train_data_tensor = torch.tensor(np.array(client_train_data[i]), dtype=torch.float32).unsqueeze(1)
            test_data_tensor = torch.tensor(np.array(client_test_data[i]), dtype=torch.float32).unsqueeze(1)
            
            datasets_dict[client] = {
                "train_dataset": MNISTDataset(
                    train_data_tensor.numpy(), np.array(client_train_labels[i])
                ),
                "test_dataset": MNISTDataset(
                    test_data_tensor.numpy(), np.array(client_test_labels[i])
                ),
            }
        
    elif partition == "noiid":
        # 现有的非IID划分方式
        label_to_data = {i: [] for i in range(10)}
        for i, label in enumerate(train_labels):
            label_to_data[label].append((train_data[i], label))

        for i, client in enumerate(client_list):
            client_train_data = []
            client_train_labels = []
            client_test_data = []
            client_test_labels = []

            for label, data_list in label_to_data.items():
                num_samples_per_label = random.randint(1, len(data_list) // 2)
                selected_train_data = random.sample(data_list, num_samples_per_label)
                for image, label in selected_train_data:
                    client_train_data.append(image)
                    client_train_labels.append(label)

            for label, data_list in label_to_data.items():
                test_data_for_label = [
                    x for x in zip(test_data, test_labels) if x[1] == label
                ]
                num_samples_for_test = len(test_data_for_label) // num_clients
                client_test_data.extend(
                    [
                        x[0]
                        for x in test_data_for_label[
                            i * num_samples_for_test : (i + 1) * num_samples_for_test
                        ]
                    ]
                )
                client_test_labels.extend(
                    [
                        x[1]
                        for x in test_data_for_label[
                            i * num_samples_for_test : (i + 1) * num_samples_for_test
                        ]
                    ]
                )

            client_train_data = np.array(client_train_data)
            client_train_data = torch.tensor(
                client_train_data, dtype=torch.float32
            ).unsqueeze(1)

            client_test_data = np.array(client_test_data)
            client_test_data = torch.tensor(
                client_test_data, dtype=torch.float32
            ).unsqueeze(1)

            datasets_dict[client] = {
                "train_dataset": MNISTDataset(
                    client_train_data.numpy(), np.array(client_train_labels)
                ),
                "test_dataset": MNISTDataset(
                    client_test_data.numpy(), np.array(client_test_labels)
                ),
            }
    else:
        raise ValueError(f"不支持的划分方式: {partition}，请使用 'iid', 'noiid' 或 'dirichlet'")

    
    # 返回包含每个客户端数据的字典
    return datasets_dict
