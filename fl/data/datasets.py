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


def load_mnist_dataset(client_list, transform=None, partition="noidd", seed=42):
    """
    加载 MNIST 数据集，并根据指定的划分方式保存数据。
    :param client_list: 客户端列表
    :param transform: 数据预处理转换
    :param partition: 划分方式，"idd" 或 "noidd"
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

    if partition == "idd":
        num_clients = len(client_list)
        label_to_data = {i: [] for i in range(10)}
        for i, label in enumerate(train_labels):
            label_to_data[label].append((train_data[i], label))

        for i, client in enumerate(client_list):
            client_train_data = []
            client_train_labels = []
            client_test_data = []
            client_test_labels = []

            for label, data_list in label_to_data.items():
                train_data_for_label = data_list[
                    i
                    * (len(data_list) // num_clients) : (i + 1)
                    * (len(data_list) // num_clients)
                ]
                for image, label in train_data_for_label:
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

    elif partition == "noidd":
        num_clients = len(client_list)
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

    return datasets_dict
