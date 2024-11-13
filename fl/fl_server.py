# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:31
# @Describe:
import random

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from fl.aggregation.aggregator import average_weight
from fl.fl_base import ModelConfig
from fl.strategy import create_client


class FLServer:
    def __init__(
        self,
        client_list,
        strategy,
        model_config: ModelConfig,
        client_dataset_dict,
        **kwargs,
    ):
        self.strategy = strategy  # 联邦学习策略（例如 FedAvg, FedProx）
        self.model_config = model_config  # 服务器的全局模型
        self._workers = {
            client: create_client(strategy, client, model_config, client_dataset_dict)
            for client in client_list
        }
        # 初始化历史记录结构，包含每个worker的详细记录
        self.history = {
            "global": {"train_loss": [], "test_accuracy": [], "test_loss": []},
            "workers": {
                client: {"train_loss": [], "accuracy": [], "test_loss": []}
                for client in client_list
            },
        }
        self.kwargs = kwargs

    def initialize_client_weights(self):
        """设置服务器端的全局模型"""
        clients_weights = []
        for worker_name, worker in self._workers.items():
            weights = worker.get_weights(return_numpy=True)
            clients_weights.append(weights)

        return average_weight(clients_weights)

    def fit(self, comm_rounds, ratio_client=1.0):
        """
        进行训练的主要流程，包括与各客户端的交互
        """
        # 迭代训练过程
        global_weight = self.initialize_client_weights()
        for round_num in range(comm_rounds):
            print(f"Starting round {round_num + 1}...")
            # 选择部分客户端进行训练
            selected_workers = {
                client_name: self._workers[client_name]
                for client_name in random.sample(
                    list(self._workers.keys()), int(len(self._workers) * ratio_client)
                )
            }
            client_weight_list = []
            sample_num_list = []
            train_loss_list = []
            accuracy_list = []
            test_loss_list = []

            # 训练
            for client_name, worker in selected_workers.items():  # 客户端训练并记录数据
                client_weight, sample_num, train_loss = worker.local_train(
                    sync_round=round_num, weights=global_weight
                )
                client_weight_list.append(client_weight)
                sample_num_list.append(sample_num)
                # 保存每个worker的训练损失
                train_loss_list.append(train_loss)

                self.history["workers"][client_name]["train_loss"].append(train_loss)

            # 评估
            for client_name, worker in tqdm(
                selected_workers.items(), desc="Evaluate Clients", unit="worker"
            ):
                # 客户端评估并记录数据
                test_acc, test_loss = worker.local_evaluate()

                accuracy_list.append(test_acc)
                test_loss_list.append(test_loss)

                self.history["workers"][client_name]["accuracy"].append(test_acc)
                self.history["workers"][client_name]["test_loss"].append(test_loss)

            # 计算当前轮次的全局平均训练损失、准确率和测试损失
            avg_train_loss = sum(train_loss_list) / len(train_loss_list)
            avg_accuracy = sum(accuracy_list) / len(accuracy_list)
            avg_test_loss = sum(test_loss_list) / len(test_loss_list)

            # 保存全局指标到历史记录中
            self.history["global"]["train_loss"].append(avg_train_loss)
            self.history["global"]["test_accuracy"].append(avg_accuracy)
            self.history["global"]["test_loss"].append(avg_test_loss)

            # 输出当前轮次的结果
            print(f"Train Loss: {avg_train_loss}")
            print(f"Test Accuracy: {avg_accuracy}")
            print(f"Test Loss: {avg_test_loss}")
            print(f"Round {round_num + 1} completed.")

            # 聚合客户端的模型更新
            global_weight = average_weight(client_weight_list, sample_num_list)

        return self.history
