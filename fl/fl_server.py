# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:31
# @Describe:
import random

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

from fl.aggregation.aggregator import average_logits, average_scaffold_parameter_c, average_weight
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
            client: create_client(strategy, client, model_config, client_dataset_dict, **kwargs)
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
        
        # 初始化SCAFFOLD全局控制变量
        global_c = None
        global_logits = None
            
            
        for round_num in range(1, comm_rounds+1):
            print("\n" + "="*50)
            print(f"Round {round_num}/{comm_rounds}")
            print("="*50)
            
            # 选择部分客户端进行训练
            selected_workers = {
                client_name: self._workers[client_name]
                for client_name in random.sample(
                    list(self._workers.keys()), int(len(self._workers) * ratio_client)
                )
            }
            
            

            # 训练
            if self.strategy == "scaffold":
                global_weight, train_loss_list, global_c = self.fit_scaffold(selected_workers, 
                                                                             round_num, global_weight,
                                                                               global_c)
            elif self.strategy == "fedavg":
                global_weight, train_loss_list = self.fit_fedavg(selected_workers, round_num, global_weight)
            elif self.strategy == "fedprox":
                global_weight, train_loss_list = self.fit_fedprox(selected_workers, round_num, global_weight)
            elif self.strategy == "moon":
                global_weight, train_loss_list = self.fit_moon(selected_workers, round_num, global_weight)
            elif self.strategy == "feddistill":
                global_weight, train_loss_list, global_logits = self.fit_fed_distill(selected_workers, 
                                                                                     round_num, global_weight,
                                                                                     global_logits)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            

            # 评估
            accuracy_list = []
            test_loss_list = []
            print("\nEvaluating clients...")
            for client_name, worker in tqdm(
                selected_workers.items(), desc="Progress", unit="client"
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
            print("\nRound Summary:")
            print(f"├─ Train Loss: {avg_train_loss:.4f}")
            print(f"├─ Test Accuracy: {avg_accuracy:.2%}")
            print(f"└─ Test Loss: {avg_test_loss:.4f}")


        return self.history
    def fit_scaffold(self, selected_workers,round_num, global_weight, global_c):
        client_weight_list = []
        client_c_list = []  # 客户端的控制变量
        sample_num_list = []
        train_loss_list = []
        
        # 参与本轮的客户端数量
        num_clients = len(selected_workers)
        # 总客户端数量
        total_clients = len(self._workers)
        
        for client_name, worker in selected_workers.items():
            # 传递全局模型和全局控制变量到客户端
            client_weight, sample_num, train_loss, c = worker.local_train(
                sync_round=round_num, weights=global_weight, cg=global_c
            )
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            client_c_list.append(c)  
            
            train_loss_list.append(train_loss)
            self.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 聚合模型权重
        global_weight = average_weight(client_weight_list, sample_num_list)
        global_c = average_scaffold_parameter_c(client_c_list, sample_num_list)
            
            
        return global_weight, train_loss_list, global_c

    def fit_fed_distill(self, selected_workers, round_num, global_weight, global_logits):
        """
        Federated Distillation server aggregation
        """
        client_weight_list = []
        client_logits_list = []  # 存储客户端的logits
        sample_num_list = []
        train_loss_list = []
        
        # 1. 收集所有客户端的权重、logits和训练损失
        for client_name, worker in selected_workers.items():
            client_weight, sample_num, train_loss, client_logits = worker.local_train(
                sync_round=round_num,
                weights=None, #原论文保持一致
                global_logits=global_logits
            )
            
            client_weight_list.append(client_weight)
            client_logits_list.append((client_logits, sample_num))  # 保存logits和样本数
            sample_num_list.append(sample_num)
            train_loss_list.append(train_loss)
            self.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 2. 聚合模型权重（FedAvg方式）
        global_weight = average_weight(client_weight_list, sample_num_list)
        
        # 3. 聚合logits（加权平均）
        global_logits = average_logits(client_logits_list, sample_num_list)
        
        return global_weight, train_loss_list, global_logits

    def fit_fedavg(self, selected_workers,round_num, global_weight):
        return self.fit_common(selected_workers,round_num, global_weight)
    def fit_fedprox(self, selected_workers,round_num, global_weight):
        return self.fit_common(selected_workers,round_num, global_weight)
    def fit_moon(self, selected_workers,round_num, global_weight):
        return self.fit_common(selected_workers,round_num, global_weight)
    def fit_common(self, selected_workers,round_num, global_weight):
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        for client_name, worker in selected_workers.items():
            client_weight, sample_num, train_loss = worker.local_train(sync_round=round_num, weights=global_weight)
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            train_loss_list.append(train_loss)
            self.history["workers"][client_name]["train_loss"].append(train_loss)
        global_weight = average_weight(client_weight_list, sample_num_list)
        return global_weight, train_loss_list
