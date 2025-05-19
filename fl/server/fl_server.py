# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:31
# @Describe:
import random

from tqdm import tqdm

from fl.aggregation.aggregator import average_weight
from fl.client.fl_base import ModelConfig
from fl.client.strategy import create_client
from fl.server.strategy.strategy_factory import StrategyFactory

class FLServer:
    def __init__(
        self,
        client_list,
        strategy,
        model_config: ModelConfig,
        client_dataset_dict,
        seed,
        **kwargs,
    ):
        self.strategy_name = strategy  # 联邦学习策略名称（例如 FedAvg, FedProx）
        self.model_config = model_config  # 服务器的全局模型
        self.seed = seed  # 随机种子
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
        
        # 创建聚合策略实例
        self.aggregation_strategy = StrategyFactory.get_strategy(strategy, kwargs)

        #设置全局测试数据集
        self.global_test_dataset = client_dataset_dict["global"]["test_dataset"]
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
        # 初始化全局权重
        global_weight = self.initialize_client_weights()
        
        # 开始联邦训练
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
            
            # 使用策略模式进行聚合
            global_weight, train_loss_list = self.aggregation_strategy.aggregate(
                self, selected_workers, round_num, global_weight
            )
            
            # 评估选中的客户端
            accuracy_list = []
            test_loss_list = []
            print("\nEvaluating clients...")
            for client_name, worker in tqdm(
                selected_workers.items(), desc="Progress", unit="client"
            ):
                # 客户端评估并记录数据
                test_acc, test_loss = worker.evaluate(self.global_test_dataset)

                accuracy_list.append(test_acc)
                test_loss_list.append(test_loss)

                self.history["workers"][client_name]["accuracy"].append(test_acc)
                self.history["workers"][client_name]["test_loss"].append(test_loss)

            # 在全局测试数据集上评估模型

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
