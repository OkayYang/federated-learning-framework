# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:31
# @Describe:
import random
import torch
from tqdm import tqdm
import numpy as np
import os

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
        
        # 设置全局随机种子以确保结果可复现
        self._set_seed(seed)
        
        # 创建客户端
        self._workers = {}
        for client in client_list:
            # 为每个客户端设置不同的种子，确保环境隔离
            client_seed = seed + int(client) if client.isdigit() else seed + hash(client) % 10000
            # 创建客户端，传入特定种子
            self._workers[client] = create_client(
                strategy, client, model_config, client_dataset_dict, 
                seed=client_seed, **kwargs
            )
        
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

        # 设置全局测试数据集
        self.global_test_dataset = client_dataset_dict["global"]["test_dataset"]
        
        # 创建用于评估的全局模型
        self.global_eval_model = self.model_config.get_model()
        self.global_eval_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.global_loss_fn = self.model_config.get_loss_fn()
    
    def _set_seed(self, seed):
        """设置全局随机种子，确保可复现性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 设置CUDA的确定性（可能会降低性能，但提高可复现性）
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
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
            client_keys = list(self._workers.keys())
            # 设置随机种子，确保可以复现客户端选择
            random.seed(self.seed + round_num)
            selected_client_names = random.sample(
                client_keys, int(len(self._workers) * ratio_client)
            )
            selected_workers = {
                client_name: self._workers[client_name]
                for client_name in selected_client_names
            }
            
            # 使用策略模式进行聚合
            global_weight, train_loss_list = self.aggregation_strategy.aggregate(
                self, selected_workers, round_num, global_weight
            )
            
            # 使用之前创建的全局评估模型
            # 将全局权重转换为模型格式
            keys = self.global_eval_model.state_dict().keys()
            global_weights_dict = {}
            for k, v in zip(keys, global_weight):
                global_weights_dict[k] = torch.Tensor(np.copy(v))
            
            # 加载聚合后的全局权重到全局评估模型
            self.global_eval_model.load_state_dict(global_weights_dict)
            self.global_eval_model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 创建数据加载器
            test_loader = torch.utils.data.DataLoader(
                self.global_test_dataset, 
                batch_size=self.model_config.get_batch_size(),
                shuffle=False,  # 不打乱顺序以确保一致性
                drop_last=False
            )
            
            # 评估全局模型
            correct = 0
            test_loss = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = self.global_eval_model(data)
                    # 累加批次损失
                    batch_loss = self.global_loss_fn(output, target).item()
                    test_loss += batch_loss * len(data)  # 按样本数加权
                    
                    # 计算预测准确度
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            # 计算全局模型的准确率和损失
            global_accuracy = correct / total if total > 0 else 0
            global_test_loss = test_loss / total if total > 0 else float('inf')
            
            # 评估每个选中的客户端
            print("\nEvaluating selected clients...")
            client_accuracies = []
            client_test_losses = []
            
            for client_name, worker in tqdm(
                selected_workers.items(), desc="Progress", unit="client"
            ):
                # 先更新客户端的模型权重为全局权重
                #worker.update_weights(global_weight)
                # 在客户端自己的测试数据上评估
                test_acc, test_loss = worker.evaluate()
                
                client_accuracies.append(test_acc)
                client_test_losses.append(test_loss)
                
                # 记录客户端历史数据
                self.history["workers"][client_name]["accuracy"].append(test_acc)
                self.history["workers"][client_name]["test_loss"].append(test_loss)
            
            # 计算客户端评估的平均值
            avg_client_accuracy = sum(client_accuracies) / len(client_accuracies) if client_accuracies else 0
            avg_client_test_loss = sum(client_test_losses) / len(client_test_losses) if client_test_losses else 0
            # 保存全局指标到历史记录中
            avg_train_loss = sum(train_loss_list) / len(train_loss_list) if train_loss_list else 0
            self.history["global"]["train_loss"].append(avg_train_loss)
            self.history["global"]["test_accuracy"].append(global_accuracy)
            self.history["global"]["test_loss"].append(global_test_loss)

            
            # 输出当前轮次的结果
            print("\nRound Summary:")
            print(f"├─ Train Loss: {avg_train_loss:.4f}")
            print(f"├─ Global Test Accuracy: {global_accuracy:.2%}")
            print(f"├─ Global Test Loss: {global_test_loss:.4f}")
            print(f"├─ Avg Client Test Accuracy: {avg_client_accuracy:.2%}")
            print(f"└─ Avg Client Test Loss: {avg_client_test_loss:.4f}")

        return self.history
