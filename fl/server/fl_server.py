# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:31
# @Describe:
import random
import torch
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader
import ray

from fl.aggregation.aggregator import average_weight
from fl.client.fl_base import ModelConfig
from fl.client.strategy import create_client
from fl.server.strategy.strategy_factory import StrategyFactory

from fl.server.progress import ProgressActor, ProgressMonitor

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
        
        # 创建全局模型实例，用于全局数据集评估
        self.global_model = model_config.get_model()
        self.global_loss_fn = model_config.get_loss_fn()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        
        # 初始化进度监控
        self.progress_actor = ProgressActor.remote()
        
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
            # 设置进度Actor
            self._workers[client].set_progress_actor.remote(self.progress_actor)
        
        # 初始化历史记录结构，包含每个worker的详细记录
        self.history = {
            "global": {"train_loss": [], "test_accuracy": [], "test_loss": []},
            "local": {"train_loss": [], "test_accuracy": [], "test_loss": []},
            "workers": {
                client: {"train_loss": [], "local_test_accuracy": [], "local_test_loss": [], "global_test_accuracy": [], "global_test_loss": []}
                for client in client_list
            },
        }
        self.kwargs = kwargs
        
        # 创建聚合策略实例
        self.aggregation_strategy = StrategyFactory.get_strategy(strategy, kwargs)
        
    
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
    
    def update_global_model_weights(self, weights):
        """更新全局模型的权重"""
        if len(weights) != len(self.global_model.state_dict()):
            raise ValueError("传入的权重数组数量与全局模型参数数量不匹配。")
        
        keys = self.global_model.state_dict().keys()
        weights_dict = {}
        
        # 更健壮的权重转换
        for k, v in zip(keys, weights):
            # 确保权重是有效的浮点型数组
            try:
                # 首先确保是numpy数组
                if not isinstance(v, np.ndarray):
                    v = np.array(v)
                    
                # 检查数据类型，确保是浮点类型
                if not np.issubdtype(v.dtype, np.floating):
                    print(f"警告: 权重 {k} 的数据类型为 {v.dtype}，转换为 float32")
                    v = v.astype(np.float32)
                
                # 检查是否有无效值
                if not np.all(np.isfinite(v)):
                    print(f"警告: 权重 {k} 包含无效值 (NaN 或 Inf)")
                    # 使用原始模型的权重
                    v = self.global_model.state_dict()[k].cpu().numpy()
                
                # 检查形状是否有效
                if 0 in v.shape or np.any(np.array(v.shape) < 0):
                    print(f"警告: 权重 {k} 的形状 {v.shape} 无效")
                    # 使用原始模型的权重
                    v = self.global_model.state_dict()[k].cpu().numpy()
                
                # 转换为张量并移至设备
                weights_dict[k] = torch.tensor(v, dtype=torch.float32).to(self.device)
                
            except Exception as e:
                print(f"更新权重 {k} 时出错: {str(e)}")
                # 使用原始模型的权重
                weights_dict[k] = self.global_model.state_dict()[k]
        
        # 加载更新后的权重字典
        self.global_model.load_state_dict(weights_dict)
    
            
    def initialize_client_weights(self):
        """设置服务器端的全局模型"""
        clients_weights = []
        # Ray: Launch all get_weights tasks
        futures = [worker.get_weights.remote(return_numpy=True) for worker in self._workers.values()]
        # Ray: Wait for all results
        clients_weights = ray.get(futures)

        return average_weight(clients_weights)
    
    def fit(self, comm_rounds, ratio_client=1.0):
        """
        进行训练的主要流程，包括与各客户端的交互
        """
        # 初始化全局权重
        global_weight = self.initialize_client_weights()
        
        # 开始联邦训练
        with ProgressMonitor(self.progress_actor) as monitor:
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
                
                # 更新全局模型权重
                self.update_global_model_weights(global_weight)

                # 全局训练损失
                avg_train_loss = sum(train_loss_list) / len(train_loss_list) if train_loss_list else 0
                
                # 使用参与方模型在全局数据集上评估
                print("\nEvaluating selected clients on global dataset...")
                
                # Ray: Launch all global_evaluate tasks
                futures = [worker.global_evaluate.remote() for worker in selected_workers.values()]
                results = ray.get(futures)
                
                client_global_accuracies = []
                client_global_test_losses = []
                
                for i, client_name in enumerate(selected_workers.keys()):
                    global_test_accuracy, global_test_loss = results[i]
                    client_global_accuracies.append(global_test_accuracy)
                    client_global_test_losses.append(global_test_loss)
                    self.history["workers"][client_name]["global_test_accuracy"].append(global_test_accuracy)
                    self.history["workers"][client_name]["global_test_loss"].append(global_test_loss)

                avg_client_global_accuracy = sum(client_global_accuracies) / len(client_global_accuracies) if client_global_accuracies else 0
                avg_client_global_test_loss = sum(client_global_test_losses) / len(client_global_test_losses) if client_global_test_losses else 0
                
                # 保存全局指标到历史记录中
                self.history["global"]["train_loss"].append(avg_train_loss)
                self.history["global"]["test_accuracy"].append(avg_client_global_accuracy)
                self.history["global"]["test_loss"].append(avg_client_global_test_loss)

                # 评估每个选中的客户端在本地数据集上的性能
                print("\nEvaluating selected clients on local dataset...")
                
                # Ray: Launch all local_evaluate tasks
                futures = [worker.local_evaluate.remote() for worker in selected_workers.values()]
                results = ray.get(futures)
                
                client_accuracies = []
                client_test_losses = []
                
                for i, client_name in enumerate(selected_workers.keys()):
                    test_acc, test_loss = results[i]
                    
                    client_accuracies.append(test_acc)
                    client_test_losses.append(test_loss)
                    
                    # 记录客户端历史数据
                    self.history["workers"][client_name]["local_test_accuracy"].append(test_acc)
                    self.history["workers"][client_name]["local_test_loss"].append(test_loss)
                
                # 计算客户端评估的平均值
                avg_client_accuracy = sum(client_accuracies) / len(client_accuracies) if client_accuracies else 0
                avg_client_test_loss = sum(client_test_losses) / len(client_test_losses) if client_test_losses else 0      

                self.history["local"]["train_loss"].append(avg_train_loss)
                self.history["local"]["test_accuracy"].append(avg_client_accuracy)
                self.history["local"]["test_loss"].append(avg_client_test_loss)

                
                # 输出当前轮次的结果
                print("\nRound Summary:")
                print(f"├─ Train Loss: {avg_train_loss:.4f}")
                print(f"├─ Global Test Accuracy: {avg_client_global_accuracy:.2%}")
                print(f"├─ Global Test Loss: {avg_client_global_test_loss:.4f}")
                print(f"├─ Avg Client Test Accuracy: {avg_client_accuracy:.2%}")
                print(f"└─ Avg Client Test Loss: {avg_client_test_loss:.4f}")

        return self.history
