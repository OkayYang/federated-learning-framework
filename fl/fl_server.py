# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:31
# @Describe:
import copy
import random

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F

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
        seed,
        **kwargs,
    ):
        self.strategy = strategy  # 联邦学习策略（例如 FedAvg, FedProx）
        self.model_config = model_config  # 服务器的全局模型
        self.seed = seed # 随机种子
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
        self.global_generator = self.kwargs.get('generator_model', None)

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
        global_class_reps = None
        if self.global_generator is not None:
            global_generator_weights = self.global_generator.get_weights(return_numpy=True)
        else:
            global_generator_weights = None
            
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
            elif self.strategy == "fedgen":
                global_weight, train_loss_list, global_logits, global_generator_weights = self.fit_fedgen(
                    selected_workers, round_num, global_weight, global_logits, global_generator_weights
                )
            elif self.strategy == "fedspd":
                global_weight, train_loss_list, global_class_reps, global_logits = self.fit_fedspd(
                    selected_workers, round_num, global_weight, global_class_reps, global_logits
                )
            elif self.strategy == "fedalone":
                train_loss_list = self.fit_fedalone(selected_workers, round_num)
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
    def fit_fedalone(self, selected_workers,round_num):
        train_loss_list = []
        for client_name, worker in selected_workers.items():
            train_loss = worker.local_train(sync_round=round_num)
            train_loss_list.append(train_loss)
            self.history["workers"][client_name]["train_loss"].append(train_loss)
        return train_loss_list
    def fit_fedspd(self, selected_workers, round_num, global_weight, global_class_reps=None, global_logits=None):
        """
        FedSPD服务器聚合方法 - 类别感知蒸馏机制
        简化版实现，专注于有效的知识迁移
        """
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        client_class_reps = []  # 收集客户端的类别表征
        client_class_logits = []  # 收集客户端的类别logits
        
        # 1. 客户端训练
        for client_name, worker in selected_workers.items():
            client_weight, sample_num, train_loss, class_reps, class_logits = worker.local_train(
                sync_round=round_num,
                weights=None, # 传递全局模型权重
                global_reps=global_class_reps,
                global_logits=global_logits
            )
            
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            train_loss_list.append(train_loss)
            client_class_reps.append(class_reps)
            client_class_logits.append(class_logits)
            
            self.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 2. 聚合模型权重 (FedAvg方式)
        global_weight = average_weight(client_weight_list, sample_num_list)
        
        # 3. 聚合类别表征和logits (简单平均)
        # 初始化全局类别表征和logits字典
        all_class_reps = {}
        all_class_logits = {}
        
        # 收集所有客户端的表征和logits
        for client_rep, client_logit in zip(client_class_reps, client_class_logits):
            for class_id, rep in client_rep.items():
                if class_id not in all_class_reps:
                    all_class_reps[class_id] = []
                all_class_reps[class_id].append(rep)
                
            for class_id, logit in client_logit.items():
                if class_id not in all_class_logits:
                    all_class_logits[class_id] = []
                all_class_logits[class_id].append(logit)
        
        # 计算每个类别的全局平均表征和logits
        global_class_reps = {}
        global_logits = {}
        
        # 对每个类别进行简单平均
        for class_id, reps in all_class_reps.items():
            if reps:
                global_class_reps[class_id] = np.mean(np.array(reps), axis=0)
                
        for class_id, logits in all_class_logits.items():
            if logits:
                global_logits[class_id] = np.mean(np.array(logits), axis=0)
        
        # 打印信息
        print(f"FedSPD 第{round_num}轮: 聚合了{len(global_class_reps)}个类别的知识")
        
        return global_weight, train_loss_list, global_class_reps, global_logits
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
    def fit_fedgen(self, selected_workers, round_num, global_weight, global_logits=None, global_generator=None):
        """
        FedGen服务器聚合
        """
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        label_count_list = []
        client_model_list = []
        
        # 1. 客户端训练
        for client_name, worker in selected_workers.items():
            client_weight, sample_num, train_loss, label_count = worker.local_train(
                sync_round=round_num,
                weights=None,
                generator_weights=global_generator
            )
            client_model_list.append(worker.get_model_copy())  # 这里仅仅是本地实验，实际中要参数传递
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            train_loss_list.append(train_loss)
            label_count_list.append(label_count)
            self.history["workers"][client_name]["train_loss"].append(train_loss)
        # 2. 聚合模型权重
        global_weight = average_weight(client_weight_list, sample_num_list)
        # 3. 生成器模型训练配置
        # 3.1 标签权重
        label_weights = []
        total_student_loss = 0
        # Compute the weight of each label across all workers
        for label in range(self.global_generator.num_classes):
            # Get the count of this label from each worker
            weights = [
                label_count.get(label, 0) for label_count in label_count_list
            ]
            # Sum the counts, add a small epsilon to avoid division by zero
            label_sum = np.sum(weights) + 1e-6  # Small tolerance to avoid zero division

            # Compute the weight for this label across all workers
            label_weights.append(np.array(weights) / label_sum)

        # 3.2 生成器模型训练
        self.global_generator.train()
        total_loss = 0
        for i in range(self.global_generator.train_epochs):
            self.global_generator.optimizer.zero_grad()
            # 随机生成标签
            sampled_labels = np.random.choice(
                self.global_generator.num_classes, self.global_generator.train_batch_size
            )
            # 确保转换为PyTorch张量
            sampled_labels = torch.LongTensor(sampled_labels)
            # 生成随机噪声
            eps = torch.randn(self.global_generator.train_batch_size, self.global_generator.latent_dim)
            # 生成合成数据
            synthetic_features = self.global_generator(eps, sampled_labels)
            diversity_loss = self.global_generator.diversity_loss(eps, synthetic_features)

            # Initialize teacher loss
            teacher_loss = 0
            teacher_logit = 0
            for idx in range(len(client_model_list)):
                # Compute the weight of each label for each worker
                # 将sampled_labels转换为numpy数组
                sampled_labels_np = sampled_labels.cpu().numpy()
                
                # 创建权重数组
                batch_weights = np.zeros((len(sampled_labels_np), 1))
                
                # 为每个样本获取权重
                for i, label in enumerate(sampled_labels_np):
                    batch_weights[i, 0] = label_weights[label][idx]
                
                weight = torch.tensor(
                    batch_weights,
                    dtype=torch.float32,
                )
                expand_weight = np.tile(weight.cpu().numpy(), (1, self.global_generator.num_classes))
                with torch.no_grad():
                    teacher_logits = client_model_list[idx](synthetic_features,start_layer=True)

                # Calculate the weighted teacher loss for each worker
                teacher_loss_ = torch.mean(
                    self.global_generator.loss_fn(teacher_logits, sampled_labels)
                    * weight
                )
                teacher_loss += teacher_loss_
                teacher_logit += teacher_logits * torch.tensor(
                    expand_weight, dtype=torch.float32
                )
             

            # 全局模型当学生,参与方模型logit平均当老师，这里没有维护全局模型，就假装第一个参与方模型
            global_model = copy.deepcopy(client_model_list[0])
            
            # 更新全局模型参数
            keys = global_model.state_dict().keys()
            weights_dict = {}
            for k, v in zip(keys, global_weight):
                weights_dict[k] = torch.Tensor(np.copy(v))
            global_model.load_state_dict(weights_dict)
            global_model.eval()
            with torch.no_grad():
                student_output = global_model(synthetic_features,start_layer=True).clone().detach()
            student_loss = F.kl_div(
                F.log_softmax(student_output, dim=1), F.softmax(teacher_logit, dim=1), reduction='batchmean'
            )
            total_student_loss += student_loss
            # 总损失
            if self.global_generator.ensemble_beta > 0:
                loss = (
                    self.global_generator.ensemble_alpha * teacher_loss
                    - self.global_generator.ensemble_beta * student_loss
                    + self.global_generator.ensemble_eta * diversity_loss
                )
            else:
                loss = (
                    self.global_generator.ensemble_alpha * teacher_loss
                    + self.global_generator.ensemble_eta * diversity_loss
                )
            loss.backward()
            self.global_generator.optimizer.step()
            total_loss += loss.item()
        print(f"FedGen 第{round_num}轮: 生成器模型训练第{i+1}轮, 总损失: {total_loss:.4f}")
        return global_weight, train_loss_list, global_logits, self.global_generator.get_weights(return_numpy=True)
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
