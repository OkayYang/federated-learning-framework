# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/19 16:30
# @Describe: 聚合策略接口和实现类

from abc import ABC
import numpy as np
import torch
import copy

from fl.aggregation.aggregator import average_weight, average_logits, average_scaffold_parameter_c


class AggregationStrategy(ABC):
    """聚合策略基类"""
    
    def initialize(self, server_kwargs):
        """
        从服务器参数初始化算法特定数据
        
        Args:
            server_kwargs: 服务器初始化时传入的参数
        """
        pass
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """
        执行聚合逻辑的默认实现 - 简单的加权平均
        
        Args:
            server: 服务器实例
            selected_workers: 选中的工作节点
            round_num: 当前轮次
            global_weights: 全局模型权重
            
        Returns:
            tuple: (更新后的全局状态, 训练损失列表)
        """
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        
        for client_name, worker in selected_workers.items():
            # 调用客户端训练
            model_weights, num_sample, avg_loss = worker.local_train(
                sync_round=round_num, 
                weights=global_weights
            )
            
            # 处理客户端返回结果
            client_weight_list.append(model_weights)
            sample_num_list.append(num_sample)
            train_loss_list.append(avg_loss)
            server.history["workers"][client_name]["train_loss"].append(avg_loss)
        
        # 执行加权平均聚合
        global_weight = average_weight(client_weight_list, sample_num_list)
        
        return global_weight, train_loss_list
    

class FedAloneStrategy(AggregationStrategy):
    """FedAlone聚合策略 - 使用基础聚合实现"""
    pass

class FedAvgStrategy(AggregationStrategy):
    """FedAvg聚合策略 - 使用基础聚合实现"""
    pass


class FedProxStrategy(AggregationStrategy):
    """FedProx聚合策略 - 使用基础聚合实现"""
    pass


class MoonStrategy(AggregationStrategy):
    """Moon聚合策略 - 使用基础聚合实现"""
    pass


class ScaffoldStrategy(AggregationStrategy):
    """Scaffold聚合策略"""
    def __init__(self):
        self.global_c = None
    
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """重写聚合方法，处理控制变量"""
        client_weight_list = []
        client_c_list = []
        sample_num_list = []
        train_loss_list = []
        
        for client_name, worker in selected_workers.items():
            # 使用自定义的客户端更新方法
            client_weight, sample_num, train_loss, c =  worker.local_train(
                sync_round=round_num,
                weights=global_weights,
                cg=self.global_c
            )
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            client_c_list.append(c)
            train_loss_list.append(train_loss)
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 聚合模型权重
        global_weight = average_weight(client_weight_list, sample_num_list)
        self.global_c = average_scaffold_parameter_c(client_c_list, sample_num_list)
        return global_weight, train_loss_list


class FedDistillStrategy(AggregationStrategy):
    """FedDistill聚合策略"""
    def __init__(self):
        self.global_logits = None
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """重写聚合方法，处理logits"""
        client_weight_list = []
        client_logits_list = []
        sample_num_list = []
        train_loss_list = []
        
        for client_name, worker in selected_workers.items():
            # 使用自定义的客户端更新方法
            client_weight, sample_num, train_loss, client_logits = worker.local_train(
                sync_round=round_num,
                weights=global_weights,
                global_logits=self.global_logits
            )
            
            # 处理结果
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            client_logits_list.append((client_logits, sample_num))
            train_loss_list.append(train_loss)
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 聚合模型权重
        global_weight = average_weight(client_weight_list, sample_num_list)

        # 聚合logits
        global_logits = average_logits(client_logits_list, sample_num_list)
        self.global_logits = global_logits
        
        return global_weight, train_loss_list


class FedGenStrategy(AggregationStrategy):
    """FedGen聚合策略"""
    
    def initialize(self, server_kwargs):
        """初始化生成器模型"""
        generator_model = server_kwargs.get('generator_model')
        if generator_model is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.global_generator = generator_model.to(self.device)
        else:
            raise ValueError("生成器模型未初始化")
    
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """重写聚合方法，处理生成器训练"""
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        label_count_list = []
        client_model_list = []
        
        for client_name, worker in selected_workers.items():
            # 使用自定义的客户端更新方法
            generator_weights = self.global_generator.get_weights(return_numpy=True)
            client_weight, sample_num, train_loss, label_count  =  worker.local_train(
                sync_round=round_num,
                weights=global_weights,
                generator_weights=generator_weights
            )
            
            # 处理结果
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            train_loss_list.append(train_loss)
            label_count_list.append(label_count)
            client_model_list.append(worker.get_model_copy())
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 聚合模型权重
        global_weight = average_weight(client_weight_list, sample_num_list)
        
        # 训练生成器
        self._train_generator(
            client_model_list, 
            label_count_list, 
            global_weight, 
            round_num
        )
        
        return global_weight, train_loss_list
    
    def _train_generator(self, client_model_list, label_count_list, global_weight, round_num):
        """训练生成器模型"""
        
        # 计算标签权重
        label_weights = []
        for label in range(self.global_generator.num_classes):
            weights = [label_count.get(label, 0) for label_count in label_count_list]
            label_sum = np.sum(weights) + 1e-6
            label_weights.append(np.array(weights) / label_sum)
        
        # 生成器训练
        self.global_generator.train()
        total_loss = 0
        
        for i in range(self.global_generator.train_epochs):
            self.global_generator.optimizer.zero_grad()
            
            # 随机生成标签
            sampled_labels = np.random.choice(
                self.global_generator.num_classes, self.global_generator.train_batch_size
            )
            sampled_labels = torch.LongTensor(sampled_labels).to(self.device)
            
            # 生成随机噪声
            
            # 生成合成数据
            eps,synthetic_features = self.global_generator(sampled_labels)
            diversity_loss = self.global_generator.diversity_loss(eps, synthetic_features)
            
            # 教师损失
            teacher_loss = 0
            teacher_logit = 0
            
            for idx in range(len(client_model_list)):
                # 计算每个标签的权重
                sampled_labels_np = sampled_labels.cpu().numpy()
                batch_weights = np.zeros((len(sampled_labels_np), 1))
                
                for i, label in enumerate(sampled_labels_np):
                    batch_weights[i, 0] = label_weights[label][idx]
                
                weight = torch.tensor(batch_weights, dtype=torch.float32).to(self.device)
                expand_weight = np.tile(weight.cpu().numpy(), (1, self.global_generator.num_classes))
                
                with torch.no_grad():
                    teacher_logits = client_model_list[idx](synthetic_features, start_layer="classify")
                
                teacher_loss_ = torch.mean(
                    self.global_generator.loss_fn(teacher_logits, sampled_labels) * weight
                )
                teacher_loss += teacher_loss_
                teacher_logit += teacher_logits * torch.tensor(
                    expand_weight, dtype=torch.float32
                ).to(self.device)
            
            # 学生损失
            global_model = copy.deepcopy(client_model_list[0])
            keys = global_model.state_dict().keys()
            weights_dict = {}
            
            for k, v in zip(keys, global_weight):
                weights_dict[k] = torch.Tensor(np.copy(v)).to(self.device)
                
            global_model.load_state_dict(weights_dict)
            global_model.eval()
            
            with torch.no_grad():
                student_output = global_model(synthetic_features, start_layer="classify").clone().detach()
            
            student_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_output, dim=1),
                torch.nn.functional.softmax(teacher_logit, dim=1),
                reduction='batchmean'
            )
            
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
        
        print(f"FedGen 第{round_num}轮: 生成器模型训练, 总损失: {total_loss:.4f}")


class FedSPDStrategy(AggregationStrategy):
    """FedSPD聚合策略 - 使用K-means聚类和多种高级聚合技术"""
    def __init__(self):
        self.global_reps = None
        self.global_logits = None
        self.eps = 1e-10  # 添加一个小的常数避免除零
        self.n_clusters = 3  # K-means聚类数量，可以根据实际情况调整
        self.use_kmeans = True  # 是否使用K-means聚类
        self.use_spectral = True  # 是否使用谱聚类
        self.use_weighted_avg = True  # 是否使用加权平均
        self.cluster_weight_power = 2.0  # 聚类权重指数，用于调整聚类中心的影响力
    
    def _compute_confidence_scores(self, logits):
        """基于预测熵计算置信度分数"""
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=-1)
        confidence = 1 - (entropy / (np.log(probs.shape[-1]) + self.eps))  # 归一化熵
        return confidence.numpy()
    
    def _compute_representation_quality(self, reps, labels):
        """基于类内紧致性和类间分离性计算表征质量"""
        reps = np.array(reps, dtype=np.float32)
        labels = np.array(labels)
        unique_classes = np.unique(labels)
        
        # 计算类内距离(紧致性)
        intra_distances = []
        for cls in unique_classes:
            cls_reps = reps[labels == cls]
            if len(cls_reps) > 1:
                centroid = np.mean(cls_reps, axis=0)
                dist = np.mean(np.linalg.norm(cls_reps - centroid, axis=1))
                intra_distances.append(dist)
        
        if not intra_distances:  # 如果没有计算出任何距离
            return 1.0  # 返回1.0而不是0.0，这样可以保持权重的有效性
            
        compactness = 1 / (np.mean(intra_distances) + self.eps)
        
        # 计算类间距离(分离性)
        centroids = []
        for cls in unique_classes:
            cls_reps = reps[labels == cls]
            centroids.append(np.mean(cls_reps, axis=0))
        
        centroids = np.array(centroids, dtype=np.float32)
        if len(centroids) > 1:
            separation = np.mean([
                np.min([np.linalg.norm(c1 - c2) for j, c2 in enumerate(centroids) if i != j])
                for i, c1 in enumerate(centroids)
            ])
        else:
            separation = 1.0  # 当只有一个类别时，设置为1.0而不是0.0
            
        quality_score = float(compactness * separation)  # 确保返回标量
        return max(quality_score, self.eps)  # 确保质量分数不为零
    
    def _kmeans_cluster_representations(self, reps_list, weights=None):
        """使用K-means聚类对表征进行聚类，返回聚类中心和权重"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            print("警告: scikit-learn未安装，无法使用K-means聚类。将使用加权平均替代。")
            return None, None
            
        reps_array = np.array(reps_list, dtype=np.float32)
        if len(reps_array) < self.n_clusters:
            # 如果样本数少于聚类数，直接返回原始表征和权重
            return reps_array, weights if weights is not None else np.ones(len(reps_array)) / len(reps_array)
            
        # 自动确定最佳聚类数
        best_n_clusters = 2
        best_score = -1
        
        # 尝试不同的聚类数，选择轮廓系数最高的
        max_clusters = min(self.n_clusters, len(reps_array) - 1)
        for n_clusters in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(reps_array)
                
                # 至少需要2个聚类且每个聚类至少有一个样本才能计算轮廓系数
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(reps_array, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            except:
                continue
                
        # 使用最佳聚类数进行聚类
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(reps_array)
        cluster_centers = kmeans.cluster_centers_
        
        # 计算每个聚类的权重
        cluster_weights = np.zeros(best_n_clusters, dtype=np.float32)
        for i in range(best_n_clusters):
            # 该聚类中的样本数
            cluster_size = np.sum(cluster_labels == i)
            # 该聚类中样本的原始权重之和
            if weights is not None:
                cluster_weights[i] = np.sum(weights[cluster_labels == i])
            else:
                cluster_weights[i] = cluster_size / len(reps_array)
                
        # 根据聚类大小和紧密度调整权重
        distances_to_center = np.zeros(best_n_clusters, dtype=np.float32)
        for i in range(best_n_clusters):
            cluster_points = reps_array[cluster_labels == i]
            if len(cluster_points) > 0:
                # 计算聚类内样本到中心的平均距离
                distances = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
                distances_to_center[i] = np.mean(distances) + self.eps
        
        # 距离越小，权重越大
        compactness_weights = 1.0 / (distances_to_center + self.eps)
        # 归一化紧密度权重
        compactness_weights = compactness_weights / (np.sum(compactness_weights) + self.eps)
        
        # 结合聚类大小和紧密度的权重
        final_weights = cluster_weights * (compactness_weights ** self.cluster_weight_power)
        final_weights = final_weights / (np.sum(final_weights) + self.eps)
        
        return cluster_centers, final_weights
    
    def _spectral_cluster_representations(self, reps_list, weights=None):
        """使用谱聚类对表征进行聚类，更适合非凸形状的聚类"""
        try:
            from sklearn.cluster import SpectralClustering
            from sklearn.metrics import silhouette_score
        except ImportError:
            print("警告: scikit-learn未安装，无法使用谱聚类。将使用加权平均替代。")
            return None, None
            
        reps_array = np.array(reps_list, dtype=np.float32)
        if len(reps_array) < self.n_clusters:
            # 如果样本数少于聚类数，直接返回原始表征和权重
            return reps_array, weights if weights is not None else np.ones(len(reps_array)) / len(reps_array)
            
        # 自动确定最佳聚类数
        best_n_clusters = 2
        best_score = -1
        
        # 尝试不同的聚类数，选择轮廓系数最高的
        max_clusters = min(self.n_clusters, len(reps_array) - 1)
        for n_clusters in range(2, max_clusters + 1):
            try:
                spectral = SpectralClustering(n_clusters=n_clusters, 
                                             affinity='nearest_neighbors',
                                             random_state=42)
                cluster_labels = spectral.fit_predict(reps_array)
                
                # 至少需要2个聚类且每个聚类至少有一个样本才能计算轮廓系数
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(reps_array, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            except:
                continue
                
        # 使用最佳聚类数进行聚类
        spectral = SpectralClustering(n_clusters=best_n_clusters, 
                                     affinity='nearest_neighbors',
                                     random_state=42)
        cluster_labels = spectral.fit_predict(reps_array)
        
        # 计算聚类中心
        cluster_centers = np.zeros((best_n_clusters, reps_array.shape[1]), dtype=np.float32)
        for i in range(best_n_clusters):
            cluster_points = reps_array[cluster_labels == i]
            if len(cluster_points) > 0:
                cluster_centers[i] = np.mean(cluster_points, axis=0)
        
        # 计算每个聚类的权重
        cluster_weights = np.zeros(best_n_clusters, dtype=np.float32)
        for i in range(best_n_clusters):
            # 该聚类中的样本数
            cluster_size = np.sum(cluster_labels == i)
            # 该聚类中样本的原始权重之和
            if weights is not None:
                cluster_weights[i] = np.sum(weights[cluster_labels == i])
            else:
                cluster_weights[i] = cluster_size / len(reps_array)
                
        # 根据聚类大小和紧密度调整权重
        distances_to_center = np.zeros(best_n_clusters, dtype=np.float32)
        for i in range(best_n_clusters):
            cluster_points = reps_array[cluster_labels == i]
            if len(cluster_points) > 0:
                # 计算聚类内样本到中心的平均距离
                distances = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
                distances_to_center[i] = np.mean(distances) + self.eps
        
        # 距离越小，权重越大
        compactness_weights = 1.0 / (distances_to_center + self.eps)
        # 归一化紧密度权重
        compactness_weights = compactness_weights / (np.sum(compactness_weights) + self.eps)
        
        # 结合聚类大小和紧密度的权重
        final_weights = cluster_weights * (compactness_weights ** self.cluster_weight_power)
        final_weights = final_weights / (np.sum(final_weights) + self.eps)
        
        return cluster_centers, final_weights
    
    def _aggregate_with_clustering(self, items_list, weights_list=None, use_spectral=False):
        """使用聚类方法聚合表征或logits"""
        if not items_list:
            return None
            
        # 提取所有项和对应权重
        items = []
        weights = []
        
        for i, (item, weight) in enumerate(items_list):
            items.append(item)
            weights.append(weight)
            
        items = np.array(items, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32) if weights else None
        
        # 根据配置选择聚类方法
        if use_spectral and self.use_spectral:
            centers, cluster_weights = self._spectral_cluster_representations(items, weights)
        elif self.use_kmeans:
            centers, cluster_weights = self._kmeans_cluster_representations(items, weights)
        else:
            centers, cluster_weights = None, None
            
        # 如果聚类失败或不使用聚类，则使用加权平均
        if centers is None or cluster_weights is None:
            if weights is not None:
                weights = weights / (np.sum(weights) + self.eps)
                return np.average(items, axis=0, weights=weights)
            else:
                return np.mean(items, axis=0)
                
        # 使用聚类中心的加权平均
        return np.average(centers, axis=0, weights=cluster_weights)
    
    def aggregate(self, server, selected_workers, round_num, global_weights):
        """重写聚合方法，处理类别表征和logits"""
        if not selected_workers:  # 如果没有选中的工作节点
            return global_weights, []  # 返回原始权重和空的损失列表
            
        client_weight_list = []
        sample_num_list = []
        train_loss_list = []
        client_class_reps = []
        client_class_logits = []
        client_qualities = []
        
        for client_name, worker in selected_workers.items():
            client_weight, sample_num, train_loss, class_reps, class_logits = worker.local_train(
                sync_round=round_num,
                weights=global_weights,
                global_reps=self.global_reps,
                global_logits=self.global_logits
            )
            
            # 计算该客户端表征的质量
            all_reps = []
            all_labels = []
            for cls, reps in class_reps.items():
                all_reps.extend([reps])
                all_labels.extend([cls])
            
            if all_reps:
                quality_score = self._compute_representation_quality(all_reps, all_labels)
            else:
                quality_score = 1.0  # 当没有表征时，设置为1.0而不是0.0
                
            client_qualities.append(quality_score)
            client_weight_list.append(client_weight)
            sample_num_list.append(sample_num)
            train_loss_list.append(train_loss)
            client_class_reps.append(class_reps)
            client_class_logits.append(class_logits)
            server.history["workers"][client_name]["train_loss"].append(train_loss)
        
        # 归一化质量分数
        if client_qualities:
            quality_weights = np.array(client_qualities, dtype=np.float32)
            sum_weights = np.sum(quality_weights) + self.eps
            quality_weights = quality_weights / sum_weights
        else:
            # 如果没有质量分数，使用均匀权重
            num_clients = len(selected_workers)
            quality_weights = np.ones(num_clients, dtype=np.float32) / num_clients
        
        # 确保sample_weights不全为零
        sample_weights = np.array([max(float(w), self.eps) for w in sample_num_list], dtype=np.float32)
        total_samples = np.sum(sample_weights)
        if total_samples < self.eps:  # 如果总样本数接近零
            sample_weights = np.ones_like(sample_weights) / len(sample_weights)
        else:
            sample_weights = sample_weights / total_samples
            
        # 使用质量感知权重聚合模型权重
        weighted_samples = sample_weights * quality_weights
        # 再次归一化确保权重和为1
        weighted_samples = weighted_samples / (np.sum(weighted_samples) + self.eps)
        global_weight = average_weight(client_weight_list, weighted_samples.tolist())
        
        # 增强型表征和logits聚合
        all_class_reps = {}
        all_class_logits = {}
        
        for client_idx, (client_rep, client_logit) in enumerate(zip(client_class_reps, client_class_logits)):
            client_weight = float(quality_weights[client_idx])  # 确保是标量
            
            for class_id, rep in client_rep.items():
                if class_id not in all_class_reps:
                    all_class_reps[class_id] = []
                all_class_reps[class_id].append((np.array(rep, dtype=np.float32), max(client_weight, self.eps)))
                
            for class_id, logit in client_logit.items():
                if class_id not in all_class_logits:
                    all_class_logits[class_id] = []
                
                # 计算该类别的置信度分数
                confidence = float(self._compute_confidence_scores(logit))
                # 结合质量和置信度的权重
                combined_weight = float(client_weight * confidence)  # 确保是标量
                combined_weight = max(combined_weight, self.eps)  # 确保权重不为零
                
                all_class_logits[class_id].append((np.array(logit, dtype=np.float32), combined_weight))
        
        # 加权聚合
        global_class_reps = {}
        global_logits = {}
        
        for class_id, reps_and_weights in all_class_reps.items():
            if reps_and_weights:
                # 使用聚类方法聚合表征
                global_class_reps[class_id] = self._aggregate_with_clustering(
                    reps_and_weights, use_spectral=True
                )
                
        for class_id, logits_and_weights in all_class_logits.items():
            if logits_and_weights:
                # 使用聚类方法聚合logits
                global_logits[class_id] = self._aggregate_with_clustering(
                    logits_and_weights, use_spectral=False
                )
        
        self.global_reps = global_class_reps
        self.global_logits = global_logits        
        return global_weight, train_loss_list
