import torch
import numpy as np
import copy
from scipy.spatial.distance import cdist

from fl.aggregation.aggregator import average_weight
from fl.server.strategy.strategy_base import AggregationStrategy


class FedSPDStrategy(AggregationStrategy):
    """FedSPD聚合策略 - 使用高级聚类和动态引导技术"""
    def __init__(self):
        self.global_reps = None
        self.global_logits = None
        self.prev_global_reps = None  # 保存上一轮的全局表征，用于稳定性和方向性计算
        self.eps = 1e-10  # 添加一个小的常数避免除零
        
        # 聚类参数
        self.n_clusters = 3  # K-means聚类数量，可以根据实际情况调整
        self.use_kmeans = True  # 是否使用K-means聚类
        self.use_spectral = True  # 是否使用谱聚类
        self.cluster_weight_power = 2.0  # 聚类权重指数，用于调整聚类中心的影响力
        
        # 动态引导参数
        self.use_momentum = True  # 使用动量平滑全局表征更新
        self.momentum = 0.8  # 动量系数
        self.use_direction_alignment = True  # 使用方向性引导
        self.direction_weight = 0.5  # 方向性引导权重
        self.reference_model = None  # 参考模型（理想情况是全局最优的预训练模型）
        self.use_gradual_adaptation = True  # 使用渐进式适应
        self.round_adaption_rate = 0.95  # 随着轮次增加适应率
    
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
    
    def _calculate_model_improvement_direction(self, prev_global_reps, current_reps, class_id):
        """计算模型改进的方向"""
        if prev_global_reps is None or class_id not in prev_global_reps:
            return None
            
        prev_rep = prev_global_reps[class_id]
        # 计算从上一轮到当前的差向量，表示改进方向
        improvement_direction = current_reps - prev_rep
        # 归一化方向向量
        norm = np.linalg.norm(improvement_direction)
        if norm > self.eps:
            improvement_direction = improvement_direction / norm
        return improvement_direction
    
    def _apply_momentum_update(self, global_class_reps, prev_global_reps, round_num):
        """应用动量更新全局表征，使聚合结果更加平滑"""
        if not self.use_momentum or prev_global_reps is None:
            return global_class_reps
            
        updated_reps = {}
        # 动态调整动量系数 - 随着轮次增加逐渐减小动量
        adjusted_momentum = self.momentum * (self.round_adaption_rate ** round_num)
        
        for class_id, rep in global_class_reps.items():
            if class_id in prev_global_reps:
                # 动量更新：new_rep = momentum * prev_rep + (1-momentum) * current_rep
                updated_reps[class_id] = adjusted_momentum * prev_global_reps[class_id] + (1 - adjusted_momentum) * rep
            else:
                updated_reps[class_id] = rep
                
        return updated_reps
    
    def _apply_direction_guidance(self, global_class_reps, prev_global_reps, round_num):
        """应用方向性引导，确保表征沿着一致的方向改进"""
        if not self.use_direction_alignment or prev_global_reps is None:
            return global_class_reps
            
        guided_reps = {}
        # 动态调整方向权重 - 随着轮次增加逐渐增加方向权重
        adjusted_direction_weight = min(0.8, self.direction_weight * (1.0 / self.round_adaption_rate ** round_num))
        
        for class_id, rep in global_class_reps.items():
            # 计算改进方向
            direction = self._calculate_model_improvement_direction(prev_global_reps, rep, class_id)
            
            if direction is not None and class_id in prev_global_reps:
                # 结合原有表征和方向引导：new_rep = rep + direction_weight * direction
                guided_reps[class_id] = rep + adjusted_direction_weight * direction
            else:
                guided_reps[class_id] = rep
                
        return guided_reps
    
    def _detect_and_handle_outliers(self, client_qualities, threshold=1.5):
        """检测并处理异常值 - 使用IQR方法"""
        qualities = np.array(client_qualities)
        q1 = np.percentile(qualities, 25)
        q3 = np.percentile(qualities, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # 将异常值限制在范围内
        clipped_qualities = np.clip(qualities, lower_bound, upper_bound)
        return clipped_qualities.tolist()
    
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
        
        # 检测并处理异常的质量分数
        client_qualities = self._detect_and_handle_outliers(client_qualities)
        
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
        
        # 保存上一轮的全局表征用于方向性计算
        self.prev_global_reps = copy.deepcopy(self.global_reps) if self.global_reps else None
        
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
        
        # 应用动量平滑聚合结果
        if self.use_momentum:
            global_class_reps = self._apply_momentum_update(global_class_reps, self.prev_global_reps, round_num)
            
        # 应用方向性引导
        if self.use_direction_alignment:
            global_class_reps = self._apply_direction_guidance(global_class_reps, self.prev_global_reps, round_num)
        
        self.global_reps = global_class_reps
        self.global_logits = global_logits
        
        return global_weight, train_loss_list