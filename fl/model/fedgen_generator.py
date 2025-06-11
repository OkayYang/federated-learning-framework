import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class FedGenGenerator(nn.Module):
    """
    生成器模型，用于生成合成数据样本
    代码参考：https://github.com/zhuangdizhu/FedGen/
    """
    def __init__(self, feature_dim, num_classes,noise_dim=64, hidden_dim=256):
        super(FedGenGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 嵌入层，将类别标签转换为嵌入向量
        self.label_embedding = nn.Embedding(num_classes, hidden_dim)

        # 生成器网络
        self.generator = nn.Sequential(
            nn.Linear(noise_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        # 训练信息
        self.train_epochs = 20
        self.train_batch_size = 64
        self.train_lr = 0.001
        self.ensemble_alpha = 1
        self.ensemble_beta = 0
        self.ensemble_eta = 1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.train_lr)
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.loss_fn = nn.CrossEntropyLoss()
        self.diversity_loss = DiversityLoss(metric='l1')

    
    def forward(self,labels):
        """
        输入噪声向量和标签，输出合成特征
        """
        z = torch.randn(labels.size(0), self.noise_dim).to(self.device)
        # 将标签转换为嵌入向量
        label_embedding = self.label_embedding(labels)
        # 将噪声向量和标签嵌入连接起来
        concat_input = torch.cat([z, label_embedding], dim=1)
        # 生成特征
        features = self.generator(concat_input)
        # 使用tanh激活函数替代硬裁剪，提供更平滑的梯度
        features = torch.tanh(features) * 5.0  # 将输出范围限制在[-5, 5]

        return z,features

    def get_weights(self, return_numpy=False):
        if not return_numpy:
            return {k: v.cpu() for k, v in self.state_dict().items()}
        else:
            weights_list = []
            for v in self.state_dict().values():
                weights_list.append(v.cpu().numpy())
            return [e.copy() for e in weights_list]

    def update_weights(self, weights: np.ndarray):
        if len(weights) != len(self.state_dict()):
            raise ValueError("传入的权重数组数量与模型参数数量不匹配。")
        keys = self.state_dict().keys()
        weights_dict = {}
        for k, v in zip(keys, weights):
            weights_dict[k] = torch.Tensor(np.copy(v)).to(self.device)
        self.load_state_dict(weights_dict)

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))