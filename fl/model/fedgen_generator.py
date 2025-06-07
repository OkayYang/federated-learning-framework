import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FedGenGenerator(nn.Module):
    """
    生成器模型，用于生成合成数据样本
    基于原论文实现: https://github.com/zhuangdizhu/FedGen
    """
    def __init__(self, feature_dim, num_classes, noise_dim=10, hidden_dim=256, dataset_name="cifar10"):
        """
        初始化生成器模型
        
        Args:
            feature_dim: 特征维度，对应模型最后一层全连接层的输入维度
            num_classes: 类别数量
            noise_dim: 噪声维度，原论文默认为10
            hidden_dim: 隐藏层维度，原论文默认为256
            dataset_name: 数据集名称，用于设置特定参数
        """
        super(FedGenGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = dataset_name.lower()

        # 嵌入层，将类别标签转换为嵌入向量
        self.label_embedding = nn.Embedding(num_classes, hidden_dim)

        # 生成器网络结构
        self.generator = nn.Sequential(
            nn.Linear(noise_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()
        )

        # 根据数据集设置训练参数
        self._set_dataset_params()
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.train_lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.diversity_loss = DiversityLoss(metric='l1')
        
    def _set_dataset_params(self):
        """根据不同数据集设置特定参数"""
        # 默认参数
        self.train_epochs = 10
        self.train_batch_size = 32
        self.train_lr = 0.001
        self.ensemble_alpha = 1.0
        self.ensemble_beta = 0.0
        self.ensemble_eta = 1.0
        
        # 针对不同数据集的特定参数
        if self.dataset_name == "mnist":
            self.train_epochs = 10
            self.train_batch_size = 32
            self.ensemble_alpha = 1.0
            self.ensemble_beta = 0.0
            self.ensemble_eta = 1.0
        elif self.dataset_name == "femnist" or self.dataset_name == "emnist":
            self.train_epochs = 10
            self.train_batch_size = 32
            self.ensemble_alpha = 1.0
            self.ensemble_beta = 0.0
            self.ensemble_eta = 1.0
        elif self.dataset_name == "cifar10":
            self.train_epochs = 20
            self.train_batch_size = 32
            self.ensemble_alpha = 10.0
            self.ensemble_beta = 1.0
            self.ensemble_eta = 5.0
        elif self.dataset_name == "cifar100":
            self.train_epochs = 50
            self.train_batch_size = 32
            self.ensemble_alpha = 10.0
            self.ensemble_beta = 1.0
            self.ensemble_eta = 5.0
    
    def forward(self, labels):
        """
        输入标签，输出噪声向量和合成特征
        
        Args:
            labels: 标签张量
            
        Returns:
            tuple: (噪声向量, 生成的特征)
        """
        batch_size = labels.size(0)
        z = torch.randn(batch_size, self.noise_dim).to(self.device)
        
        # 将标签转换为嵌入向量
        label_embedding = self.label_embedding(labels)
        
        # 将噪声向量和标签嵌入连接起来
        concat_input = torch.cat([z, label_embedding], dim=1)
        
        # 生成特征
        features = self.generator(concat_input)
        
        return z, features

    def get_weights(self, return_numpy=False):
        """获取模型权重"""
        if not return_numpy:
            return {k: v.cpu() for k, v in self.state_dict().items()}
        else:
            weights_list = []
            for v in self.state_dict().values():
                weights_list.append(v.cpu().numpy())
            return [e.copy() for e in weights_list]

    def update_weights(self, weights):
        """更新模型权重"""
        if len(weights) != len(self.state_dict()):
            raise ValueError("传入的权重数组数量与模型参数数量不匹配。")
        keys = self.state_dict().keys()
        weights_dict = {}
        for k, v in zip(keys, weights):
            weights_dict[k] = torch.Tensor(np.copy(v)).to(self.device)
        self.load_state_dict(weights_dict)

class DiversityLoss(nn.Module):
    """
    多样性损失，鼓励生成多样化的特征
    原作者实现: https://github.com/zhuangdizhu/FedGen
    """
    def __init__(self, metric):
        """
        初始化多样性损失
        
        Args:
            metric: 距离度量方式，可选 'l1', 'l2', 'cosine'
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """计算两个张量之间的距离"""
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(f"不支持的度量方式: {metric}")

    def pairwise_distance(self, tensor, how):
        """计算张量行之间的成对距离"""
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        前向传播计算多样性损失
        
        Args:
            noises: 噪声张量
            layer: 生成的特征层
        
        Returns:
            torch.Tensor: 多样性损失值
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))