# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 11:07
# @Describe:
import copy
import torch
from tqdm import tqdm
import numpy as np

from fl.client.fl_base import BaseClient

class Scaffold(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define Scaffold hyperparameters here
        self.cg = []
        self.c = []
        for param in self.model.parameters():
            self.c.append(torch.zeros_like(param, device=self.device))
            self.cg.append(torch.zeros_like(param, device=self.device))
        self.eta_l = kwargs.get('lr', 0.01)
        self.global_model = copy.deepcopy(self.model).to(self.device)  # 全局模型副本
        

    def local_train(self, sync_round: int, weights=None, cg=None):
        """
        训练方法，根据当前通信轮次(sync_round)进行相应的训练更新
        :param weights: 服务器传递过来的模型权重
        :param sync_round: 当前的通信轮次
        :param cg: 全局控制变量
        """
        # 1. 加载服务器传来的全局模型权重
        if weights is not None:
            self.update_weights(weights)
            # 更新全局模型副本
            self.global_model.load_state_dict(self.model.state_dict())
            for param in self.global_model.parameters():
                param.requires_grad = False
        
        # 2. 更新全局控制变量
        if cg is not None:
            self.cg = [torch.tensor(g, dtype=torch.float32, device=self.device).detach().clone() if isinstance(g, np.ndarray) else g.detach().clone().to(self.device) for g in cg]
        
        # 3. 开始本地训练
        self.model.train()
        total_loss = 0
        num_sample = len(self.train_loader.dataset)
        total_batches = len(self.train_loader) * self.epochs
        
        with tqdm(
            total=total_batches,
            desc=f"Client {self.client_id} Training Progress (Scaffold)"
        ) as pbar:
            for epoch in range(self.epochs):  # 多轮本地训练
                epoch_loss = 0
                for data, target in self.train_loader:  # 获取每个 batch
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()  # 清除之前的梯度
                    output = self.model(data)  # 前向传播
                    loss = self.loss(output, target)  # 计算损失
                    epoch_loss += loss.item()  # 累加损失
                    loss.backward()  # 反向传播，计算梯度
                    
                    # 手动更新参数，按照公式 y_i ← y_i - η_l(g_i(y_i) - c_i + c)
                    # 修改梯度：g_i ← g_i - c_i + c
                    for i, param in enumerate(self.model.parameters()):
                        if param.grad is not None:
                            param.grad = param.grad - self.c[i] + self.cg[i]
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    # 更新进度条
                    pbar.update(1)
                total_loss += epoch_loss
                avg_loss = epoch_loss / len(self.train_loader)
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}"
                })

        # Update c after local training is completed
        # 使用Option II更新控制变量: c_i ← c_i − c + 1/(Kηl) * (x − y_i)
        # 其中x是全局模型参数，y_i是本地模型参数
        global_params = list(self.global_model.parameters())
        local_params = list(self.model.parameters())
        
        # 更新本地控制变量
        with torch.no_grad():
            for i, (global_param, local_param) in enumerate(zip(global_params, local_params)):
                # 计算参数差: (x - y_i)
                param_diff = global_param.data - local_param.data
                
                # 按论文公式: c_i ← c_i − c + 1/(Kηl) * (x − y_i)
                # K是本地迭代次数，ηl是学习率
                update_term = param_diff / (self.eta_l * total_batches)
                self.c[i] = self.c[i].detach() - self.cg[i].detach() + update_term
        
        # 计算当前本地控制变量的副本（用于返回给服务器）
        c_current = [c.clone().detach() for c in self.c]

        # 4. 获取训练后的权重
        model_weights = self.get_weights(return_numpy=True)

        # 5. 返回更新后的权重给服务器，同时返回样本数、平均损失和当前控制变量
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        return model_weights, num_sample, avg_loss, c_current
