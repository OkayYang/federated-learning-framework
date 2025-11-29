# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 11:07
# @Describe: SCAFFOLD算法客户端实现 - 使用控制变量减少客户端漂移

import copy
import torch
from tqdm import tqdm
import numpy as np

from fl.client.fl_base import BaseClient

class ScaffoldOptimizer(torch.optim.Optimizer):
    """
    SCAFFOLD优化器实现 - 直接在优化步骤中应用控制变量修正
    """
    def __init__(self, params, lr, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):
        """
        执行优化步骤，应用控制变量修正
        :param server_controls: 服务器控制变量
        :param client_controls: 客户端控制变量
        :param closure: 闭包函数
        :return: 损失值
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                # 应用SCAFFOLD修正: g_i = g_i + (c - c_i)
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

        return loss

class Scaffold(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 定义Scaffold超参数和控制变量
        # 服务器控制变量(论文中的c)，用于校正本地更新
        self.scv = copy.deepcopy(self.model)
        # 客户端控制变量(论文中的c_i)，用于记录本地梯度偏差
        self.ccv = copy.deepcopy(self.model)
        # 本地学习率
        self.eta_l = kwargs.get('lr', 0.01)
        # 保存全局模型副本，用于计算参数差异
        self.global_model = copy.deepcopy(self.model).to(self.device)
        
    def local_train(self, sync_round: int, weights=None, cg=None):
        """
        SCAFFOLD本地训练方法，根据当前通信轮次(sync_round)进行相应的训练更新
        :param weights: 服务器传递过来的模型权重
        :param sync_round: 当前的通信轮次
        :param cg: 全局控制变量(服务器聚合后的控制变量)
        :return: 更新后的模型权重、样本数量、训练损失和更新后的客户端控制变量
        """
        # 1. 加载服务器传来的全局模型权重
        if weights is not None:
            self.update_weights(weights)
            # 更新全局模型副本，用于后续计算参数差异
            self.global_model.load_state_dict(self.model.state_dict())
            for param in self.global_model.parameters():
                param.requires_grad = False  # 全局模型不需要计算梯度
        
        # 2. 更新服务器控制变量(scv)
        if cg is not None:
            # 将全局控制变量加载到scv中
            scv_state_dict = self.scv.state_dict()
            for i, (name, _) in enumerate(self.scv.named_parameters()):
                # 处理不同数据类型的控制变量
                if isinstance(cg[i], np.ndarray):
                    scv_state_dict[name] = torch.tensor(cg[i], dtype=torch.float32, device=self.device)
                else:
                    scv_state_dict[name] = cg[i].detach().clone().to(self.device)
            self.scv.load_state_dict(scv_state_dict)
        
        # 3. 准备训练
        self.model.train()
        total_loss = 0
        num_sample = len(self.train_loader.dataset)  # 本地数据集大小
        
        # 保存训练前的全局模型状态，用于后续更新控制变量
        global_state_dict = copy.deepcopy(self.model.state_dict())
        
        # 4. 创建ScaffoldOptimizer替代标准优化器
        # 提取服务器和客户端控制变量的状态字典
        server_controls = {}
        client_controls = {}
        for name, param in self.model.named_parameters():
            server_controls[name] = self.scv.state_dict()[name]
            client_controls[name] = self.ccv.state_dict()[name]

        # 创建自定义SCAFFOLD优化器
        scaffold_optimizer = ScaffoldOptimizer(
            self.model.parameters(), 
            lr=self.eta_l, 
            weight_decay=1e-4 if hasattr(self.optimizer, 'weight_decay') else 0
        )
        
        # 记录本地训练步数
        local_steps = 0
        
        # 使用进度条显示训练进度
        total_batches = len(self.train_loader) * self.epochs
        from fl.utils import ClientProgress
        
        with ClientProgress(
            progress_actor=self.progress_actor,
            client_id=self.client_id,
            total=total_batches,
            desc=f"Client {self.client_id} Training (Scaffold)"
        ) as pbar:
            for epoch in range(self.epochs):  # 多轮本地训练
                epoch_loss = 0
                for data, target in self.train_loader:  # 获取每个batch数据
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 前向传播
                    output = self.model(data)
                    loss = self.loss(output, target)
                    epoch_loss += loss.item()
                    
                    # 清除之前的梯度
                    scaffold_optimizer.zero_grad()
                    
                    # 反向传播，计算梯度
                    loss.backward()
                    
                    # 使用SCAFFOLD优化器更新参数
                    scaffold_optimizer.step(server_controls, client_controls)
                    
                    local_steps += 1
                    # 更新进度条
                    pbar.update(1)
                
                # 计算并显示每个epoch的平均损失
                total_loss += epoch_loss
                avg_loss = epoch_loss / len(self.train_loader)
                # 获取当前学习率
                current_lr = scaffold_optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.epochs}",
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
                
        # 应用学习率调度(如果有)
        if self.scheduler:
            self.scheduler.step()
        
        # 5. 更新客户端控制变量
        # 使用Option II更新控制变量: c_i ← c_i - c + 1/(Kηl) * (x − y_i)
        # 其中x是全局模型参数，y_i是本地更新后的模型参数
        new_ccv_state = copy.deepcopy(self.ccv.state_dict())
        current_state_dict = self.model.state_dict()  # 当前模型状态
        
        # 更新本地控制变量
        for key in new_ccv_state:
            if key in global_state_dict and key in current_state_dict:
                # 计算参数差异: (x - y_i)
                param_diff = global_state_dict[key] - current_state_dict[key]
                
                # 按论文公式更新客户端控制变量: c_i ← c_i − c + 1/(Kηl) * (x − y_i)
                # K是本地迭代次数，ηl是学习率
                new_ccv_state[key] = self.ccv.state_dict()[key] - self.scv.state_dict()[key] + param_diff / (local_steps * self.eta_l)
        
        # 将更新后的控制变量加载回客户端控制变量模型
        self.ccv.load_state_dict(new_ccv_state)
        
        # 6. 准备返回数据
        # 提取客户端控制变量参数，用于返回给服务器
        c_current = []
        for _, param in self.ccv.named_parameters():
            c_current.append(param.data.clone().detach())

        # 获取训练后的模型权重
        model_weights = self.get_weights(return_numpy=True)

        # 计算平均损失
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        
        # 返回更新后的权重、样本数量、训练损失和客户端控制变量
        return model_weights, num_sample, avg_loss, c_current
