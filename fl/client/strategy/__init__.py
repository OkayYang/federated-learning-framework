# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/6 20:30
# @Describe:
from torch.utils.data import DataLoader

from fl.client.fl_base import ModelConfig
from fl.client.strategy.fed_alone import FedAlone
from fl.client.strategy.fed_avg import FedAvg
from fl.client.strategy.fed_distill import FedDistill
from fl.client.strategy.fed_prox import FedProx
from fl.client.strategy.fed_spd import FedSPD
from fl.client.strategy.fed_ftg import FedFTG
from fl.client.strategy.fed_spdlc import FedSPDLC
from fl.client.strategy.moon import Moon
from fl.client.strategy.scaffold import Scaffold
from fl.client.strategy.fed_gen import FedGen
from fl.client.strategy.fed_gkd import FedGKD

# 策略映射字典
_strategy_map = {
    "fedalone": FedAlone,
    "fedavg": FedAvg,
    "feddistill": FedDistill,
    "fedprox": FedProx,
    "fedspd": FedSPD,
    "moon": Moon,
    "scaffold": Scaffold,
    "fedgen": FedGen,
    "fedftg": FedFTG,
    "fedgkd": FedGKD,
    "fedspd-lc": FedSPDLC
}

def create_client(
        strategy: str,
        client_id: str,
        model_config: ModelConfig,
        client_dataset_dict,
        **kwargs
):
    """构建模型并返回，自动设置损失函数和优化器"""
    if model_config.model_fn is None:
        raise ValueError("Model function is required.")
    if model_config.loss_fn is None:
        raise ValueError("Loss function is required.")
    if model_config.optim_fn is None:
        raise ValueError("Optimizer function is required.")

    # 创建模型
    model = model_config.get_model()
    # 设置损失函数
    loss = model_config.get_loss_fn()
    # 设置优化器
    optimizer = model_config.get_optimizer(model.parameters())
    # 设置调度器
    scheduler = model_config.get_scheduler(optimizer)
    epochs = model_config.get_epochs()
    batch_size = model_config.get_batch_size()

    client_dataset = client_dataset_dict[client_id]
    global_test_dataset = client_dataset_dict["global"]["test_dataset"]
    train_dataLoader = DataLoader(client_dataset['train_dataset'], batch_size=batch_size, shuffle=True,drop_last=True)
    test_dataLoader = DataLoader(client_dataset['test_dataset'], batch_size=batch_size, shuffle=True,drop_last=True)
    global_test_dataLoader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

    strategy = strategy.lower()
    if strategy not in _strategy_map:
        raise ValueError(f"Unknown strategy: {strategy}. Available strategies: {list(_strategy_map.keys())}")
    
    client_class = _strategy_map[strategy]
    # Wrap with Ray Actor
    from fl.client.ray_client import RayClientActor
    # Set num_cpus to a small value to allow many clients to coexist if needed, 
    # or let Ray manage it. Default is 1. 
    # If we have many clients, we might want to set num_cpus < 1 if they are idle most of the time,
    # but here they are active during training. 
    # Let's assume 1 CPU per client for now or let user configure.
    # For simulation on single machine with many clients, we might want to limit resources per actor
    # so we can create many of them, OR we create them on demand.
    # However, the current architecture creates all clients at startup.
    # To avoid OOM or resource exhaustion with many clients, we can specify num_cpus=0.1 or similar,
    # or rely on Ray's scheduling.
    # Given the user asked for "distributed", we assume they have resources or want parallel execution.
    # We'll use default resources for now.
    return RayClientActor.remote(
        client_class,
        client_id,
        model,
        loss,
        optimizer,
        epochs,
        batch_size,
        train_dataLoader,
        test_dataLoader,
        global_test_dataLoader,
        scheduler,  # 传递调度器
        **kwargs
    )

# 打印可用策略
print(f"Available strategies: {list(_strategy_map.keys())}")