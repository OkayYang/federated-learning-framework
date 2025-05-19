# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/19 16:30
# @Describe: 聚合策略工厂

from fl.server.strategy.strategy import (
    FedAvgStrategy,
    FedProxStrategy,
    MoonStrategy,
    ScaffoldStrategy,
    FedDistillStrategy,
    FedGenStrategy,
    FedSPDStrategy,
    FedAloneStrategy
)


class StrategyFactory:
    """聚合策略工厂"""
    
    @staticmethod
    def get_strategy(strategy_name, server_kwargs):
        """
        根据策略名称获取对应的聚合策略实例
        
        Args:
            strategy_name: 策略名称
            server_kwargs: 服务器初始化时传入的参数
            
        Returns:
            AggregationStrategy: 聚合策略实例
        """
        strategy_map = {
            "fedavg": FedAvgStrategy(),
            "fedprox": FedProxStrategy(),
            "moon": MoonStrategy(),
            "scaffold": ScaffoldStrategy(),
            "feddistill": FedDistillStrategy(),
            "fedgen": FedGenStrategy(),
            "fedspd": FedSPDStrategy(),
            "fedalone": FedAloneStrategy()
        }
        
        if strategy_name.lower() not in strategy_map:
            raise ValueError(f"不支持的策略: {strategy_name}")
        
        strategy = strategy_map[strategy_name.lower()]
        strategy.initialize(server_kwargs)
        
        return strategy 