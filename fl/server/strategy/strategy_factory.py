# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2025/05/19 16:30
# @Describe: 聚合策略工厂

from fl.server.strategy.strategy_base import (
    AggregationStrategy, 
    FedAloneStrategy,
    FedAvgStrategy, 
    FedProxStrategy,
    MoonStrategy
)
from fl.server.strategy.fedftg_strategy import FedFTGStrategy
from fl.server.strategy.fedgen_strategy import FedGenStrategy
from fl.server.strategy.fedspd_strategy import FedSPDStrategy
from fl.server.strategy.scaffold_strategy import ScaffoldStrategy
from fl.server.strategy.feddistill_strategy import FedDistillStrategy
from fl.server.strategy.fedgkd_strategy import FedGKDStrategy
from fl.server.strategy.fedspdlc_strategy import FedSPDLCStrategy

class StrategyFactory:
    """聚合策略工厂"""
    
    @staticmethod
    def get_strategy(strategy_name, kwargs=None):
        """
        根据策略名称获取聚合策略实例
        :param strategy_name: 策略名称, eg. FedAvg, FedProx
        :return: 策略实例
        """
        if kwargs is None:
            kwargs = {}
            
        strategy_name = strategy_name.lower()
        if strategy_name == "fedavg":
            strategy = FedAvgStrategy()
        elif strategy_name == "fedprox":
            strategy = FedProxStrategy()
        elif strategy_name == "moon":
            strategy = MoonStrategy()
        elif strategy_name == "fedalone":
            strategy = FedAloneStrategy()
        elif strategy_name == "scaffold":
            strategy = ScaffoldStrategy()
        elif strategy_name == "feddistill":
            strategy = FedDistillStrategy()
        elif strategy_name == "fedgen":
            strategy = FedGenStrategy()
        elif strategy_name == "fedspd":
            strategy = FedSPDStrategy()
        elif strategy_name == "fedspd-lc":
            strategy = FedSPDLCStrategy()
        elif strategy_name == "fedftg":
            strategy = FedFTGStrategy()
        elif strategy_name == "fedgkd":
            strategy = FedGKDStrategy()
        else:
            raise ValueError(f"不支持的策略名称: {strategy_name}")
            
        # 初始化策略（如果有特定参数）
        if hasattr(strategy, "initialize"):
            strategy.initialize(kwargs)
            
        return strategy 