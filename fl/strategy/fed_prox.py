# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 15:47
# @Describe:
from fl.fl_base import BaseClient


class FedProx(BaseClient):
    def local_train(self, sync_round: int, weights=None):
        pass
