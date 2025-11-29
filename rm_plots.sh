#! /bin/bash

# 检查是否提供了数据集名称参数
if [ $# -eq 1 ]; then
    DATASET=$1
    echo "清除 $DATASET 数据集的所有图表、历史记录和日志..."
    rm -rf ./plots/$DATASET
    rm -rf ./logs/$DATASET
else
    echo "清除所有数据集的图表、历史记录和日志..."
    # 删除所有数据集目录
    rm -rf ./plots/mnist
    rm -rf ./plots/femnist
    rm -rf ./plots/cifar10
    rm -rf ./plots/cifar100
    rm -rf ./plots/tinyimagenet
    
    # 删除所有日志目录
    rm -rf ./logs/mnist
    rm -rf ./logs/femnist
    rm -rf ./logs/cifar10
    rm -rf ./logs/cifar100
    rm -rf ./logs/tinyimagenet
    rm -rf ./logs/**.log
    
    # 删除旧的图表和历史记录（兼容性）
    rm -f ./plots/*.png
    rm -f ./plots/history/*.pkl
    rm -f ./plots/comparison/*.png
fi

echo "清除完成！"