#!/bin/bash

# 顺序执行联邦学习实验脚本
# 该脚本按顺序运行多个数据集的实验，并在后台执行

# 创建日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/sequential_experiment_$(date +%Y%m%d_%H%M%S).log"

# 将所有输出重定向到日志文件，并在后台运行
{
    echo "开始顺序执行联邦学习实验..."

    # 运行CIFAR10实验
    echo "========================================"
    echo "  开始CIFAR10数据集实验"
    echo "========================================"
    ./run_comparison.sh cifar10
    echo "CIFAR10实验已完成"

    # 等待5秒
    echo "等待5秒..."
    sleep 5

    # 运行CIFAR100实验
    echo "========================================"
    echo "  开始CIFAR100数据集实验"
    echo "========================================"
    ./run_comparison.sh cifar100
    echo "CIFAR100实验已完成"

    echo "========================================"
    echo "  CIFAR实验已完成!"
    echo "========================================"
    echo "结果图表已保存到 ./plots 目录"

    # 等待5秒
    echo "等待5秒..."
    sleep 5

    # 运行SVHN实验
    echo "========================================"
    echo "  开始SVHN数据集实验"
    echo "========================================"
    ./run_comparison.sh svhn
    echo "SVHN实验已完成"

    # 运行TinyImageNet实验
    echo "========================================"
    echo "  开始TinyImageNet数据集实验"
    echo "========================================"
    ./run_comparison.sh tinyimagenet
    echo "TinyImageNet实验已完成"

    # 等待5秒
    echo "等待5秒..."
    sleep 5

    echo "========================================"
    echo "  所有实验已完成!"
    echo "========================================"
    echo "结果图表已保存到 ./plots 目录"
    echo "实验日志已保存到 $LOG_FILE"
    echo "========================================"
} > "$LOG_FILE" 2>&1 &

# 输出进程ID和日志文件位置
PROCESS_ID=$!
echo "实验已在后台启动，进程ID: $PROCESS_ID"
echo "日志文件: $LOG_FILE"
echo "可以使用 'tail -f $LOG_FILE' 命令查看实时日志"