#!/bin/bash

# 联邦学习算法比较实验脚本
# 这个脚本会运行FedAvg, FedProx和MOON三种算法，并生成对比结果

echo "========================================"
echo "  联邦学习算法比较实验"
echo "========================================"

# 设置基本参数
DATASET="mnist"  # 可选: femnist, mnist
BATCH_SIZE=64
LOCAL_EPOCHS=20
COMM_ROUNDS=100
LEARNING_RATE=0.01
OPTIMIZER="adam"
SEED=42
PARTITION="noiid" 
NUM_CLIENTS=10  
DIR_BETA=0.4




# FedProx和MOON的特定参数
FEDPROX_MU=0.01
MOON_MU=1.0
MOON_TEMP=0.5

# 运行FedAvg算法
echo ""
echo "========================================"
echo "  运行FedAvg算法"
echo "========================================"
echo ""

python main.py \
    --dataset $DATASET \
    --strategy fedavg \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --seed $SEED \
    --partition $PARTITION \
    --num_clients $NUM_CLIENTS


# 运行FedProx算法
echo ""
echo "========================================"
echo "  运行FedProx算法"
echo "========================================"
echo ""

python main.py \
    --dataset $DATASET \
    --strategy fedprox \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --mu $FEDPROX_MU \
    --seed $SEED \
    --partition $PARTITION \
    --num_clients $NUM_CLIENTS


# 运行MOON算法
echo ""
echo "========================================"
echo "  运行MOON算法"
echo "========================================"
echo ""

python main.py \
    --dataset $DATASET \
    --strategy moon \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --mu $MOON_MU \
    --temperature $MOON_TEMP \
    --seed $SEED \
    --partition $PARTITION \
    --num_clients $NUM_CLIENTS


# 运行Scaffold算法
echo ""
echo "========================================"
echo "  运行Scaffold算法"
echo "========================================"
echo ""

python main.py \
    --dataset $DATASET \
    --strategy scaffold \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --seed $SEED \
    --partition $PARTITION \
    --num_clients $NUM_CLIENTS

# 复制结果到指定目录


# 生成对比结果图表
echo ""
echo "========================================"
echo "  生成算法对比结果"
echo "========================================"
echo ""


# 运行对比脚本
python compare_strategies.py

echo ""
echo "========================================"
echo "  算法对比实验完成!"
echo "========================================"
echo "对比图表已保存到 ./plots 目录"
echo "========================================" 
echo ""
