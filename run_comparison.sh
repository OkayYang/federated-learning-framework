#!/bin/bash

# 联邦学习算法比较实验脚本
# 这个脚本会运行FedAvg, FedProx, MOON, Scaffold和FedGen算法，并生成对比结果
# 使用方法: ./run_comparison.sh [数据集名称]
# 示例: ./run_comparison.sh mnist

echo "========================================"
echo "  联邦学习算法比较实验"
echo "========================================"

# 检查命令行参数
if [ $# -eq 0 ]; then
    echo "错误: 未指定数据集名称"
    echo "使用方法: ./run_comparison.sh [数据集名称]"
    echo "支持的数据集: mnist, femnist, cifar10, cifar100"
    exit 1
fi

# 从命令行参数获取数据集名称
DATASET="$1"

# 验证数据集名称
if [[ "$DATASET" != "mnist" && "$DATASET" != "femnist" && "$DATASET" != "cifar10" && "$DATASET" != "cifar100" ]]; then
    echo "错误: 不支持的数据集 '$DATASET'"
    echo "支持的数据集: mnist, femnist, cifar10, cifar100"
    exit 1
fi

echo "选择的数据集: $DATASET"

# 设置基本参数
BATCH_SIZE=64    # 批处理大小
LOCAL_EPOCHS=3  # 本地训练轮数
COMM_ROUNDS=3  # 通信轮数
LEARNING_RATE=0.01  # 学习率
OPTIMIZER="adam"    # 优化器: adam, sgd
SEED=42             # 随机种子，保证实验可重复性
PARTITION="dirichlet"   # 数据分区方式: iid, noiid, dirichlet
NUM_CLIENTS=10      # 客户端数量
DIR_BETA=0.3       # Dirichlet分布参数，仅在PARTITION="dirichlet"时使用

# 创建日志目录
LOG_DIR="./logs/${DATASET}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/experiment_$(date +%Y%m%d_%H%M%S).log"

# 记录实验参数到日志
{
    echo "========================================"
    echo "  联邦学习算法比较实验"
    echo "========================================"
    echo "数据集: $DATASET"
    echo "批处理大小: $BATCH_SIZE"
    echo "本地训练轮数: $LOCAL_EPOCHS"
    echo "通信轮数: $COMM_ROUNDS"
    echo "学习率: $LEARNING_RATE"
    echo "优化器: $OPTIMIZER"
    echo "随机种子: $SEED"
    echo "数据分区方式: $PARTITION"
    echo "客户端数量: $NUM_CLIENTS"
    echo "Dirichlet参数: $DIR_BETA"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

# 检查CUDA
python check_cuda.py | tee -a "$LOG_FILE"

# 运行FedAlone算法
{
    echo ""
    echo "========================================"
    echo "  运行FedAlone算法"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

python main.py \
    --dataset $DATASET \
    --strategy fedalone \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --seed $SEED \
    --partition $PARTITION \
    --dir_beta $DIR_BETA \
    --num_clients $NUM_CLIENTS 2>&1 | tee -a "$LOG_FILE"

# 运行FedAvg算法
{
    echo ""
    echo "========================================"
    echo "  运行FedAvg算法"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

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
    --dir_beta $DIR_BETA \
    --num_clients $NUM_CLIENTS 2>&1 | tee -a "$LOG_FILE"


# 运行FedProx算法
{
    echo ""
    echo "========================================"
    echo "  运行FedProx算法"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

python main.py \
    --dataset $DATASET \
    --strategy fedprox \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --seed $SEED \
    --partition $PARTITION \
    --dir_beta $DIR_BETA \
    --num_clients $NUM_CLIENTS 2>&1 | tee -a "$LOG_FILE"

# 运行Scaffold算法
{
    echo ""
    echo "========================================"
    echo "  运行Scaffold算法"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

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
    --dir_beta $DIR_BETA \
    --num_clients $NUM_CLIENTS 2>&1 | tee -a "$LOG_FILE"

# 运行MOON算法
{
    echo ""
    echo "========================================"
    echo "  运行MOON算法"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

python main.py \
    --dataset $DATASET \
    --strategy moon \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --seed $SEED \
    --partition $PARTITION \
    --dir_beta $DIR_BETA \
    --num_clients $NUM_CLIENTS 2>&1 | tee -a "$LOG_FILE"

# 运行FedDistill算法
{
    echo ""
    echo "========================================"
    echo "  运行FedDistill算法"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

python main.py \
    --dataset $DATASET \
    --strategy feddistill \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --seed $SEED \
    --partition $PARTITION \
    --dir_beta $DIR_BETA \
    --num_clients $NUM_CLIENTS 2>&1 | tee -a "$LOG_FILE"


# 运行FedGen算法
{
    echo ""
    echo "========================================"
    echo "  运行FedGen算法"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

python main.py \
    --dataset $DATASET \
    --strategy fedgen \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --seed $SEED \
    --partition $PARTITION \
    --dir_beta $DIR_BETA \
    --num_clients $NUM_CLIENTS 2>&1 | tee -a "$LOG_FILE"


# 运行FedSPD算法
{
    echo ""
    echo "========================================"
    echo "  运行FedSPD算法"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

python main.py \
    --dataset $DATASET \
    --strategy fedspd \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --seed $SEED \
    --partition $PARTITION \
    --dir_beta $DIR_BETA \
    --num_clients $NUM_CLIENTS 2>&1 | tee -a "$LOG_FILE"


# 生成对比结果图表
{
    echo ""
    echo "========================================"
    echo "  生成算法对比结果"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

# 运行对比脚本，传入数据集名称
python compare_strategies.py $DATASET 2>&1 | tee -a "$LOG_FILE"

{
    echo ""
    echo "========================================"
    echo "  算法对比实验完成!"
    echo "========================================"
    echo "对比图表已保存到 ./plots/$DATASET 目录"
    echo "实验日志已保存到 $LOG_FILE"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"
