#!/bin/bash

# 运行单个联邦学习算法并保存日志
# 使用方法: ./run_single.sh [数据集名称] [算法名称]
# 示例: ./run_single.sh mnist fedavg

echo "========================================"
echo "  联邦学习单算法实验"
echo "========================================"

# 检查命令行参数
if [ $# -lt 2 ]; then
    echo "错误: 未指定足够的参数"
    echo "使用方法: ./run_single.sh [数据集名称] [算法名称]"
    echo "支持的数据集: mnist, femnist, cifar10, cifar100"
    echo "支持的算法: fedavg, fedprox, moon, scaffold, feddistill, fedgen, fedspd, fedalone"
    exit 1
fi

# 从命令行参数获取数据集名称和算法名称
DATASET="$1"
STRATEGY="$2"

# 验证数据集名称
if [[ "$DATASET" != "mnist" && "$DATASET" != "femnist" && "$DATASET" != "cifar10" && "$DATASET" != "cifar100" ]]; then
    echo "错误: 不支持的数据集 '$DATASET'"
    echo "支持的数据集: mnist, femnist, cifar10, cifar100"
    exit 1
fi

# 验证算法名称
if [[ "$STRATEGY" != "fedavg" && "$STRATEGY" != "fedprox" && "$STRATEGY" != "moon" && 
      "$STRATEGY" != "scaffold" && "$STRATEGY" != "feddistill" && "$STRATEGY" != "fedgen" && 
      "$STRATEGY" != "fedspd" && "$STRATEGY" != "fedalone" ]]; then
    echo "错误: 不支持的算法 '$STRATEGY'"
    echo "支持的算法: fedavg, fedprox, moon, scaffold, feddistill, fedgen, fedspd, fedalone"
    exit 1
fi

echo "选择的数据集: $DATASET"
echo "选择的算法: $STRATEGY"

# 设置基本参数
BATCH_SIZE=64    # 批处理大小
LOCAL_EPOCHS=5   # 本地训练轮数
COMM_ROUNDS=30   # 通信轮数
LEARNING_RATE=0.01  # 学习率
OPTIMIZER="adam"    # 优化器: adam, sgd
SEED=42             # 随机种子，保证实验可重复性
PARTITION="dirichlet"   # 数据分区方式: iid, noiid, dirichlet
NUM_CLIENTS=10      # 客户端数量
DIR_BETA=0.3        # Dirichlet分布参数，仅在PARTITION="dirichlet"时使用

# 创建日志目录
LOG_DIR="./logs/${DATASET}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${STRATEGY}_$(date +%Y%m%d_%H%M%S).log"

# 记录实验参数到日志
{
    echo "========================================"
    echo "  联邦学习单算法实验"
    echo "========================================"
    echo "数据集: $DATASET"
    echo "算法: $STRATEGY"
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

# 运行算法
{
    echo ""
    echo "========================================"
    echo "  运行 $STRATEGY 算法"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE"

python main.py \
    --dataset $DATASET \
    --strategy $STRATEGY \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --comm_rounds $COMM_ROUNDS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --seed $SEED \
    --partition $PARTITION \
    --dir_beta $DIR_BETA \
    --num_clients $NUM_CLIENTS 2>&1 | tee -a "$LOG_FILE"

{
    echo ""
    echo "========================================"
    echo "  实验完成!"
    echo "========================================"
    echo "图表已保存到 ./plots/$DATASET 目录"
    echo "实验日志已保存到 $LOG_FILE"
    echo "========================================"
    echo ""
} | tee -a "$LOG_FILE" 