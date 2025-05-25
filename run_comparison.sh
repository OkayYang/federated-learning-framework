#!/bin/bash

# 联邦学习算法比较实验脚本
# 这个脚本会运行指定的联邦学习算法，并生成对比结果
# 使用方法: ./run_comparison.sh [数据集名称] [算法列表]
# 示例: ./run_comparison.sh mnist "fedavg,fedprox,moon"
# 示例: ./run_comparison.sh mnist all  # 运行所有算法

echo "========================================"
echo "  联邦学习算法比较实验"
echo "========================================"

# 检查命令行参数
if [ $# -lt 1 ]; then
    echo "错误: 未指定数据集名称"
    echo "使用方法: ./run_comparison.sh [数据集名称] [算法列表]"
    echo "支持的数据集: mnist, femnist, cifar10, cifar100"
    echo "支持的算法: fedavg, fedprox, moon, scaffold, feddistill, fedgen, fedspd, fedalone"
    echo "示例: ./run_comparison.sh mnist \"fedavg,fedprox,moon\""
    echo "示例: ./run_comparison.sh mnist all  # 运行所有算法"
    exit 1
fi

# 从命令行参数获取数据集名称
DATASET="$1"

# 获取要运行的算法列表
if [ $# -eq 2 ]; then
    if [ "$2" = "all" ]; then
        ALGORITHMS="fedavg,fedprox,moon,feddistill,fedgen,fedspd,fedalone"
    else
        ALGORITHMS="$2"
    fi
else
    # 默认运行所有算法
    ALGORITHMS="fedavg,fedprox,moon,feddistill,fedgen,fedspd,fedalone"
fi

# 验证数据集名称
if [[ "$DATASET" != "mnist" && "$DATASET" != "femnist" && "$DATASET" != "cifar10" && "$DATASET" != "cifar100" && "$DATASET" != "tinyimagenet" ]]; then
    echo "错误: 不支持的数据集 '$DATASET'"
    echo "支持的数据集: mnist, femnist, cifar10, cifar100, tinyimagenet"
    exit 1
fi

echo "选择的数据集: $DATASET"
echo "选择的算法: $ALGORITHMS"

# 设置基本参数
BATCH_SIZE=64    # 批处理大小
LOCAL_EPOCHS=10  # 本地训练轮数
COMM_ROUNDS=30  # 通信轮数
RATIO_CLIENT=0.8  # 每轮参与训练的客户端比例
LEARNING_RATE=0.01  # 学习率
OPTIMIZER="adam"    # 优化器: adam, sgd
SEED=42             # 随机种子，保证实验可重复性
PARTITION="dirichlet"   # 数据分区方式: iid, noiid, dirichlet
NUM_CLIENTS=20      # 客户端数量
DIR_BETA=0.2       # Dirichlet分布参数，仅在PARTITION="dirichlet"时使用

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
    echo "算法列表: $ALGORITHMS"
    echo "批处理大小: $BATCH_SIZE"
    echo "本地训练轮数: $LOCAL_EPOCHS"
    echo "通信轮数: $COMM_ROUNDS"
    echo "每轮参与训练的客户端比例: $RATIO_CLIENT"
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

# 运行指定的算法
IFS=',' read -ra ALGORITHM_LIST <<< "$ALGORITHMS"
for strategy in "${ALGORITHM_LIST[@]}"; do
    {
        echo ""
        echo "========================================"
        echo "  运行${strategy}算法"
        echo "========================================"
        echo ""
    } | tee -a "$LOG_FILE"

    python main.py \
        --dataset $DATASET \
        --strategy $strategy \
        --batch_size $BATCH_SIZE \
        --local_epochs $LOCAL_EPOCHS \
        --comm_rounds $COMM_ROUNDS \
        --ratio_client $RATIO_CLIENT \
        --lr $LEARNING_RATE \
        --optimizer $OPTIMIZER \
        --seed $SEED \
        --partition $PARTITION \
        --dir_beta $DIR_BETA \
        --num_clients $NUM_CLIENTS 2>&1 | tee -a "$LOG_FILE"
done

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
