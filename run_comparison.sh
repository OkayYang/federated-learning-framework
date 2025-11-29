#!/bin/sh
export RAY_BACKEND_LOG_LEVEL=error
export RAY_worker_register_timeout_seconds=60  # 可选，减少某些超时日志
# 联邦学习算法比较实验脚本
# 这个脚本会运行指定的联邦学习算法，并生成对比结果
# 使用方法: ./run_comparison.sh [数据集名称] [算法列表]
# 示例: ./run_comparison.sh mnist "fedavg,fedprox,moon"
# 示例: ./run_comparison.sh mnist all  # 运行所有算法

# 创建日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "  联邦学习算法比较实验"
echo "========================================"

# 检查命令行参数
if [ $# -lt 1 ]; then
    echo "错误: 未指定数据集名称"
    echo "使用方法: ./run_comparison.sh [数据集名称] [算法列表]"
    echo "支持的数据集: mnist, femnist, cifar10, cifar100"
    echo "支持的算法: fedavg, fedprox, moon, scaffold, feddistill, fedgen, fedspd, fedalone, fedftg, fedgkd"
    echo "示例: ./run_comparison.sh mnist \"fedavg,fedprox,moon\""
    echo "示例: ./run_comparison.sh mnist all  # 运行所有算法"
    exit 1
fi

# 从命令行参数获取数据集名称
DATASET="$1"

# 获取要运行的算法列表
if [ $# -eq 2 ]; then
    if [ "$2" = "all" ]; then
        ALGORITHMS="fedavg,fedprox,fedspd,fedspd-lc,moon,feddistill,fedgen,scaffold,fedalone,fedftg,fedgkd"
    else
        ALGORITHMS="$2"
    fi
else
    # 默认运行所有算法
    ALGORITHMS="fedavg,fedprox,fedspd,fedspd-lc,moon,feddistill,fedgen,fedalone,fedftg,fedgkd"
fi

# 验证数据集名称（使用 POSIX sh 语法）
case "$DATASET" in
    mnist|femnist|cifar10|cifar100|tinyimagenet|svhn|emnist)
        # 支持的数据集
        ;;
    *)
        echo "错误: 不支持的数据集 '$DATASET'"
        echo "支持的数据集: mnist, femnist, cifar10, cifar100, tinyimagenet, svhn, emnist"
        exit 1
        ;;
esac

echo "选择的数据集: $DATASET"
echo "选择的算法: $ALGORITHMS"

# 设置基本参数
BATCH_SIZE=64    # 批处理大小
LOCAL_EPOCHS=10  # 本地训练轮数
COMM_ROUNDS=50  # 通信轮数
RATIO_CLIENT=0.5  # 每轮参与训练的客户端比例
LEARNING_RATE=0.01  # 学习率
OPTIMIZER="adam"    # 优化器: adam, sgd
SEED=2026             # 随机种子，保证实验可重复性
PARTITION="iid"   # 数据分区方式: iid, noiid, dirichlet
NUM_CLIENTS=20      # 客户端数量
DIR_BETA=0.2       # Dirichlet分布参数，仅在PARTITION="dirichlet"时使用
DATA_FRACTION=1.0    # 数据集采样比例


# 创建数据集特定的日志目录
DATASET_LOG_DIR="./logs/${DATASET}"
mkdir -p "$DATASET_LOG_DIR"

# 记录实验参数
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
echo "数据集采样比例: $DATA_FRACTION"
echo "========================================"
echo ""

# 检查CUDA
python check_cuda.py

# 运行指定的算法（兼容 POSIX sh：不使用 bash 数组和 here-string）
OLD_IFS="$IFS"
IFS=','
set -- $ALGORITHMS
IFS="$OLD_IFS"

for strategy in "$@"; do
    echo ""
    echo "========================================"
    echo "  运行${strategy}算法"
    echo "========================================"
    echo ""

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
        --data_fraction $DATA_FRACTION \
        --num_clients $NUM_CLIENTS
done

# 生成对比结果图表
echo ""
echo "========================================"
echo "  生成算法对比结果"
echo "========================================"
echo ""

# 运行对比脚本，传入数据集名称
python compare_strategies.py $DATASET

echo ""
echo "========================================"
echo "  算法对比实验完成!"
echo "========================================"
echo "对比图表已保存到 ./plots/$DATASET 目录"
echo "========================================"
echo ""
