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
LOCAL_EPOCHS=2  # 本地训练轮数
COMM_ROUNDS=2  # 通信轮数
LEARNING_RATE=0.01  # 学习率
OPTIMIZER="adam"    # 优化器: adam, sgd
SEED=42             # 随机种子，保证实验可重复性
PARTITION="dirichlet"   # 数据分区方式: iid, noiid, dirichlet
NUM_CLIENTS=10      # 客户端数量
DIR_BETA=0.4        # Dirichlet分布参数，仅在PARTITION="dirichlet"时使用

# FedProx和MOON的特定参数
FEDPROX_MU=0.01     # FedProx的正则化参数
MOON_MU=1.0         # MOON的对比损失权重
MOON_TEMP=0.5       # MOON的温度参数

# 数据集相关参数 - 根据不同数据集设置
if [ "$DATASET" = "mnist" ]; then
    # MNIST数据集参数
    NUM_CLASSES=10   # MNIST有10个数字类别(0-9)
    FEATURE_DIM=1600 # 64 * 5 * 5
    DATASET_DESC="MNIST: 10个类别(数字0-9)的28x28灰度图像数据集"
elif [ "$DATASET" = "femnist" ]; then
    # FEMNIST数据集参数
    NUM_CLASSES=62   # FEMNIST有62个类别(字母和数字)
    FEATURE_DIM=256  # 16 * 4 * 4
    DATASET_DESC="FEMNIST: 62个类别(数字、大小写字母)的28x28灰度手写图像数据集"
elif [ "$DATASET" = "cifar10" ]; then
    # CIFAR-10数据集参数
    NUM_CLASSES=10   # CIFAR-10有10个类别
    FEATURE_DIM=2048 # 128 * 4 * 4
    DATASET_DESC="CIFAR-10: 10个类别的32x32彩色图像数据集"
elif [ "$DATASET" = "cifar100" ]; then
    # CIFAR-100数据集参数
    NUM_CLASSES=100  # CIFAR-100有100个类别
    FEATURE_DIM=4096 # 3次下采样后，32x32 -> 4x4
    DATASET_DESC="CIFAR-100: 100个类别的32x32彩色图像数据集"
else
    # 默认参数
    NUM_CLASSES=10   # 默认假设10个类别
    FEATURE_DIM=784  # 默认假设MNIST大小
    DATASET_DESC="未知数据集，使用默认参数"
fi

# 输出当前数据集信息
echo "===== 数据集信息 ====="
echo "$DATASET_DESC"
echo "类别数: $NUM_CLASSES, 映射层输入特征维度: $FEATURE_DIM"
echo "======================"

# FedGen的特定参数 - 根据数据集设置不同参数
if [ "$DATASET" = "mnist" ]; then
    # MNIST数据集的FedGen参数（数字识别，简单灰度图像）
    FEDGEN_ENSEMBLE_ALPHA=1.0  # 教师损失权重
    FEDGEN_ENSEMBLE_BETA=0.5   # 学生损失权重
    FEDGEN_ENSEMBLE_ETA=0.1    # 多样性损失权重
    FEDGEN_LATENT_DIM=32       # 潜在空间维度
    FEDGEN_HIDDEN_DIM=128      # 隐藏层维度
    FEDGEN_TRAIN_EPOCHS=10     # 生成器训练轮数
    FEDGEN_COMMENT="# MNIST是简单的灰度数字图像，使用较小的生成器模型"
elif [ "$DATASET" = "femnist" ]; then
    # FEMNIST数据集的FedGen参数（手写字符识别，更复杂
    FEDGEN_LATENT_DIM=64       # 潜在空间维度（较大以捕捉更多变化）
    FEDGEN_HIDDEN_DIM=256      # 隐藏层维度（增加模型复杂度）
    FEDGEN_TRAIN_EPOCHS=5      # 生成器训练轮数
    FEDGEN_COMMENT="# FEMNIST包含手写字符，比MNIST更复杂，需要更大的生成器"
elif [ "$DATASET" = "cifar10" ]; then
    # CIFAR-10数据集的FedGen参数（彩色图像，10个类别）
    FEDGEN_LATENT_DIM=128      # 潜在空间维度（增大以处理彩色图像）
    FEDGEN_HIDDEN_DIM=512      # 隐藏层维度（增加以处理更复杂的特征）
    FEDGEN_TRAIN_EPOCHS=15     # 生成器训练轮数（增加以提高生成质量）
    FEDGEN_COMMENT="# CIFAR-10包含彩色图像，需要更大的模型和更多训练轮数来捕捉复杂特征"
elif [ "$DATASET" = "cifar100" ]; then
    # CIFAR-100数据集的FedGen参数（彩色图像，100个类别）
    FEDGEN_LATENT_DIM=256      # 潜在空间维度（增大以表示100个类别）
    FEDGEN_HIDDEN_DIM=1024     # 隐藏层维度（大幅增加以处理100个类别）
    FEDGEN_TRAIN_EPOCHS=20     # 生成器训练轮数（增加以提高复杂数据的生成质量）
    FEDGEN_COMMENT="# CIFAR-100有100个类别的彩色图像，需要非常大的生成器模型和更多训练轮数"
else
    
    echo "错误: 不支持的数据集 '$DATASET'"
    echo "支持的数据集: mnist, femnist, cifar10, cifar100"
    exit 1
fi

echo "$FEDGEN_COMMENT"

# 运行FedAvg算法
# echo ""
# echo "========================================"
# echo "  运行FedAvg算法"
# echo "========================================"
# echo ""

# python main.py \
#     --dataset $DATASET \
#     --strategy fedavg \
#     --batch_size $BATCH_SIZE \
#     --local_epochs $LOCAL_EPOCHS \
#     --comm_rounds $COMM_ROUNDS \
#     --lr $LEARNING_RATE \
#     --optimizer $OPTIMIZER \
#     --seed $SEED \
#     --partition $PARTITION \
#     --dir_beta $DIR_BETA \
#     --num_clients $NUM_CLIENTS


# # 运行FedProx算法
# echo ""
# echo "========================================"
# echo "  运行FedProx算法"
# echo "========================================"
# echo ""

# python main.py \
#     --dataset $DATASET \
#     --strategy fedprox \
#     --batch_size $BATCH_SIZE \
#     --local_epochs $LOCAL_EPOCHS \
#     --comm_rounds $COMM_ROUNDS \
#     --lr $LEARNING_RATE \
#     --optimizer $OPTIMIZER \
#     --mu $FEDPROX_MU \
#     --seed $SEED \
#     --partition $PARTITION \
#     --dir_beta $DIR_BETA \
#     --num_clients $NUM_CLIENTS


# # 运行MOON算法
# echo ""
# echo "========================================"
# echo "  运行MOON算法"
# echo "========================================"
# echo ""

# python main.py \
#     --dataset $DATASET \
#     --strategy moon \
#     --batch_size $BATCH_SIZE \
#     --local_epochs $LOCAL_EPOCHS \
#     --comm_rounds $COMM_ROUNDS \
#     --lr $LEARNING_RATE \
#     --optimizer $OPTIMIZER \
#     --mu $MOON_MU \
#     --temperature $MOON_TEMP \
#     --seed $SEED \
#     --partition $PARTITION \
#     --dir_beta $DIR_BETA \
#     --num_clients $NUM_CLIENTS


# # 运行Scaffold算法
# echo ""
# echo "========================================"
# echo "  运行Scaffold算法"
# echo "========================================"
# echo ""

# python main.py \
#     --dataset $DATASET \
#     --strategy scaffold \
#     --batch_size $BATCH_SIZE \
#     --local_epochs $LOCAL_EPOCHS \
#     --comm_rounds $COMM_ROUNDS \
#     --lr $LEARNING_RATE \
#     --optimizer $OPTIMIZER \
#     --seed $SEED \
#     --partition $PARTITION \
#     --dir_beta $DIR_BETA \
#     --num_clients $NUM_CLIENTS \
#     --num_classes $NUM_CLASSES 

# 运行FedGen算法
echo ""
echo "========================================"
echo "  运行FedGen算法"
echo "========================================"
echo ""

# 输出当前使用的FedGen参数设置
echo "使用数据集: $DATASET"
echo "生成器网络: latent_dim=$FEDGEN_LATENT_DIM, hidden_dim=$FEDGEN_HIDDEN_DIM, train_epochs=$FEDGEN_TRAIN_EPOCHS"
echo "数据集信息: 类别数=$NUM_CLASSES, 映射层输入特征维度=$FEATURE_DIM"
echo "$FEDGEN_COMMENT"

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
    --num_clients $NUM_CLIENTS \
    --latent_dim $FEDGEN_LATENT_DIM \
    --hidden_dim $FEDGEN_HIDDEN_DIM \
    --num_classes $NUM_CLASSES \
    --feature_dim $FEATURE_DIM

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
