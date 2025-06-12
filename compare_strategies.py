import os
import glob
import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.signal import savgol_filter

# 检查命令行参数，获取数据集名称
if len(sys.argv) > 1:
    dataset = sys.argv[1].lower()
else:
    print("请指定数据集名称作为参数")
    print("用法: python compare_strategies.py [数据集名称]")
    sys.exit(1)

# 数据集特定的历史记录目录
history_dir = f"./plots/{dataset}/history/"
# 确保目录存在
if not os.path.exists(history_dir):
    print(f"目录 {history_dir} 不存在，请先训练模型并保存历史记录")
    exit()
    
# 保存对比图表的目录
compare_dir = f"./plots/{dataset}/comparison/"
os.makedirs(compare_dir, exist_ok=True)

# 搜索所有历史记录文件
history_files = glob.glob(os.path.join(history_dir, "*.pkl"))
if not history_files:
    print(f"在 {history_dir} 目录中未找到历史记录文件")
    exit()
    
# 加载所有历史记录文件
strategies_data = {}
for file_path in history_files:
    # 从文件名中提取策略名称
    filename = os.path.basename(file_path)
    strategy_name = filename.split('_')[0]  # 提取策略名称
    
    try:
        with open(file_path, "rb") as f:
            history = pickle.load(f)
            strategies_data[strategy_name] = history
            print(f"已加载策略 {strategy_name} 的历史数据")
    except Exception as e:
        print(f"加载 {file_path} 时出错: {e}")

if not strategies_data:
    print("没有成功加载任何历史数据")
    exit()

# 图表样式设置
plt.style.use('default')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'figure.dpi': 300
})

# 定义颜色和标记
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

# 平滑函数 - 使用Savitzky-Golay滤波器
def smooth_data(data, window_length=5, polyorder=2):
    # 确保数据点足够，否则不平滑
    if len(data) < window_length:
        return data
    
    # 使数据长度足够进行滤波
    if len(data) % 2 == 0 and window_length % 2 == 1:
        window_length = min(window_length, len(data) - 1)
    
    # 确保window_length是奇数且小于数据长度
    window_length = min(window_length, len(data) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    
    if window_length < 3:
        return data
    
    try:
        return savgol_filter(data, window_length, polyorder)
    except Exception:
        # 如果出错，返回原始数据
        return data

# -------------- 绘制Global Test Accuracy图 (原始数据) --------------
plt.figure(figsize=(12, 8))

for i, (strategy, history) in enumerate(strategies_data.items()):
    epochs = range(1, len(history["global"]["test_accuracy"]) + 1)
    
    # 使用原始数据
    accuracy = history["global"]["test_accuracy"]
    
    plt.plot(
        epochs, 
        accuracy, 
        color=colors[i % len(colors)],
        marker=markers[i % len(markers)],
        markersize=8,
        markevery=max(1, len(epochs)//8),
        linewidth=2.5,
        label=strategy
    )

plt.xlabel("Communication Rounds", fontsize=16)
plt.ylabel("Test Accuracy", fontsize=16)
plt.title(f"Global Test Accuracy (Raw Data) - {dataset.upper()} Dataset", fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', frameon=True)

# 自适应调整y轴范围
global_min = min([min(history["global"]["test_accuracy"]) for history in strategies_data.values()])
global_max = max([max(history["global"]["test_accuracy"]) for history in strategies_data.values()])
plt.ylim(global_min - 0.02, global_max + 0.02)  # 添加少许边距

plt.tight_layout()
plt.savefig(f"{compare_dir}/global_test_accuracy_raw.png", dpi=300, bbox_inches="tight")
plt.close()

# -------------- 绘制Global Test Accuracy图 (平滑数据) --------------
plt.figure(figsize=(12, 8))

for i, (strategy, history) in enumerate(strategies_data.items()):
    epochs = range(1, len(history["global"]["test_accuracy"]) + 1)
    
    # 平滑数据
    accuracy = smooth_data(history["global"]["test_accuracy"])
    
    plt.plot(
        epochs, 
        accuracy, 
        color=colors[i % len(colors)],
        marker=markers[i % len(markers)],
        markersize=8,
        markevery=max(1, len(epochs)//8),
        linewidth=2.5,
        label=f"{strategy}"
    )

plt.xlabel("Communication Rounds", fontsize=16)
plt.ylabel("Test Accuracy", fontsize=16)
plt.title(f"Global Test Accuracy (Smoothed) - {dataset.upper()} Dataset", fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', frameon=True)

# 自适应调整y轴范围
global_min = min([min(history["global"]["test_accuracy"]) for history in strategies_data.values()])
global_max = max([max(history["global"]["test_accuracy"]) for history in strategies_data.values()])
plt.ylim(global_min - 0.02, global_max + 0.02)  # 添加少许边距

plt.tight_layout()
plt.savefig(f"{compare_dir}/global_test_accuracy_smoothed.png", dpi=300, bbox_inches="tight")
plt.close()

# -------------- 绘制Local Test Accuracy图 (原始数据) --------------
plt.figure(figsize=(12, 8))

for i, (strategy, history) in enumerate(strategies_data.items()):
    epochs = range(1, len(history["local"]["test_accuracy"]) + 1)
    
    # 使用原始数据
    accuracy = history["local"]["test_accuracy"]
    
    plt.plot(
        epochs, 
        accuracy, 
        color=colors[i % len(colors)],
        marker=markers[i % len(markers)],
        markersize=8,
        markevery=max(1, len(epochs)//8),
        linewidth=2.5,
        label=strategy
    )

plt.xlabel("Communication Rounds", fontsize=16)
plt.ylabel("Test Accuracy", fontsize=16)
plt.title(f"Local Test Accuracy (Raw Data) - {dataset.upper()} Dataset", fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', frameon=True)

# 自适应调整y轴范围
local_min = min([min(history["local"]["test_accuracy"]) for history in strategies_data.values()])
local_max = max([max(history["local"]["test_accuracy"]) for history in strategies_data.values()])
plt.ylim(local_min - 0.02, local_max + 0.02)  # 添加少许边距

plt.tight_layout()
plt.savefig(f"{compare_dir}/local_test_accuracy_raw.png", dpi=300, bbox_inches="tight")
plt.close()

# -------------- 绘制Local Test Accuracy图 (平滑数据) --------------
plt.figure(figsize=(12, 8))

for i, (strategy, history) in enumerate(strategies_data.items()):
    epochs = range(1, len(history["local"]["test_accuracy"]) + 1)
    
    # 平滑数据
    accuracy = smooth_data(history["local"]["test_accuracy"])
    
    plt.plot(
        epochs, 
        accuracy, 
        color=colors[i % len(colors)],
        marker=markers[i % len(markers)],
        markersize=8,
        markevery=max(1, len(epochs)//8),
        linewidth=2.5,
        label=f"{strategy} (smoothed)"
    )

plt.xlabel("Communication Rounds", fontsize=16)
plt.ylabel("Test Accuracy", fontsize=16)
plt.title(f"Local Test Accuracy (Smoothed) - {dataset.upper()} Dataset", fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', frameon=True)

# 自适应调整y轴范围
local_min = min([min(history["local"]["test_accuracy"]) for history in strategies_data.values()])
local_max = max([max(history["local"]["test_accuracy"]) for history in strategies_data.values()])
plt.ylim(local_min - 0.02, local_max + 0.02)  # 添加少许边距

plt.tight_layout()
plt.savefig(f"{compare_dir}/local_test_accuracy_smoothed.png", dpi=300, bbox_inches="tight")

print(f"准确率对比图已保存到 {compare_dir} 目录")
