import os
import glob
import pickle
import matplotlib.pyplot as plt
import sys

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

# 为图例预留更多空间，增加高度为标题预留空间
fig, axes = plt.subplots(2, 3, figsize=(24, 14))

# 收集所有线条以便放到外部图例
lines = []
labels = []

# 第一行：全局指标
# 1. 全局训练损失子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["global"]["train_loss"]) + 1)
    line, = axes[0, 0].plot(epochs, history["global"]["train_loss"], marker="o", markersize=4)
    # 只在第一个子图收集线条和标签
    if strategy not in labels:
        lines.append(line)
        labels.append(strategy)

axes[0, 0].set_xlabel("Communication Rounds", fontsize=12)
axes[0, 0].set_ylabel("Training Loss", fontsize=12)
axes[0, 0].set_title("Global Training Loss", fontsize=14, pad=10)
axes[0, 0].grid(True)

# 2. 全局测试损失子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["global"]["test_loss"]) + 1)
    axes[0, 1].plot(epochs, history["global"]["test_loss"], marker="s", markersize=4)

axes[0, 1].set_xlabel("Communication Rounds", fontsize=12)
axes[0, 1].set_ylabel("Test Loss", fontsize=12)
axes[0, 1].set_title("Global Test Loss", fontsize=14, pad=10)
axes[0, 1].grid(True)

# 3. 全局测试准确率子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["global"]["test_accuracy"]) + 1)
    axes[0, 2].plot(epochs, history["global"]["test_accuracy"], marker="^", markersize=4)

axes[0, 2].set_xlabel("Communication Rounds", fontsize=12)
axes[0, 2].set_ylabel("Test Accuracy", fontsize=12)
axes[0, 2].set_title("Global Test Accuracy", fontsize=14, pad=10)
axes[0, 2].grid(True)

# 第二行：本地指标
# 4. 本地训练损失子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["local"]["train_loss"]) + 1)
    axes[1, 0].plot(epochs, history["local"]["train_loss"], marker="o", markersize=4)

axes[1, 0].set_xlabel("Communication Rounds", fontsize=12)
axes[1, 0].set_ylabel("Training Loss", fontsize=12)
axes[1, 0].set_title("Local Training Loss", fontsize=14, pad=10)
axes[1, 0].grid(True)

# 5. 本地测试损失子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["local"]["test_loss"]) + 1)
    axes[1, 1].plot(epochs, history["local"]["test_loss"], marker="s", markersize=4)

axes[1, 1].set_xlabel("Communication Rounds", fontsize=12)
axes[1, 1].set_ylabel("Test Loss", fontsize=12)
axes[1, 1].set_title("Local Test Loss", fontsize=14, pad=10)
axes[1, 1].grid(True)

# 6. 本地测试准确率子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["local"]["test_accuracy"]) + 1)
    axes[1, 2].plot(epochs, history["local"]["test_accuracy"], marker="^", markersize=4)

axes[1, 2].set_xlabel("Communication Rounds", fontsize=12)
axes[1, 2].set_ylabel("Test Accuracy", fontsize=12)
axes[1, 2].set_title("Local Test Accuracy", fontsize=14, pad=10)
axes[1, 2].grid(True)

# 添加整个图表的图例 - 放在右侧
fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=12, 
           title="Strategies", title_fontsize=14)

# 调整布局，为右侧图例和顶部标题预留空间
plt.tight_layout(rect=[0, 0, 0.85, 0.95])

# 添加整体标题，并提高它的位置以避免重叠
plt.suptitle(f"Comparison of Different Federated Learning Strategies on {dataset.upper()} Dataset", 
             fontsize=16, y=0.98)

# 保存全部指标的对比图
plt.savefig(f"{compare_dir}/strategies_full_comparison.png", dpi=300, bbox_inches="tight")

# 再创建一个只有全局指标的图
plt.figure(figsize=(24, 7))
fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7))

# 为第二个图表收集线条
lines2 = []
labels2 = []

# 1. 全局训练损失子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["global"]["train_loss"]) + 1)
    line, = axes2[0].plot(epochs, history["global"]["train_loss"], marker="o", markersize=4)
    if strategy not in labels2:
        lines2.append(line)
        labels2.append(strategy)

axes2[0].set_xlabel("Communication Rounds", fontsize=12)
axes2[0].set_ylabel("Training Loss", fontsize=12)
axes2[0].set_title("Global Training Loss", fontsize=14, pad=10)
axes2[0].grid(True)

# 2. 全局测试损失子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["global"]["test_loss"]) + 1)
    axes2[1].plot(epochs, history["global"]["test_loss"], marker="s", markersize=4)

axes2[1].set_xlabel("Communication Rounds", fontsize=12)
axes2[1].set_ylabel("Test Loss", fontsize=12)
axes2[1].set_title("Global Test Loss", fontsize=14, pad=10)
axes2[1].grid(True)

# 3. 全局测试准确率子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["global"]["test_accuracy"]) + 1)
    axes2[2].plot(epochs, history["global"]["test_accuracy"], marker="^", markersize=4)

axes2[2].set_xlabel("Communication Rounds", fontsize=12)
axes2[2].set_ylabel("Test Accuracy", fontsize=12)
axes2[2].set_title("Global Test Accuracy", fontsize=14, pad=10)
axes2[2].grid(True)

# 添加整个图表的图例 - 放在右侧
fig2.legend(lines2, labels2, loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=12, 
            title="Strategies", title_fontsize=14)

# 调整布局，为右侧图例和顶部标题预留空间
plt.tight_layout(rect=[0, 0, 0.85, 0.92])

# 添加整体标题，并提高它的位置以避免重叠
plt.suptitle(f"Global Metrics Comparison of Federated Learning Strategies on {dataset.upper()} Dataset", 
             fontsize=16, y=0.98)

# 保存只有全局指标的对比图
plt.savefig(f"{compare_dir}/strategies_global_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

# 显示第一个包含所有指标的图
plt.figure(fig.number)
plt.show()

print(f"对比图表已保存到 {compare_dir} 目录")
