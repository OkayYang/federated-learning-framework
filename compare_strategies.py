import os
import glob
import pickle
import matplotlib.pyplot as plt

history_dir="./plots/history/"
# 确保目录存在
if not os.path.exists(history_dir):
    print(f"目录 {history_dir} 不存在，请先训练模型并保存历史记录")
    exit()
    

# 保存对比图表的目录
compare_dir = "./plots/comparison/"
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
    if filename.startswith("history_"):
        # 如果文件名格式为 history_{strategy}.pkl
        strategy_name = filename[8:].split('.')[0]  # 去掉"history_"前缀和".pkl"后缀
    else:
        # 如果文件名格式为 {strategy}.pkl
        strategy_name = filename.split('.')[0]
    
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
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# 1. 训练损失子图 - 收集所有线条以便放到外部图例
lines = []
labels = []
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["global"]["train_loss"]) + 1)
    line, = axes[0].plot(epochs, history["global"]["train_loss"], marker="o", markersize=4)
    # 只在第一个子图收集线条和标签
    lines.append(line)
    labels.append(strategy)

axes[0].set_xlabel("Communication Rounds", fontsize=12)
axes[0].set_ylabel("Training Loss", fontsize=12)
axes[0].set_title("Global Training Loss", fontsize=14, pad=10)  # 增加标题和图的间距
# 不在这里显示图例
axes[0].grid(True)

# 2. 测试损失子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["global"]["test_loss"]) + 1)
    axes[1].plot(epochs, history["global"]["test_loss"], marker="s", markersize=4)

axes[1].set_xlabel("Communication Rounds", fontsize=12)
axes[1].set_ylabel("Test Loss", fontsize=12)
axes[1].set_title("Global Test Loss", fontsize=14, pad=10)  # 增加标题和图的间距
# 不在这里显示图例
axes[1].grid(True)

# 3. 测试准确率子图
for strategy, history in strategies_data.items():
    epochs = range(1, len(history["global"]["test_accuracy"]) + 1)
    axes[2].plot(epochs, history["global"]["test_accuracy"], marker="^", markersize=4)

axes[2].set_xlabel("Communication Rounds", fontsize=12)
axes[2].set_ylabel("Test Accuracy", fontsize=12)
axes[2].set_title("Global Test Accuracy", fontsize=14, pad=10)  # 增加标题和图的间距
# 不在这里显示图例
axes[2].grid(True)

# 添加整个图表的图例 - 放在右侧
fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=12, 
           title="Strategies", title_fontsize=14)

# 调整布局，为右侧图例和顶部标题预留空间
plt.tight_layout(rect=[0, 0, 0.85, 0.92])  # 上方也留出空间

# 添加整体标题，并提高它的位置以避免重叠
plt.suptitle("Comparison of Different Federated Learning Strategies", 
             fontsize=16, y=0.98)  # y参数调高标题位置

plt.savefig(f"{compare_dir}/strategies_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"对比图表已保存到 {compare_dir} 目录")
