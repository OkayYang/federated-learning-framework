# 联邦学习框架

<p align="center">
<a href="./README.zh-CN.md">简体中文</a> | <a href="./README.md">English</a>
</p>

这是一个可扩展的 **联邦学习（Federated Learning, FL）** 框架，支持多种算法、数据集，并具有未来的扩展能力。该项目参考了蚂蚁集团开源框架 [SecretFlow]( https://github.com/secretflow/secretflow)，提供了模块化设计，便于自定义和测试各种联邦学习算法。

## 特性

- **自定义数据集**：支持 `FEMNIST`、`MNIST`、`CIFAR10` 和 `CIFAR100` 数据集，适用于联邦学习实验。
- **联邦学习算法**：实现了 `FedAvg`、`FedProx`、`MOON`、`SCAFFOLD`、`FedDistill`、`FedGen`、`FedSPD` 算法。
- **隐私保护功能**：计划加入同态加密和差分隐私等技术，以保证联邦学习过程中的数据隐私。

## 已实现功能

### 数据集
- [x] **FEMNIST 数据集**：FEMNIST（Federated Extended MNIST）是一个用于联邦学习研究的手写字母分类数据集，基于EMNIST数据集扩展而来。与传统的MNIST数据集不同，FEMNIST模拟了分布式环境中的非独立同分布（Non-IID）问题，将数据划分给多个用户，每个用户拥有自己的训练数据。该数据集包含3,550个用户，总共有80,526个样本，平均每个用户有226.83个样本。由于样本数存在差异，标准差为88.94，且标准差与平均数的比值为0.39，表明数据的分布具有一定的波动性。FEMNIST旨在为联邦学习提供更加贴近现实的数据分布，适用于测试和研究分布式训练中的挑战，例如数据隐私保护和异构数据处理。**官方已经给出了按用户划分的[代码用例](https://github.com/TalwalkarLab/leaf)，但由于用户量太大不便于实现，我仅仅从中挑选了几个用户作为测试，因此若使用该数据集不需要在自定义客户端。**
- [x] **MNIST 数据集**：经典的手写数字分类数据集，代码提供了IID和NoIID两种划分方式这个大家比较熟悉就不再过多介绍。
- [x] **CIFAR10 数据集**：常用的彩色图像分类数据集，包含10个类别的32x32彩色图像。代码提供了IID、Non-IID和基于狄利克雷分布的划分方式。
- [x] **CIFAR100 数据集**：CIFAR10的扩展版本，包含100个类别的32x32彩色图像。适合测试联邦学习在大量类别情况下的性能。同样支持IID、Non-IID和基于狄利克雷分布的划分方式。

### 联邦学习算法

#### FedAvg
联邦平均（Federated Averaging）是由Google提出的基础联邦学习算法。它在本地设备上执行多轮梯度下降，然后将参数发送到服务器进行平均，从而构建全局模型。

#### FedProx
FedProx（Federated Proximal）通过添加近端项来改进FedAvg，限制本地更新与全局模型之间的差异，从而增强系统的鲁棒性，尤其是在异构数据环境中。

#### MOON
MOON（Model-Contrastive Federated Learning）使用对比学习来解决客户端漂移问题，通过让本地模型不仅学习分类任务，还要最小化与全局模型的表示差异。

#### SCAFFOLD
SCAFFOLD（Stochastic Controlled Averaging for Federated Learning）使用方差减少技术，通过控制变量来纠正客户端更新的漂移，加速收敛并提高性能。SCAFFOLD通过修改梯度来纠正客户端漂移：$g_i ← g_i - c_i + c$，其中$g_i$是原始梯度，$c_i$是客户端控制变量，$c$是全局控制变量。实现中使用了梯度裁剪以提高稳定性。

#### FedDistill
联邦蒸馏（Federated Distillation）通过传递softmax输出（logits）而非模型参数来实现知识共享，减少通信开销，并使客户端能够使用个性化模型。

#### FedGen
联邦生成学习（Federated Learning with Generative Models）是一种创新的联邦学习方法，通过生成模型来解决数据异构问题。FedGen使用一个中央生成器来合成特征，帮助客户端学习缺失的类别数据，特别适合解决Non-IID数据分布问题。

### 未来计划

- [ ] **同态加密联邦学习**：加入同态加密保护数据隐私，使得在训练过程中不会暴露用户数据。
- [ ] **差分隐私联邦学习**：实现差分隐私机制，保护用户数据隐私。

## 环境要求

- Python 3.9
- 其他依赖库：`torch` `torchvision`, `numpy`, `matplotlib` 等（具体依赖请参考 `requirements.txt`）

## 安装

1. 克隆项目：
   ```bash
   git clone https://github.com/OkayYang/federated-learning-framework.git
   cd federated-learning-framework
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 命令行参数运行
框架提供了命令行参数接口，可以通过以下命令运行：

```bash
# 基本用法 - 使用默认参数
python main.py

# 使用参数
python main.py --strategy fedavg --dataset cifar10 --partition dirichlet --num_clients 10 --dir_beta 0.3 --batch_size 64 --local_epochs 20 --comm_rounds 30 --ratio_client 0.8 --lr 0.01 --optimizer adam --seed 42 --plot_distribution True
```

#### 可用参数：

- **数据集参数**：
  - `--dataset`: 指定数据集 (femnist, mnist, cifar10, cifar100)
  - `--partition`: 数据分区方式 (iid, noiid, dirichlet)，仅用于MNIST/CIFAR系列数据集
  - `--dir_beta`: 当使用dirichlet划分方式时的狄利克雷分布的参数
  - `--num_clients`: MNIST/CIFAR系列数据集的客户端数量

- **训练参数**：
  - `--batch_size`: 训练批次大小
  - `--local_epochs`: 本地训练轮数
  - `--comm_rounds`: 通信轮数
  - `--ratio_client`: 每轮参与训练的客户端比例
  - `--lr`: 学习率
  - `--optimizer`: 优化器类型 (adam 或 sgd)

- **算法参数**：
  - `--strategy`: 联邦学习策略 (fedavg, fedprox, moon, scaffold, feddistill, fedgen, fedspd, fedalone)

- **其他参数**：
  - `--seed`: 随机种子，确保实验可重复性
  - `--plot_distribution`: 是否绘制客户端标签分布图

### 代码调用方式

#### 1. 加载数据集
加载 `FEMNIST` 或 `MNIST` 数据集并将其划分到各个客户端：

```python
from fl.data import datasets
# feminist_dataset数据集内置了客户端，不需要自定义客户端
# Load the FEMNIST dataset
client_list, dataset_dict = datasets.load_feminist_dataset()

# Load the MNIST dataset for clients
# client_list = ["alice","bob","rick"]
# dataset_dict = datasets.load_mnist_dataset(client_list)
```

#### 2. 配置模型和客户端本地训练参数
定义联邦学习模型（例如，`FeMNISTNet`）和训练参数：

```python
from fl.model.model import FeMNISTNet
from fl.client.fl_base import ModelConfig
from fl.utils import optim_wrapper
import torch.optim as optim

loss_fn = torch.nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-2)

model_config = ModelConfig(
    model_fn=FeMNISTNet,  # 使用的模型
    loss_fn=loss_fn,  # 损失函数
    optim_fn=optim_fn,  # 优化器
    epochs=10,  # 训练轮数
    batch_size=32  # 每批次大小
)
```

#### 3. 启动联邦学习服务器
初始化并启动联邦学习服务器，使用所选算法进行训练：

```python
from fl.server.fl_server import FLServer

# 使用FedAvg算法
fl_server = FLServer(
    client_list=client_list,
    strategy="fedavg",
    model_config=model_config,
    client_dataset_dict=dataset_dict
)

history = fl_server.fit(comm_rounds=100, ratio_client=1)  # 100个通信轮次,每次选取客户端比例
```

#### 4. 可视化结果
训练完成后，使用图表展示全局和每个客户端的训练指标：

```python
from fl.utils import plot_global_metrics, plot_worker_metrics

plot_global_metrics(history)  # 全局性能指标
plot_worker_metrics(history)  # 每个客户端的性能指标
```

### 可视化与对比工具

框架提供了多种可视化工具，用于分析和比较不同联邦学习算法的性能表现。这些工具可以帮助研究人员直观地理解各种策略的优缺点。

#### 策略对比可视化

`compare_strategies.py` 脚本提供了一个全面的可视化工具，用于比较不同联邦学习算法的性能：

```bash
# 运行策略对比脚本
python compare_strategies.py
```

这个脚本会自动:
1. 从 `./plots/history/` 目录加载所有保存的历史记录文件
2. 生成三个子图，分别展示:
   - 训练损失 (Training Loss)
   - 测试损失 (Test Loss)
   - 测试准确率 (Test Accuracy)

### 算法对比实验

框架提供了一个便捷的对比脚本 `run_comparison.sh`，可以自动运行所有支持的算法并生成对比结果：

```bash
# 运行所有算法的对比实验
bash run_comparison.sh

# 生成的对比图表位于 ./plots/comparison/ 目录
```

对比实验会展示不同算法在训练损失、测试损失和测试准确率上的表现差异，有助于选择最适合特定场景的联邦学习算法。

## 贡献

欢迎提交问题（issues）和拉取请求（pull requests）以改进本项目。如果你有任何建议或功能需求，也可以在问题区留言。

## 许可证

本项目采用 [Apache-2.0](LICENSE.txt)。 