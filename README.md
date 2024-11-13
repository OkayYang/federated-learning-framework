

# 联邦学习框架

这是一个可扩展的 **联邦学习（Federated Learning, FL）** 框架，支持多种算法、数据集，并具有未来的扩展能力。该项目参考了蚂蚁集团开源框架 [SecretFlow]( https://github.com/secretflow/secretflow)，提供了模块化设计，便于自定义和测试各种联邦学习算法。

## 特性

- **自定义数据集**：支持 `FEMNIST` 和 `MNIST` 数据集，适用于联邦学习实验。
- **联邦学习算法**：实现了 `FedAvg` 算法，未来将扩展支持 `FedProx`、`FedBN`、`SCAFFOLD`、`MOON` 等经典算法、个性化联邦学习算法等。
- **隐私保护功能**：计划加入同态加密和差分隐私等技术，以保证联邦学习过程中的数据隐私。

## 已实现功能

### 数据集
- [x] **FEMNIST 数据集**：FEMNIST（Federated Extended MNIST）是一个用于联邦学习研究的手写字母分类数据集，基于EMNIST数据集扩展而来。与传统的MNIST数据集不同，FEMNIST模拟了分布式环境中的非独立同分布（Non-IID）问题，将数据划分给多个用户，每个用户拥有自己的训练数据。该数据集包含3,550个用户，总共有80,526个样本，平均每个用户有226.83个样本。由于样本数存在差异，标准差为88.94，且标准差与平均数的比值为0.39，表明数据的分布具有一定的波动性。FEMNIST旨在为联邦学习提供更加贴近现实的数据分布，适用于测试和研究分布式训练中的挑战，例如数据隐私保护和异构数据处理。**官方已经给出了按用户划分的[代码用例](https://github.com/TalwalkarLab/leaf)，但由于用户量太大不便于实现，我仅仅从中挑选了几个用户作为测试，因此若使用该数据集不需要在自定义客户端。本人也推荐采用这个数据集更能模拟真实的联邦场景**
- [x] **MNIST 数据集**：经典的手写数字分类数据集，代码提供了IID和NoIID两种划分方式这个大家比较熟悉就不再过多介绍。

### 联邦学习算法
- [x] **FedAvg**：实现了经典的联邦平均算法（Federated Averaging，FedAvg），这是联邦学习中的基础算法。

### 未来计划
- [ ] **FedProx**：添加 FedProx 算法，优化联邦学习中的非独立同分布（Non-IID）问题。
- [ ] **FedBN**：实现 FedBN 算法，用于解决分布式训练中的批量归一化问题。
- [ ] **SCAFFOLD**：支持 SCAFFOLD 算法，改善联邦学习中的客户端偏差。
- [ ] **MOON**：实现 MOON 算法，探索联邦学习中的个性化模型。
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

### 1. 加载数据集
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

### 2. 配置模型和客户端本地训练参数
定义联邦学习模型（例如，`FeMNISTNet`）和训练参数：

```python
from fl.model.model import FeMNISTNet
from fl.fl_base import ModelConfig
from fl.utils import optim_wrapper
import torch.optim as optim

loss_fn = torch.nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-2)

model_config = ModelConfig(
    model_fn=FeMNISTNet,  # 使用的模型
    loss_fn=loss_fn,      # 损失函数
    optim_fn=optim_fn,    # 优化器
    epochs=10,            # 训练轮数
    batch_size=32         # 每批次大小
)
```

### 3. 启动联邦学习服务器
初始化并启动联邦学习服务器，使用 `FedAvg` 算法进行训练：

```python
from fl.fl_server import FLServer

fl_server = FLServer(
    client_list=client_list,
    strategy="fedavg",
    model_config=model_config,
    client_dataset_dict=dataset_dict
)

history = fl_server.fit(comm_rounds=100, ratio_client=1)  # 100个通信轮次,每次选取客户端比例
```

### 4. 可视化结果
训练完成后，使用图表展示全局和每个客户端的训练指标：

```python
from fl.utils import plot_global_metrics, plot_worker_metrics

plot_global_metrics(history)  # 全局性能指标
plot_worker_metrics(history)  # 每个客户端的性能指标
```

## 贡献

欢迎提交问题（issues）和拉取请求（pull requests）以改进本项目。如果你有任何建议或功能需求，也可以在问题区留言。

## 许可证

本项目采用 [Apache-2.0](LICENSE.txt)。

