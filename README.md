# Federated Learning Framework

<p align="center">
<a href="./README.zh-CN.md">简体中文</a> | <a href="./README.md">English</a>
</p>

This is an extensible **Federated Learning (FL)** framework that supports multiple algorithms and datasets with future expansion capabilities. The project references Ant Group's open-source framework [SecretFlow](https://github.com/secretflow/secretflow), providing a modular design for easy customization and testing of various federated learning algorithms.

## Features

- **Custom Datasets**: Supports `FEMNIST`, `MNIST`, `CIFAR10`, and `CIFAR100` datasets, suitable for federated learning experiments.
- **Federated Learning Algorithms**: Implements `FedAvg`, `FedProx`, `MOON`, `SCAFFOLD`, `FedDistill`, `FedGen`, and `FedSPD` algorithms.
- **Privacy Protection Features**: Plans to incorporate homomorphic encryption and differential privacy technologies to ensure data privacy during federated learning.

## Implemented Features

### Datasets
- [x] **FEMNIST Dataset**: FEMNIST (Federated Extended MNIST) is a handwritten letter classification dataset for federated learning research, extended from the EMNIST dataset. Unlike traditional MNIST, FEMNIST simulates non-IID problems in distributed environments by partitioning data among multiple users. The dataset contains 3,550 users with 80,526 samples in total, averaging 226.83 samples per user. The standard deviation is 88.94, with a ratio of 0.39 to the mean, indicating some distribution variability. FEMNIST provides a more realistic data distribution for testing and researching distributed training challenges, such as data privacy protection and heterogeneous data processing. **The official code example for user partitioning is available [here](https://github.com/TalwalkarLab/leaf), but due to the large number of users, we've selected only a few for testing purposes. Therefore, no custom client implementation is needed when using this dataset.**
- [x] **MNIST Dataset**: Classic handwritten digit classification dataset, supporting both IID and Non-IID partitioning methods.
- [x] **CIFAR10 Dataset**: Common color image classification dataset containing 10 categories of 32x32 color images. Supports IID, Non-IID, and Dirichlet-based partitioning.
- [x] **CIFAR100 Dataset**: Extended version of CIFAR10, containing 100 categories of 32x32 color images. Suitable for testing federated learning performance with a large number of categories. Also supports IID, Non-IID, and Dirichlet-based partitioning.

### Federated Learning Algorithms

#### FedAvg
Federated Averaging (FedAvg) is the fundamental federated learning algorithm proposed by Google. It performs multiple rounds of gradient descent on local devices and then averages the parameters on the server to build a global model.

#### FedProx
FedProx improves FedAvg by adding a proximal term that limits the difference between local updates and the global model, enhancing system robustness, especially in heterogeneous data environments.

#### MOON
MOON (Model-Contrastive Federated Learning) uses contrastive learning to address client drift by making local models learn not only the classification task but also minimize representation differences with the global model.

#### SCAFFOLD
SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) uses variance reduction techniques to correct client update drift through control variates, accelerating convergence and improving performance. SCAFFOLD modifies gradients to correct client drift: $g_i ← g_i - c_i + c$, where $g_i$ is the original gradient, $c_i$ is the client control variate, and $c$ is the global control variate. Gradient clipping is implemented for stability.

#### FedDistill
Federated Distillation shares knowledge by transmitting softmax outputs (logits) instead of model parameters, reducing communication overhead and enabling clients to use personalized models.

#### FedGen
Federated Learning with Generative Models (FedGen) is an innovative federated learning approach that uses generative models to address data heterogeneity. FedGen employs a central generator to synthesize features, helping clients learn missing class data, particularly suitable for solving Non-IID data distribution problems.

### Future Plans

- [ ] **Homomorphic Encryption Federated Learning**: Incorporate homomorphic encryption to protect data privacy during training.
- [ ] **Differential Privacy Federated Learning**: Implement differential privacy mechanisms to protect user data privacy.

## Requirements

- Python 3.9
- Other dependencies: `torch`, `torchvision`, `numpy`, `matplotlib`, etc. (see `requirements.txt` for details)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OkayYang/federated-learning-framework.git
   cd federated-learning-framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface
The framework provides a command-line interface that can be run as follows:

```bash
# Basic usage - with default parameters
python main.py

# With parameters
python main.py --strategy fedavg --dataset cifar10 --partition dirichlet --num_clients 10 --dir_beta 0.3 --batch_size 64 --local_epochs 20 --comm_rounds 30 --ratio_client 0.8 --lr 0.01 --optimizer adam --seed 42 --plot_distribution True
```

#### Available Parameters:

- **Dataset Parameters**:
  - `--dataset`: Specify dataset (femnist, mnist, cifar10, cifar100)
  - `--partition`: Data partitioning method (iid, noiid, dirichlet), only for MNIST/CIFAR series
  - `--dir_beta`: Dirichlet distribution parameter when using dirichlet partitioning
  - `--num_clients`: Number of clients for MNIST/CIFAR series datasets

- **Training Parameters**:
  - `--batch_size`: Training batch size
  - `--local_epochs`: Number of local training epochs
  - `--comm_rounds`: Number of communication rounds
  - `--ratio_client`: Proportion of clients participating in each round
  - `--lr`: Learning rate
  - `--optimizer`: Optimizer type (adam or sgd)

- **Algorithm Parameters**:
  - `--strategy`: Federated learning strategy (fedavg, fedprox, moon, scaffold, feddistill, fedgen, fedspd, fedalone)

- **Other Parameters**:
  - `--seed`: Random seed for reproducibility
  - `--plot_distribution`: Whether to plot client label distribution

### Code Usage

#### 1. Load Dataset
Load `FEMNIST` or `MNIST` dataset and partition it among clients:

```python
from fl.data import datasets
# FEMNIST dataset has built-in clients, no need for custom clients
# Load the FEMNIST dataset
client_list, dataset_dict = datasets.load_feminist_dataset()

# Load the MNIST dataset for clients
# client_list = ["alice","bob","rick"]
# dataset_dict = datasets.load_mnist_dataset(client_list)
```

#### 2. Configure Model and Client Training Parameters
Define the federated learning model (e.g., `FeMNISTNet`) and training parameters:

```python
from fl.model.model import FeMNISTNet
from fl.client.fl_base import ModelConfig
from fl.utils import optim_wrapper
import torch.optim as optim

loss_fn = torch.nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-2)

model_config = ModelConfig(
    model_fn=FeMNISTNet,  # Model to use
    loss_fn=loss_fn,      # Loss function
    optim_fn=optim_fn,    # Optimizer
    epochs=10,            # Number of epochs
    batch_size=32         # Batch size
)
```

#### 3. Start Federated Learning Server
Initialize and start the federated learning server with the selected algorithm:

```python
from fl.server.fl_server import FLServer

# Using FedAvg algorithm
fl_server = FLServer(
    client_list=client_list,
    strategy="fedavg",
    model_config=model_config,
    client_dataset_dict=dataset_dict
)

history = fl_server.fit(comm_rounds=100, ratio_client=1)  # 100 communication rounds, client ratio
```

#### 4. Visualize Results
After training, visualize global and per-client training metrics:

```python
from fl.utils import plot_global_metrics, plot_worker_metrics

plot_global_metrics(history)  # Global performance metrics
plot_worker_metrics(history)  # Per-client performance metrics
```

### Visualization and Comparison Tools

The framework provides various visualization tools for analyzing and comparing the performance of different federated learning algorithms. These tools help researchers intuitively understand the advantages and disadvantages of various strategies.

#### Strategy Comparison Visualization

The `compare_strategies.py` script provides a comprehensive visualization tool for comparing different federated learning algorithms:

```bash
# Run strategy comparison script
python compare_strategies.py
```

This script automatically:
1. Loads all saved history files from the `./plots/history/` directory
2. Generates three subplots showing:
   - Training Loss
   - Test Loss
   - Test Accuracy

### Algorithm Comparison Experiments

The framework provides a convenient comparison script `run_comparison.sh` that can automatically run all supported algorithms and generate comparison results:

```bash
# Run comparison experiments for all algorithms
bash run_comparison.sh

# Generated comparison charts are located in ./plots/comparison/ directory
```

The comparison experiments show the performance differences of different algorithms in terms of training loss, test loss, and test accuracy, helping to choose the most suitable federated learning algorithm for specific scenarios.

## Contributing

Issues and pull requests are welcome to improve this project. If you have any suggestions or feature requests, please leave a message in the issues section.

## License

This project is licensed under [Apache-2.0](LICENSE.txt).