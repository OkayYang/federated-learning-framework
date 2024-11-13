# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2024/11/7 15:15
# @Describe:


import torch.nn as nn
import torch.optim as optim

from fl.data import datasets

# Import necessary libraries and modules
from fl.fl_base import ModelConfig
from fl.fl_server import FLServer
from fl.model.model import FeMNISTNet, MNISTNetCNN
from fl.utils import (
    optim_wrapper,
    plot_client_label_distribution,
    plot_global_metrics,
    plot_worker_metrics,
)


def setup_and_train_federated_model():
    """
    Set up the federated learning system, train the model, and plot the results.

    This function initializes the necessary configurations, including:
    - Loading the dataset (FEMNIST)
    - Defining the model, loss function, and optimizer
    - Setting up the federated learning server
    - Training the model using the Federated Averaging (FedAvg) strategy
    - Plotting global and worker-level performance metrics

    The training is done using the `FeMNISTNet` model on the FEMNIST dataset.
    """
    # Load the FEMNIST dataset
    client_list, dataset_dict = datasets.load_feminist_dataset()

    # Load the MNIST dataset for clients
    # client_list = ["alice","bob","rick"]
    # dataset_dict = datasets.load_mnist_dataset(client_list)

    # Define the loss function (CrossEntropyLoss for classification tasks)
    loss_fn = nn.CrossEntropyLoss

    # Set up the optimizer (Adam with a learning rate of 0.01)
    optim_fn = optim_wrapper(optim.Adam, lr=1e-2)

    # Configure the model and training parameters
    model_config = ModelConfig(
        model_fn=FeMNISTNet,  # Model function (FeMNISTNet defined in the model file)
        loss_fn=loss_fn,  # Loss function
        optim_fn=optim_fn,  # Optimizer function
        epochs=10,  # Number of training epochs
        batch_size=32,  # Batch size
    )

    # Initialize the federated learning server with the given parameters
    fl_server = FLServer(
        client_list=client_list,  # List of clients (users)
        strategy="fedavg",  # Federated learning strategy (FedAvg)
        model_config=model_config,  # Model configuration
        client_dataset_dict=dataset_dict,  # Dictionary of datasets for each client
    )

    # Start the federated learning training process (100 communication rounds)
    history = fl_server.fit(
        comm_rounds=50,  # Number of communication rounds (or federated training rounds)
        ratio_client=1,  # Ratio of clients to be sampled per round (1 means all clients)
    )

    # Plot global metrics (e.g., accuracy, loss) across all federated training rounds
    plot_global_metrics(history)

    # Plot worker metrics (performance of individual clients/workers)
    plot_worker_metrics(history)


# Run the federated learning setup and training
if __name__ == "__main__":
    setup_and_train_federated_model()
