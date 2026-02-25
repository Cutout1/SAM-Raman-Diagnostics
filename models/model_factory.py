import torch.nn as nn
from typing import List
from models.resnet_1d import ResNet
from models.conf_cnn import ConfigurableCNN1D


def get_model(
    model_name: str,
    hidden_sizes: List[int],
    num_blocks: List[int],
    input_dim: int,
    in_channels: int,
    num_classes: int,
    activation: str = "selu",
    ) -> nn.Module:
    """Retrieves and initializes select model architecture.

    Args:
        model_name: name of desired model to initialize
        hidden_sizes: list of hidden layer sizes
        num_blocks: list of block sizes per layer
        input_dim: input dimension of model
        in_channels: number of input channels
        num_classes: number of output classes
        activation: activation function to use

    Returns:
        Initialization of model
        """
    if model_name.lower() == "resnet":
        return ResNet(
            hidden_sizes=hidden_sizes,
            num_blocks=num_blocks,
            input_dim=input_dim,
            in_channels=in_channels,
            num_classes=num_classes,
            activation=activation,
            )
    elif model_name.lower() == "conf_cnn":
        return ConfigurableCNN1D(
            input_channels=in_channels,
            input_length=input_dim,
            cnn_configs=[
            {"filters": 60, "kernel_size": 60, "stride": 1},
            {"filters": 70, "kernel_size": 11, "stride": 4},
            {"filters": 93, "kernel_size": 45, "stride": 1}
            ],
            pool_configs=[
            {"kernel_size": 5, "stride": 5},
            {"kernel_size": 4, "stride": 4},
            {"kernel_size": 2, "stride": 2}
            ],
            fc_units=[360, 224, 122],
            dropout_rate=[0.1, 0.5, 0],
            num_classes=num_classes,
            activation=activation
            )
    elif model_name.lower() == "covid_cnn":
        return ConfigurableCNN1D(
            input_channels=in_channels,
            input_length=input_dim,
            cnn_configs=[
            {"filters": 100, "kernel_size": 100, "stride": 1},
            {"filters": 100, "kernel_size": 5, "stride": 2},
            {"filters": 25, "kernel_size": 9, "stride": 5}
            ],
            pool_configs=[
            {"kernel_size": 1, "stride": 1},
            {"kernel_size": 4, "stride": 4},
            {"kernel_size": 2, "stride": 2}
            ],
            fc_units=[732, 189, 152],
            dropout_rate=[0.1, 0.7, 0],
            num_classes=num_classes,
            activation=activation
            )
    else:
        raise ValueError(f"Model {model_name} not supported. Choose from 'resnet'.")
