from typing import List, Union

import torch.nn as nn
from utilities.types import ActivationFunction, LinearLayer


def create_nn(
    sizes: List[int],
    activations: List[ActivationFunction],
) -> nn.Sequential:
    layers: List[Union[LinearLayer, ActivationFunction]] = []
    for in_size, out_size, activation in zip(sizes, sizes[1:], activations):
        layers.append(nn.Linear(in_features=in_size, out_features=out_size))
        layers.append(activation)

    return nn.Sequential(*layers)


def create_value_net(
    activations: List[ActivationFunction],
    hidden_layer_sizes: List[int],
    observation_dim: int,
) -> nn.Sequential:
    assert (
        len(activations) == len(hidden_layer_sizes) + 1
    ), "Value net must be given exactly one more activation function than hidden layers"
    return create_nn(
        [observation_dim] + hidden_layer_sizes + [1],
        activations,
    )


def create_q_net(
    activations: List[ActivationFunction],
    hidden_layer_sizes: List[int],
    observation_dim: int,
    number_of_actions: int,
) -> nn.Sequential:
    assert (
        len(activations) == len(hidden_layer_sizes) + 1
    ), "Q net must be given exactly one more activation function than hidden layers"
    return create_nn(
        [observation_dim] + hidden_layer_sizes + [number_of_actions], activations
    )
