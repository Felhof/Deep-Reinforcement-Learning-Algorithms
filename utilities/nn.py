from collections import deque
from typing import cast, List

import torch.nn as nn
from utilities.types import (
    ActivationFunction,
    ConvolutionSpec,
    NNParameters,
    ObservationDim,
    PoolingLayerSpec,
)


def _add_convolutional_layers(
    model: nn.Sequential,
    in_channels: int,
    convolutions: List[ConvolutionSpec],
    pooling_layers: List[PoolingLayerSpec],
) -> None:
    pooling_deque = deque(pooling_layers)
    for out_channels, kernel_size, stride in convolutions:
        model.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            )
        )
        model.append(nn.ReLU())
        if len(pooling_deque) > 0:
            layer_type, kernel_size = pooling_deque.popleft()
            match layer_type:
                case "Max":
                    model.append(nn.MaxPool2d(kernel_size=kernel_size))
        in_channels = out_channels
    model.append(nn.Flatten())


def _add_linear_layers(
    model: nn.Sequential,
    sizes: List[int],
    activations: List[ActivationFunction],
) -> None:
    for in_size, out_size, activation in zip(sizes, sizes[1:], activations):
        model.append(nn.Linear(in_features=in_size, out_features=out_size))
        model.append(activation)


def create_nn(
    observation_dim: ObservationDim, output_size: int, parameters: NNParameters
) -> nn.Sequential:
    convolutions = parameters.get("convolutions", [])
    assert (
        isinstance(observation_dim, int) or convolutions != []
    ), "If the observation is an image there must be a convolutional layer."

    model = nn.Sequential()
    if convolutions:
        in_channels = cast(tuple, observation_dim)[2]
        pooling_layers = parameters.get("pooling_layers", [])
        _add_convolutional_layers(
            model,
            in_channels=in_channels,
            convolutions=convolutions,
            pooling_layers=pooling_layers,
        )

    sizes: List[int] = (
        parameters["linear_layer_sizes"] + [output_size]
        if convolutions
        else [cast(int, observation_dim)]
        + parameters["linear_layer_sizes"]
        + [output_size]
    )
    _add_linear_layers(
        model, sizes=sizes, activations=parameters["linear_layer_activations"]
    )

    return model


def create_value_net(
    observation_dim: ObservationDim,
    parameters: NNParameters,
) -> nn.Sequential:
    assert (
        len(parameters["linear_layer_activations"])
        == len(parameters["linear_layer_sizes"]) + 1
    ), "Value net must be given exactly one more activation function than hidden layers"
    return create_nn(
        observation_dim=observation_dim, output_size=1, parameters=parameters
    )


def create_q_net(
    observation_dim: ObservationDim,
    number_of_actions: int,
    parameters: NNParameters,
) -> nn.Sequential:
    assert (
        len(parameters["linear_layer_activations"])
        == len(parameters["linear_layer_sizes"]) + 1
    ), "Q net must be given exactly one more activation function than hidden layers"
    return create_nn(
        observation_dim=observation_dim,
        output_size=number_of_actions,
        parameters=parameters,
    )
