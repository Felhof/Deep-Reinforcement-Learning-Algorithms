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
