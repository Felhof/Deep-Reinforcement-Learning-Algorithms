from typing import Dict, List, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from typing_extensions import TypeAlias

ActivationFunction = Union[
    torch.nn.Identity, torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh
]

AdamOptimizer: TypeAlias = torch.optim.Adam

LinearLayer: TypeAlias = nn.Linear

ConvolutionSpec = Tuple[int, int, int]
PoolingLayerSpec = Tuple[str, int]

NNParameters = TypedDict(
    "NNParameters",
    {
        "convolutions": List[ConvolutionSpec],
        "pooling_layers": List[PoolingLayerSpec],
        "linear_layer_activations": List[ActivationFunction],
        "linear_layer_sizes": List[int],
        "learning_rate": float,
        "device": str,
    },
    total=False,
)

ObservationDim = Union[int, Tuple[int, int, int]]

PolicyParameters = TypedDict(
    "PolicyParameters",
    {
        "action_type": str,
        "number_of_actions": int,
        "observation_dim": ObservationDim,
        "policy_net_parameters": NNParameters,
    },
)

AgentHyperParameters = TypedDict(
    "AgentHyperParameters",
    {
        "action_type": str,
        "number_of_actions": int,
        "observation_dim": int,
        "episode_length": int,
        "episodes_per_training_step": int,
        "value_updates_per_training_step": int,
        "discount_rate": float,
        "gae_exp_mean_discount_rate": float,
        "policy_net_parameters": NNParameters,
        "q_net_parameters": NNParameters,
        "value_net_parameters": NNParameters,
        "minibatch_size": int,
        "buffer_size": int,
        "initial_exploration_rate": float,
        "pure_exploration_steps": int,
        "gradient_clipping_norm": Optional[float],
        "kl_divergence_limit": float,
        "backtracking_coefficient": float,
        "backtracking_iterations": int,
        "damping_coefficient": float,
        "conjugate_gradient_iterations": int,
        "clip_range": float,
        "actor_parameters": NNParameters,
        "critic_parameters": NNParameters,
        "initial_temperature": float,
        "learn_temperature": bool,
        "temperature_learning_rate": float,
        "soft_update_interpolation_factor": float,
    },
    total=False,
)

HyperParameters = Dict[str, AgentHyperParameters]

EpochRewards = List[List[float]]
