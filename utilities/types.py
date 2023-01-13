from typing import Dict, List, Optional, TypedDict, Union

import torch
import torch.nn as nn
from typing_extensions import TypeAlias

ActivationFunction = Union[
    torch.nn.Identity, torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh
]

AdamOptimizer: TypeAlias = torch.optim.Adam

LinearLayer: TypeAlias = nn.Linear

NNParameters = TypedDict(
    "NNParameters",
    {
        "activations": List[ActivationFunction],
        "hidden_layer_sizes": List[int],
        "learning_rate": float,
    },
)

PolicyParameters = TypedDict(
    "PolicyParameters",
    {
        "action_type": str,
        "number_of_actions": int,
        "observation_dim": int,
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
        "random_episodes": int,
        "gradient_clipping_norm": Optional[float],
        "kl_divergence_limit": float,
        "backtracking_coefficient": float,
        "backtracking_iterations": int,
        "damping_coefficient": float,
        "conjugate_gradient_iterations": int,
        "dtype_name": str,
        "clip_range": float,
    },
    total=False,
)

HyperParameters = Dict[str, AgentHyperParameters]

EpochRewards = List[List[float]]
