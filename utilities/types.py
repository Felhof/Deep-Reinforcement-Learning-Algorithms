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
    {"sizes": List[int], "activations": List[ActivationFunction]},
)

AgentHyperParameters = TypedDict(
    "AgentHyperParameters",
    {
        "episode_length": int,
        "episodes_per_training_step": int,
        "value_updates_per_training_step": int,
        "discount_rate": float,
        "generalized_advantage_estimate_exponential_mean_discount_rate": float,
        "policy_parameters": NNParameters,
        "q_net_parameters": NNParameters,
        "value_net_parameters": NNParameters,
        "policy_learning_rate": float,
        "value_net_learning_rate": float,
        "q_net_learning_rate": float,
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
        "use_double_precision": bool,
    },
    total=False,
)

HyperParameters = Dict[str, AgentHyperParameters]
