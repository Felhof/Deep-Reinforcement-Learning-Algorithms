from abc import ABC, abstractmethod
from typing import Iterator, OrderedDict

import torch
from torch import nn
from torch.distributions import Categorical, Distribution, Normal
from torch.nn import Parameter
from utilities.nn import create_nn
from utilities.types import (
    AdamOptimizer,
    NNParameters,
    ObservationDim,
    PolicyParameters,
)

ACTION_TYPES = ["Discrete", "Continuous"]


class Policy(ABC):
    def __init__(
        self: "Policy",
        output_size: int,
        observation_dim: ObservationDim,
        parameters: NNParameters,
    ) -> None:
        self.policy_net: nn.Sequential = create_nn(
            observation_dim=observation_dim,
            output_size=output_size,
            parameters=parameters,
        )
        self.optimizer: AdamOptimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=parameters["learning_rate"],
        )

    @abstractmethod
    def get_policy(
        self: "Policy", obs: torch.Tensor, detached: bool = False
    ) -> Distribution:
        pass

    def compute_loss(
        self: "Policy",
        obs: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = self.get_log_probs_from_actions(obs, actions)
        return -(log_probs * weights).mean()

    def get_action(self: "Policy", obs: torch.Tensor) -> torch.Tensor:
        return self.get_policy(obs).sample()

    def get_log_probs_from_actions(
        self: "Policy", obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.get_policy(obs).log_prob(actions)

    def get_parameters(self: "Policy") -> Iterator[Parameter]:
        return self.policy_net.parameters()

    def get_state_dict(self: "Policy") -> OrderedDict[str, torch.Tensor]:
        return self.policy_net.state_dict()

    def load_state_dict(
        self: "Policy", state_dict: OrderedDict[str, torch.Tensor]
    ) -> None:
        self.policy_net.load_state_dict(state_dict)

    def reset_gradients(self) -> None:
        self.optimizer.zero_grad()

    def update_gradients(
        self: "Policy",
    ) -> None:
        self.optimizer.step()


class CategoricalPolicy(Policy):
    def __init__(
        self: "CategoricalPolicy",
        number_of_actions: int,
        observation_dim: ObservationDim,
        policy_net_parameters: NNParameters,
    ) -> None:
        assert (
            number_of_actions > 1
        ), "Must have more than 1 action for categorical policy."
        super().__init__(
            output_size=number_of_actions,
            observation_dim=observation_dim,
            parameters=policy_net_parameters,
        )

    def get_policy(
        self: "CategoricalPolicy", obs: torch.Tensor, detached: bool = False
    ) -> Categorical:
        logits: torch.Tensor = self.policy_net(obs)
        if detached:
            return Categorical(logits=logits.detach().clone())
        return Categorical(logits=logits)

    def compute_kl_loss(self: "Policy", obs: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.policy_net(obs)
        kl_div = torch.distributions.kl.kl_divergence(
            Categorical(logits=logits), Categorical(logits=logits.detach().clone())
        )
        return kl_div.mean()

    def get_log_probs(self: "CategoricalPolicy", obs: torch.Tensor) -> torch.Tensor:
        return torch.log(self.get_policy(obs).probs)


class ContinuousPolicy(Policy):
    def __init__(
        self: "ContinuousPolicy",
        number_of_actions: int,
        observation_dim: ObservationDim,
        policy_net_parameters: NNParameters,
    ):
        self.number_of_actions = number_of_actions
        super().__init__(
            output_size=number_of_actions
            * 2,  # for every action we need a mean and std
            observation_dim=observation_dim,
            parameters=policy_net_parameters,
        )

    def get_policy(
        self: "ContinuousPolicy", obs: torch.Tensor, detached: bool = False
    ) -> Distribution:
        mean, log_std = self.policy_net(obs).split(self.number_of_actions, dim=-1)
        mean = mean.squeeze(-1)
        std = log_std.exp().squeeze(-1)
        if detached:
            return Normal(mean.detach().clone(), std.detach().clone())
        return Normal(mean, std)


def create_policy(parameters: PolicyParameters) -> Policy:
    action_type = parameters["action_type"]
    assert (
        action_type in ACTION_TYPES
    ), f"Action type must be one of {', '.join(ACTION_TYPES)}."

    number_of_actions = parameters["number_of_actions"]
    observation_dim = parameters["observation_dim"]
    policy_net_parameters = parameters["policy_net_parameters"]
    if action_type == "Discrete":
        return CategoricalPolicy(
            number_of_actions, observation_dim, policy_net_parameters
        )
    if action_type == "Continuous":
        return ContinuousPolicy(
            number_of_actions, observation_dim, policy_net_parameters
        )
    raise ValueError(f"Action type must be one of {', '.join(ACTION_TYPES)}.")
