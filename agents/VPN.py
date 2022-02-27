from typing import List, Tuple, Union

import gym  # type: ignore
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
from utilities.config import Config  # type: ignore
from utilities.types import (  # type: ignore
    ActivationFunction,
    AdamOptimizer,
    LinearLayer,
    NNParameters,
)


def create_nn(
    sizes: List[int],
    activations: List[ActivationFunction],
) -> nn.Sequential:
    layers: List[Union[LinearLayer, ActivationFunction]] = []
    for in_size, out_size, activation in zip(sizes, sizes[1:], activations):
        layers.append(nn.Linear(in_features=in_size, out_features=out_size))
        layers.append(activation)

    return nn.Sequential(*layers)


def get_dimension_format_string(
    x_dim: int, y_dim: int = 1, dtype: str = "float32"
) -> str:
    if y_dim == 1:
        return f"{x_dim}{dtype}"
    return f"({x_dim},{y_dim}){dtype}"


class VPN:
    def __init__(self: "VPN", config: Config) -> None:
        self.config = config
        self.environment: gym.Env[np.ndarray, Union[int, np.ndarray]] = gym.make(
            self.config.environment_name
        )
        self.episode_length: int = self.config.hyperparameters["VPN"]["episode_length"]
        self.episodes_per_training_step: int = self.config.hyperparameters["VPN"].get(
            "episodes_per_training_step"
        )
        self.gamma: float = self.config.hyperparameters["VPN"].get("discount_rate")
        self.lamda: float = self.config.hyperparameters["VPN"].get(
            "generalized_advantage_estimate_exponential_mean_discount_rate"
        )
        policy_parameters: NNParameters = self.config.hyperparameters["VPN"].get(
            "policy_parameters"
        )
        q_net_parameters: NNParameters = self.config.hyperparameters["VPN"].get(
            "q_net_parameters"
        )
        self.policy: nn.Sequential = create_nn(
            policy_parameters["sizes"],
            policy_parameters["activations"],
        )
        self.q_net: nn.Sequential = create_nn(
            q_net_parameters["sizes"],
            q_net_parameters["activations"],
        )
        self.policy_optimizer: AdamOptimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.hyperparameters["VPN"].get("policy_learning_rate"),
        )
        self.q_net_optimizer: AdamOptimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=self.config.hyperparameters["VPN"].get("q_net_learning_rate"),
        )

    def train(self: "VPN") -> np.ndarray:
        avg_reward_per_step = np.empty(self.config.training_steps_per_epoch)
        for step in range(self.config.training_steps_per_epoch):
            self.policy_optimizer.zero_grad()
            (
                obs_histories,
                action_histories,
                reward_histories,
                done_histories,
            ) = self._run_episodes()

            obs_tensor = torch.tensor(obs_histories, dtype=torch.float32)
            q_value_tensor = self.q_net.forward(obs_tensor)
            action_tensor = torch.tensor(action_histories, dtype=torch.int64)
            state_action_values = q_value_tensor.gather(
                2, action_tensor.unsqueeze(-1).type(torch.int64)
            ).squeeze(-1)
            (
                advantage_histories,
                value_target_histories,
            ) = self._get_generalized_advantage_estimates(
                state_action_values.squeeze().detach().numpy(),
                reward_histories,
                done_histories,
            )
            advantage_tensor = torch.tensor(advantage_histories, dtype=torch.float32)
            reward_tensor = torch.tensor(reward_histories, dtype=torch.float32)

            avg_reward_per_step[step] = reward_tensor.sum(dim=1).mean()
            rewards_to_go = torch.cumsum(reward_tensor.flip(1), 1).flip(1)

            policy_loss = self._compute_policy_loss(
                obs_tensor, action_tensor, advantage_tensor
            )
            policy_loss.backward()
            self.policy_optimizer.step()

            for _ in range(
                self.config.hyperparameters["VPN"].get(
                    "value_updates_per_training_step"
                )
            ):
                self.q_net_optimizer.zero_grad()
                q_loss = nn.MSELoss()(state_action_values, rewards_to_go)
                q_loss.backward()
                self.q_net_optimizer.step()

        return avg_reward_per_step

    def _run_episodes(
        self: "VPN",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_histories = np.empty(
            self.episodes_per_training_step,
            dtype=get_dimension_format_string(
                self.episode_length, self.config.observation_dim
            ),
        )
        action_histories = np.empty(
            self.episodes_per_training_step,
            dtype=get_dimension_format_string(
                self.episode_length, self.config.action_dim
            ),
        )
        reward_histories = np.empty(
            self.episodes_per_training_step,
            dtype=get_dimension_format_string(self.episode_length),
        )
        done_histories = np.empty(
            self.episodes_per_training_step,
            dtype=get_dimension_format_string(self.episode_length, dtype="bool"),
        )
        for episode in range(self.episodes_per_training_step):
            obs_history = np.zeros(
                self.episode_length,
                dtype=get_dimension_format_string(self.config.observation_dim),
            )
            action_history = np.zeros(
                self.episode_length,
                dtype=get_dimension_format_string(self.config.action_dim),
            )
            reward_history = np.zeros(self.episode_length, dtype=np.float32)
            done_history = np.ones(self.episode_length, dtype=bool)
            obs = self.environment.reset()
            for step in range(self.episode_length):
                obs_history[step] = obs
                action = self._get_action(obs)
                next_obs, reward, done, info = self.environment.step(action)
                action_history[step] = action
                reward_history[step] = reward
                done_history[step] = done
                obs = next_obs
                if done:
                    break
            obs_histories[episode] = obs_history
            action_histories[episode] = action_history
            reward_histories[episode] = reward_history
        return obs_histories, action_histories, reward_histories, done_histories

    def _get_policy(self: "VPN", obs: torch.Tensor) -> Categorical:
        logits: torch.Tensor = self.policy(obs)
        return Categorical(logits=logits)

    def _get_action(self: "VPN", obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        return self._get_policy(obs_tensor).sample().numpy()

    def _compute_policy_loss(
        self: "VPN", obs: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        log_probs = self._get_policy(obs).log_prob(actions)
        return -(log_probs * weights).mean()

    def _get_generalized_advantage_estimates(
        self: "VPN",
        value_histories: np.ndarray,
        reward_histories: np.ndarray,
        done_histories: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantage_histories = np.empty(
            self.episodes_per_training_step,
            dtype=get_dimension_format_string(self.episode_length),
        )
        value_target_histories = np.empty(
            self.episodes_per_training_step,
            dtype=get_dimension_format_string(self.episode_length),
        )

        def _get_episode_generalized_advantage_estimate(
            values: np.ndarray, rewards: np.ndarray, done: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            advantage = np.zeros(self.episode_length)
            for t in reversed(range(self.episode_length)):
                next_value = values[t + 1] if t < self.episode_length - 1 else 0
                delta = rewards[t] + self.gamma * next_value * done[t] - values[t]
                next_advantage = advantage[t + 1] if t < self.episode_length - 1 else 0
                advantage[t] = (
                    delta + self.lamda * self.gamma * next_advantage * done[t]
                )
            value_targets = np.add(
                advantage,
                values,
            )
            return advantage, value_targets

        for idx, (values_ep, rewards_ep, done_ep) in enumerate(
            zip(value_histories, reward_histories, done_histories)
        ):
            (
                advantage_ep,
                value_targets_ep,
            ) = _get_episode_generalized_advantage_estimate(
                values_ep, rewards_ep, done_ep
            )
            advantage_histories[idx] = advantage_ep
            value_target_histories[idx] = value_targets_ep

        return advantage_histories, value_target_histories
