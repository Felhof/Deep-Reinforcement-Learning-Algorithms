from typing import List, Tuple, Union

import gym
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
from utilities.buffer.PGBuffer import PGBuffer
from utilities.config import Config
from utilities.nn import create_nn
from utilities.types import AdamOptimizer, NNParameters


class VPG:
    def __init__(self: "VPG", config: Config) -> None:
        self.config = config
        self.environment: gym.Env[np.ndarray, Union[int, np.ndarray]] = gym.make(
            self.config.environment_name
        )
        self.episode_length: int = self.config.episode_length

        self.episodes_per_training_step: int = self.config.hyperparameters["VPG"][
            "episodes_per_training_step"
        ]
        self.gamma: float = self.config.hyperparameters["VPG"]["discount_rate"]
        self.lamda: float = self.config.hyperparameters["VPG"][
            "generalized_advantage_estimate_exponential_mean_discount_rate"
        ]
        policy_parameters: NNParameters = self.config.hyperparameters["VPG"][
            "policy_parameters"
        ]
        q_net_parameters: NNParameters = self.config.hyperparameters["VPG"][
            "q_net_parameters"
        ]
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
            lr=self.config.hyperparameters["VPG"]["policy_learning_rate"],
        )
        self.q_net_optimizer: AdamOptimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=self.config.hyperparameters["VPG"]["q_net_learning_rate"],
        )
        buffer_size = self.episode_length * self.episodes_per_training_step
        self.buffer = PGBuffer(config, buffer_size)

    def train(self: "VPG") -> List[float]:
        avg_reward_per_training_step: List[float] = []

        def update_policy(
            obs: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor
        ) -> None:
            policy_loss = self._compute_policy_loss(obs, actions, advantages)
            policy_loss.backward()
            self.policy_optimizer.step()

        def update_q_net(
            obs: torch.Tensor, actions: torch.Tensor, rewards_to_go: torch.Tensor
        ) -> None:

            for _ in range(
                self.config.hyperparameters["VPG"]["value_updates_per_training_step"]
            ):
                state_action_values = self._get_state_action_values(obs, actions)
                self.q_net_optimizer.zero_grad()
                q_loss = nn.MSELoss()(state_action_values, rewards_to_go)
                q_loss.backward()
                self.q_net_optimizer.step()

        for _training_step in range(self.config.training_steps_per_epoch):
            self.policy_optimizer.zero_grad()
            avg_training_step_reward = self._run_episodes()

            (
                obs,
                actions,
                rewards,
                advantages,
                rewards_to_go,
            ) = self.buffer.get_data()

            update_policy(
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(advantages, dtype=torch.float32),
            )

            update_q_net(
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(rewards_to_go, dtype=torch.float32),
            )

            avg_reward_per_training_step.append(avg_training_step_reward)
            self.buffer.reset()

        return avg_reward_per_training_step

    def _run_episodes(self: "VPG") -> float:
        episode_rewards: List[float] = []
        for _episode in range(self.episodes_per_training_step):
            episode_reward = 0
            obs = self.environment.reset()
            for step in range(self.episode_length):
                action, value = self._get_action_and_value(
                    torch.tensor(obs, dtype=torch.float32)
                )
                next_obs, reward, done, info = self.environment.step(action)
                episode_reward += reward
                self.buffer.add_transition_data(obs, action, value, reward)
                obs = next_obs

                if done:
                    self.buffer.end_episode()
                    episode_rewards.append(episode_reward)
                    break
                elif step == self.episode_length - 1:
                    _, last_value = self._get_action_and_value(
                        torch.tensor(obs, dtype=torch.float32)
                    )
                    self.buffer.end_episode(last_value=last_value)
                    episode_rewards.append(episode_reward)

        return np.mean(episode_rewards)

    def _get_state_action_value(
        self: "VPG", obs: torch.Tensor, action: torch.Tensor
    ) -> float:
        q_value_tensor = self.q_net.forward(obs)
        return q_value_tensor[action].item()

    def _get_state_action_values(
        self: "VPG", obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        q_value_tensor = self.q_net.forward(obs)
        return q_value_tensor.gather(
            1, actions.unsqueeze(-1).type(torch.int64)
        ).squeeze(-1)

    def _get_policy(self: "VPG", obs: torch.Tensor) -> Categorical:
        logits: torch.Tensor = self.policy(obs)
        return Categorical(logits=logits)

    def _get_action(self: "VPG", obs: torch.Tensor) -> torch.Tensor:
        return self._get_policy(obs).sample()

    def _get_action_and_value(
        self: "VPG", obs: torch.Tensor
    ) -> Tuple[np.ndarray, float]:
        action = self._get_action(obs)
        value = self._get_state_action_value(obs, action)
        return action.numpy(), value

    def _compute_policy_loss(
        self: "VPG", obs: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        log_probs = self._get_policy(obs).log_prob(actions)
        return -(log_probs * weights).mean()
