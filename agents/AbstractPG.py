from abc import ABC, abstractmethod
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
from utilities.utils import get_dimension_format_string


class AbstractPG(ABC):
    def __init__(self: "AbstractPG", config: Config) -> None:
        self.config = config
        self.dtype_name = config.hyperparameters.get("dtype_name", "float32")
        if self.dtype_name == "float64":
            self.tensor_type = torch.float64
            torch.set_default_tensor_type("torch.DoubleTensor")
        else:
            self.tensor_type = torch.float32
            torch.set_default_tensor_type(torch.FloatTensor)
        self.environment: gym.Env[np.ndarray, Union[int, np.ndarray]] = gym.make(
            self.config.environment_name
        )
        self.episode_length: int = self.config.episode_length

        self.episodes_per_training_step: int = self.config.hyperparameters[
            "policy_gradient"
        ]["episodes_per_training_step"]
        self.gamma: float = self.config.hyperparameters["policy_gradient"][
            "discount_rate"
        ]
        self.lamda: float = self.config.hyperparameters["policy_gradient"][
            "generalized_advantage_estimate_exponential_mean_discount_rate"
        ]
        policy_parameters: NNParameters = self.config.hyperparameters[
            "policy_gradient"
        ]["policy_parameters"]
        value_net_parameters: NNParameters = self.config.hyperparameters[
            "policy_gradient"
        ]["value_net_parameters"]
        self.policy: nn.Sequential = create_nn(
            policy_parameters["sizes"],
            policy_parameters["activations"],
        )
        self.value_net: nn.Sequential = create_nn(
            value_net_parameters["sizes"],
            value_net_parameters["activations"],
        )
        self.policy_optimizer: AdamOptimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.hyperparameters["policy_gradient"]["policy_learning_rate"],
        )
        self.value_net_optimizer: AdamOptimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=self.config.hyperparameters["policy_gradient"][
                "value_net_learning_rate"
            ],
        )
        self.buffer = PGBuffer(config, self.episodes_per_training_step)

    @abstractmethod
    def _update_policy(
        self: "AbstractPG",
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        pass

    def train(self: "AbstractPG") -> List[float]:
        avg_reward_per_training_step: List[float] = []

        for _training_step in range(self.config.training_steps_per_epoch):
            # print(f"Training Step {_training_step}")
            self.policy_optimizer.zero_grad()
            avg_training_step_reward = self._run_episodes()

            (
                obs,
                actions,
                rewards,
                advantages,
                rewards_to_go,
            ) = self.buffer.get_data()

            self._update_policy(
                torch.tensor(obs, dtype=self.tensor_type),
                torch.tensor(actions, dtype=self.tensor_type),
                torch.tensor(advantages, dtype=self.tensor_type),
            )

            # print("Updating Value Net")
            self._update_value_net(
                torch.tensor(obs, dtype=self.tensor_type),
                torch.tensor(actions, dtype=self.tensor_type),
                torch.tensor(rewards_to_go, dtype=self.tensor_type),
            )
            # print("Finished Value Net Update")

            avg_reward_per_training_step.append(avg_training_step_reward)
            self.buffer.reset()

        return avg_reward_per_training_step

    def _run_episodes(self: "AbstractPG") -> float:
        episode_rewards: List[float] = []
        for _episode in range(self.episodes_per_training_step):
            episode_reward = 0
            states = np.zeros(
                self.episode_length,
                dtype=get_dimension_format_string(
                    self.config.observation_dim,
                    dtype=self.dtype_name,
                ),
            )
            actions = np.zeros(
                self.episode_length,
                dtype=get_dimension_format_string(
                    self.config.action_dim,
                    dtype=self.dtype_name,
                ),
            )
            values = np.zeros(
                self.episode_length,
                dtype=self.dtype_name,
            )
            rewards = np.zeros(
                self.episode_length,
                dtype=self.dtype_name,
            )

            obs = self.environment.reset()
            for step in range(self.episode_length):
                action, value = self._get_action_and_value(
                    torch.tensor(obs, dtype=self.tensor_type)
                )
                next_obs, reward, done, info = self.environment.step(action)
                episode_reward += reward
                states[step] = obs
                actions[step] = action
                values[step] = value
                rewards[step] = reward
                obs = next_obs

                if done:
                    self.buffer.add_transition_data(states, actions, values, rewards)
                    episode_rewards.append(episode_reward)
                    break
                elif step == self.episode_length - 1:
                    _, last_value = self._get_action_and_value(
                        torch.tensor(obs, dtype=self.tensor_type)
                    )
                    self.buffer.add_transition_data(
                        states, actions, values, rewards, last_value=last_value
                    )
                    episode_rewards.append(episode_reward)

        return float(np.mean(episode_rewards))

    def _get_state_action_value(
        self: "AbstractPG", obs: torch.Tensor, action: torch.Tensor
    ) -> float:
        q_value_tensor = self.value_net.forward(obs)
        return q_value_tensor[action].item()

    def _get_state_action_values(
        self: "AbstractPG", obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        q_value_tensor = self.value_net.forward(obs)
        return q_value_tensor.gather(
            1, actions.unsqueeze(-1).type(torch.int64)
        ).squeeze(-1)

    def _get_policy(self: "AbstractPG", obs: torch.Tensor) -> Categorical:
        logits: torch.Tensor = self.policy(obs)
        return Categorical(logits=logits)

    def _get_action(self: "AbstractPG", obs: torch.Tensor) -> torch.Tensor:
        return self._get_policy(obs).sample()

    def _get_action_and_value(
        self: "AbstractPG", obs: torch.Tensor
    ) -> Tuple[np.ndarray, float]:
        action = self._get_action(obs)
        value = self._get_state_action_value(obs, action)
        return action.numpy(), value

    def _compute_policy_loss(
        self: "AbstractPG",
        obs: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = self._log_probs_from_actions(obs, actions)
        return -(log_probs * weights).mean()

    def _update_value_net(
        self: "AbstractPG",
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards_to_go: torch.Tensor,
    ) -> None:
        for _ in range(
            self.config.hyperparameters["policy_gradient"][
                "value_updates_per_training_step"
            ]
        ):
            state_action_values = self._get_state_action_values(obs, actions)
            self.value_net_optimizer.zero_grad()
            q_loss = nn.MSELoss()(state_action_values, rewards_to_go)
            q_loss.backward()
            self.value_net_optimizer.step()

    def _log_probs_from_actions(
        self: "AbstractPG", obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self._get_policy(obs).log_prob(actions)

    def _log_probs(self: "AbstractPG", obs: torch.Tensor) -> torch.Tensor:
        return torch.log(self._get_policy(obs).probs)
