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


class AbstractPG(ABC):
    def __init__(self: "AbstractPG", config: Config) -> None:
        self.config = config
        if config.hyperparameters["policy_gradient"].get("use_double_precision", False):
            torch.set_default_tensor_type("torch.DoubleTensor")
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
        q_net_parameters: NNParameters = self.config.hyperparameters["policy_gradient"][
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
            lr=self.config.hyperparameters["policy_gradient"]["policy_learning_rate"],
        )
        self.q_net_optimizer: AdamOptimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=self.config.hyperparameters["policy_gradient"]["q_net_learning_rate"],
        )
        buffer_size = self.episode_length * self.episodes_per_training_step
        self.buffer = PGBuffer(config, buffer_size)
        self.tensor_type = torch.float32

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
            print(f"Training Step {_training_step}")
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

            print("Updating Value Net")
            self._update_q_net(
                torch.tensor(obs, dtype=self.tensor_type),
                torch.tensor(actions, dtype=self.tensor_type),
                torch.tensor(rewards_to_go, dtype=self.tensor_type),
            )
            print("Finished Value Net Update")

            avg_reward_per_training_step.append(avg_training_step_reward)
            self.buffer.reset()

        return avg_reward_per_training_step

    def _run_episodes(self: "AbstractPG") -> float:
        # print("Running Episodes")
        episode_rewards: List[float] = []
        for _episode in range(self.episodes_per_training_step):
            # print(f'Episode {_episode}')
            episode_reward = 0
            obs = self.environment.reset()
            for step in range(self.episode_length):
                action, value = self._get_action_and_value(
                    torch.tensor(obs, dtype=self.tensor_type)
                )
                next_obs, reward, done, info = self.environment.step(action)
                episode_reward += reward
                self.buffer.add_transition_data(obs, action, value, reward)
                obs = next_obs

                if done:
                    # print(f'Episode {_episode} ended due to loss')
                    self.buffer.end_episode()
                    episode_rewards.append(episode_reward)
                    break
                elif step == self.episode_length - 1:
                    # print(f'Episode {_episode} ended due to reaching time limit')
                    _, last_value = self._get_action_and_value(
                        torch.tensor(obs, dtype=self.tensor_type)
                    )
                    self.buffer.end_episode(last_value=last_value)
                    episode_rewards.append(episode_reward)

        # print("Finished Running Episodes")
        return np.mean(episode_rewards)

    def _get_state_action_value(
        self: "AbstractPG", obs: torch.Tensor, action: torch.Tensor
    ) -> float:
        q_value_tensor = self.q_net.forward(obs)
        return q_value_tensor[action].item()

    def _get_state_action_values(
        self: "AbstractPG", obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        q_value_tensor = self.q_net.forward(obs)
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

    def _update_q_net(
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
            self.q_net_optimizer.zero_grad()
            q_loss = nn.MSELoss()(state_action_values, rewards_to_go)
            q_loss.backward()
            self.q_net_optimizer.step()

    def _log_probs_from_actions(
        self: "AbstractPG", obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self._get_policy(obs).log_prob(actions)

    def _log_probs(self: "AbstractPG", obs: torch.Tensor) -> torch.Tensor:
        return torch.log(self._get_policy(obs).probs)
