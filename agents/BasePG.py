from abc import abstractmethod
from typing import Tuple

from agents.BaseAgent import BaseAgent
from agents.Policy import create_policy, Policy
import numpy as np
import torch
import torch.nn as nn
from utilities.buffer.PGBuffer import PGBuffer
from utilities.nn import create_value_net
from utilities.types.types import AdamOptimizer, NNParameters, PolicyParameters
from utilities.utils import get_dimension_format_string


class BasePG(BaseAgent):
    def __init__(self: "BasePG", **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma: float = self.config.hyperparameters["policy_gradient"][
            "discount_rate"
        ]
        self.episodes_per_training_step: int = self.config.hyperparameters[
            "policy_gradient"
        ]["episodes_per_training_step"]
        self.lamda: float = self.config.hyperparameters["policy_gradient"][
            "gae_exp_mean_discount_rate"
        ]
        policy_parameters: PolicyParameters = {
            "action_type": self.environment.action_type,
            "number_of_actions": self.environment.number_of_actions,
            "observation_dim": self.environment.observation_dim,
            "policy_net_parameters": self.config.hyperparameters["policy_gradient"][
                "policy_net_parameters"
            ]
            | {"device": self.device},
        }
        self.policy: Policy = create_policy(policy_parameters)
        value_net_parameters: NNParameters = self.config.hyperparameters[
            "policy_gradient"
        ]["value_net_parameters"] | {"device": self.device}
        self.value_net: nn.Sequential = create_value_net(
            observation_dim=self.environment.observation_dim,
            parameters=value_net_parameters,
        )
        self.value_net_optimizer: AdamOptimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=value_net_parameters["learning_rate"]
        )
        self.buffer = PGBuffer(
            self.config,
            buffer_size=self.episodes_per_training_step,
            action_dim=self.environment.action_dim,
            observation_dim=self.environment.observation_dim,
        )
        self.result_storage = kwargs["result_storage"]

    @abstractmethod
    def _update_policy(
        self: "BasePG",
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        pass

    def evaluate(self: "BasePG", time_to_save: bool = False) -> float:
        self.policy.policy_net.eval()
        self.value_net.eval()
        reward = super().evaluate(time_to_save=time_to_save)
        self.policy.policy_net.train()
        self.value_net.train()
        return reward

    def get_best_action(self: "BasePG", obs: torch.Tensor) -> np.ndarray:
        best_action = self.policy.get_best_action(obs)
        return best_action.numpy()

    def load(self: "BasePG", filename) -> None:
        self.policy.policy_net.load_state_dict(
            torch.load(f"{filename}_policy_model.pt")
        )
        self.value_net.load_state_dict(torch.load(f"{filename}_value_model.pt"))
        if torch.cuda.is_available() and self.config.device != "cpu":
            self.policy.policy_net = self.policy.policy_net.cuda()
            self.value_net = self.value_net.cuda()

    def _training_loop(self: "BasePG") -> None:
        self.policy.reset_gradients()

        self.logger.info("Running episodes")
        self._run_episodes()
        self.logger.log_table(
            scope="training_step", level="INFO", attributes=["reward"]
        )

        (
            obs,
            actions,
            rewards,
            advantages,
            rewards_to_go,
        ) = self.buffer.get_data()

        self.logger.info("Updating policy")
        self.logger.start_timer(scope="epoch", level="INFO", attribute="policy_update")
        self._update_policy(
            torch.tensor(obs, dtype=self.tensor_type, device=self.device),
            torch.tensor(actions, dtype=self.tensor_type, device=self.device),
            torch.tensor(advantages, dtype=self.tensor_type, device=self.device),
        )
        self.logger.stop_timer(scope="epoch", level="INFO", attribute="policy_update")

        self.logger.info("Updating value net")
        self._update_value_net(
            torch.tensor(obs, dtype=self.tensor_type, device=self.device),
            torch.tensor(rewards_to_go, dtype=self.tensor_type, device=self.device),
        )

        self.buffer.reset()
        self.logger.clear(scope="training_step")

    def _run_episodes(self: "BasePG") -> None:
        self.logger.start_timer(scope="epoch", level="INFO", attribute="episodes")
        for _episode in range(self.episodes_per_training_step):
            episode_reward: float = 0
            states = np.zeros(
                self.episode_length,
                dtype=get_dimension_format_string(
                    self.environment.observation_dim,
                    dtype=self.dtype_name,
                ),
            )
            actions = np.zeros(
                self.episode_length,
                dtype=get_dimension_format_string(
                    self.environment.action_dim,
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

            obs, _ = self.environment.reset()
            for step in range(self.episode_length):
                self.current_timestep += 1
                action, value = self._get_action_and_value(
                    torch.tensor(
                        np.array(obs), dtype=self.tensor_type, device=self.device
                    )
                )
                next_obs, reward, terminated, truncated, info = self.environment.step(
                    action
                )
                episode_reward += float(reward)
                states[step] = obs
                actions[step] = action
                values[step] = value
                rewards[step] = float(reward)
                obs = next_obs

                if self.has_reached_timestep_limit():
                    break
                if terminated:
                    self.buffer.add_transition_data(states, actions, values, rewards)
                    self.logger.store(scope="training_step", reward=episode_reward)
                    break
                if step == self.episode_length - 1 or truncated:
                    _, last_value = self._get_action_and_value(
                        torch.tensor(
                            np.array(obs), dtype=self.tensor_type, device=self.device
                        )
                    )
                    self.buffer.add_transition_data(
                        states, actions, values, rewards, last_value=last_value
                    )
                    self.logger.store(scope="training_step", reward=episode_reward)

            if self.has_reached_timestep_limit():
                break

        self.logger.stop_timer(scope="epoch", level="INFO", attribute="episodes")

    def _get_state_value(self: "BasePG", obs: torch.Tensor) -> float:
        value_tensor = self.value_net.forward(obs)
        return value_tensor.item()

    def _get_state_values(self: "BasePG", obs: torch.Tensor) -> torch.Tensor:
        value_tensor = self.value_net.forward(obs)
        return value_tensor.squeeze(-1)

    def _get_action_and_value(
        self: "BasePG", obs: torch.Tensor
    ) -> Tuple[np.ndarray, float]:
        action = self.policy.get_action(obs)
        value = self._get_state_value(obs)
        return action.cpu().numpy(), value

    def _update_value_net(
        self: "BasePG", obs: torch.Tensor, rewards_to_go: torch.Tensor
    ) -> None:
        self.logger.start_timer(
            scope="epoch", level="INFO", attribute="value_net_update"
        )
        for _ in range(
            self.config.hyperparameters["policy_gradient"][
                "value_updates_per_training_step"
            ]
        ):
            state_values = self._get_state_values(obs)
            self.value_net_optimizer.zero_grad()
            loss = nn.MSELoss()(state_values, rewards_to_go)
            loss.backward()
            self.value_net_optimizer.step()
        self.logger.stop_timer(
            scope="epoch", level="INFO", attribute="value_net_update"
        )
