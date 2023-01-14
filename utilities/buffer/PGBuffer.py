from typing import Tuple

import numpy as np
from utilities.config import Config
from utilities.types import ObservationDim
from utilities.utils import get_dimension_format_string


class PGBuffer:
    def __init__(
        self: "PGBuffer",
        config: Config,
        buffer_size: int = 40000,
        action_dim: int = 1,
        observation_dim: ObservationDim = 2,
    ) -> None:
        self.buffer_size = buffer_size
        self.episode_length = config.episode_length
        self.gamma = config.hyperparameters["policy_gradient"]["discount_rate"]
        self.lamda = config.hyperparameters["policy_gradient"][
            "gae_exp_mean_discount_rate"
        ]
        self.top_index = 0
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.dtype_name: str = config.hyperparameters["policy_gradient"].get(
            "dtype_name", "float32"
        )
        (
            self.states,
            self.actions,
            self.values,
            self.rewards,
            self.advantages,
            self.rewards_to_go,
        ) = self._create_empty_buffers()

    def _get_episode_generalized_advantage_estimates(
        self: "PGBuffer", rewards: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        delta: np.ndarray = rewards + self.gamma * values[1:] - values[:-1]

        advantage = np.zeros(self.episode_length)

        for t in reversed(range(self.episode_length)):
            next_advantage = advantage[t + 1] if t < self.episode_length - 1 else 0
            advantage[t] = delta[t] + self.lamda * self.gamma * next_advantage

        return advantage

    def _create_empty_buffers(
        self: "PGBuffer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        state_dim = (
            (self.episode_length,) + self.observation_dim
            if isinstance(self.observation_dim, tuple)
            else (self.episode_length, self.observation_dim)
        )
        states = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(
                state_dim,
                dtype=self.dtype_name,
            ),
        )
        actions = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(
                (self.episode_length,) + (self.action_dim,),
                dtype=self.dtype_name,
            ),
        )
        values = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(
                self.episode_length, dtype=self.dtype_name
            ),
        )
        rewards = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(
                self.episode_length, dtype=self.dtype_name
            ),
        )
        advantages = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(
                self.episode_length, dtype=self.dtype_name
            ),
        )
        rewards_to_go = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(
                self.episode_length, dtype=self.dtype_name
            ),
        )

        return states, actions, values, rewards, advantages, rewards_to_go

    def add_transition_data(
        self: "PGBuffer",
        state: np.ndarray,
        action: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        last_value: float = 0.0,
    ) -> None:
        self.states[self.top_index] = state
        self.actions[self.top_index] = action
        self.values[self.top_index] = value
        self.rewards[self.top_index] = reward

        value = np.append(value, last_value)
        self.advantages[
            self.top_index
        ] = self._get_episode_generalized_advantage_estimates(reward, value)

        self.rewards_to_go[self.top_index] = np.cumsum(reward[::-1])[::-1]

        self.top_index += 1

    def get_data(
        self: "PGBuffer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        return (
            self.states,
            self.actions,
            self.rewards,
            self.advantages,
            self.rewards_to_go,
        )

    def reset(self: "PGBuffer") -> None:
        self.top_index = 0
        (
            self.states,
            self.actions,
            self.values,
            self.rewards,
            self.advantages,
            self.rewards_to_go,
        ) = self._create_empty_buffers()
