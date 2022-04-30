from typing import Tuple

import numpy as np
from utilities.config import Config
from utilities.utils import get_dimension_format_string


class PGBuffer:
    def __init__(self: "PGBuffer", config: Config, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.config = config
        self.current_episode_start_index = 0
        self.top_index = 0
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
        episode_duration = self.top_index - self.current_episode_start_index
        gamma: float = self.config.hyperparameters["VPG"]["discount_rate"]
        lamda: float = self.config.hyperparameters["VPG"][
            "generalized_advantage_estimate_exponential_mean_discount_rate"
        ]
        delta: np.ndarray = rewards[:-1] + gamma * values[1:] - values[:-1]

        advantage = np.zeros(episode_duration)

        for t in reversed(range(episode_duration)):
            next_advantage = advantage[t + 1] if t < episode_duration - 1 else 0
            advantage[t] = delta[t] + lamda * gamma * next_advantage

        return advantage

    def _create_empty_buffers(
        self: "PGBuffer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        states = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(self.config.observation_dim),
        )
        actions = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(self.config.action_dim),
        )
        values = np.zeros(
            self.buffer_size,
            dtype=np.float32,
        )
        rewards = np.zeros(self.buffer_size, dtype=np.float32)
        advantages = np.zeros(self.buffer_size, dtype=np.float32)
        rewards_to_go = np.zeros(self.buffer_size, dtype=np.float32)

        return states, actions, values, rewards, advantages, rewards_to_go

    def add_transition_data(
        self: "PGBuffer",
        state: np.ndarray,
        action: np.ndarray,
        value: float,
        reward: float,
    ) -> None:
        self.states[self.top_index] = state
        self.actions[self.top_index] = action
        self.values[self.top_index] = value
        self.rewards[self.top_index] = reward
        self.top_index += 1

    def end_episode(self: "PGBuffer", last_value: float = 0) -> None:
        episode_slice = slice(self.current_episode_start_index, self.top_index)
        rewards = np.append(self.rewards[episode_slice], last_value)
        values = np.append(self.values[episode_slice], last_value)
        self.advantages[
            episode_slice
        ] = self._get_episode_generalized_advantage_estimates(rewards, values)
        self.rewards_to_go[episode_slice] = np.cumsum(rewards[::-1])[::-1][:-1]
        self.current_episode_start_index = self.top_index

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
        self.current_episode_start_index = 0
        self.top_index = 0
        (
            self.states,
            self.actions,
            self.values,
            self.rewards,
            self.advantages,
            self.rewards_to_go,
        ) = self._create_empty_buffers()
