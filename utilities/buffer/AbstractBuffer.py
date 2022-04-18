from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from utilities.config import Config
from utilities.utils import get_dimension_format_string


class AbstractBuffer(ABC):
    def __init__(self: "AbstractBuffer", config: Config, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.config = config
        (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.done,
        ) = self.reset_buffer()
        self.top_index = 0

    @abstractmethod
    def add_step_data(
        self: "AbstractBuffer",
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def get_data(
        self: "AbstractBuffer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        pass

    def reset_buffer(
        self: "AbstractBuffer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        states = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(1, self.config.observation_dim),
        )
        actions = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(1, self.config.action_dim),
        )
        rewards = np.zeros(self.buffer_size, dtype=np.float32)
        next_states = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(1, self.config.observation_dim),
        )
        done = np.zeros(self.buffer_size, dtype="bool")

        return states, actions, rewards, next_states, done
