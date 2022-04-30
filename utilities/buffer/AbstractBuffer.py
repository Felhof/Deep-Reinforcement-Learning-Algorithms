from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from utilities.config import Config
from utilities.utils import get_dimension_format_string


class AbstractBuffer(ABC):
    def __init__(self: "AbstractBuffer", config: Config, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.config = config
        self.top_index = 0
        (
            self.states,
            self.actions,
            self.values,
            self.rewards,
            self.done,
        ) = self.reset_buffer()

    @abstractmethod
    def add_step_data(
        self: "AbstractBuffer",
        action: np.ndarray,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def get_data(
        self: "AbstractBuffer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        pass

    def reset_buffer(
        self: "AbstractBuffer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.top_index = 0
        states = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(1, self.config.observation_dim),
        )
        actions = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(1, self.config.action_dim),
        )
        values = np.zeros(
            self.buffer_size,
            dtype=np.float32,
        )
        rewards = np.zeros(self.buffer_size, dtype=np.float32)
        done = np.zeros(self.buffer_size, dtype="bool")

        return states, actions, values, rewards, done
