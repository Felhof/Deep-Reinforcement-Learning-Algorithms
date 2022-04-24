from typing import Tuple

import numpy as np

from utilities.config import Config
from utilities.utils import get_dimension_format_string


class DQNBuffer:
    def __init__(self: "DQNBuffer", config: Config) -> None:
        self.minibatch_size = config.hyperparameters["DQN"]["minibatch_size"]
        self.buffer_size = config.hyperparameters["DQN"]["buffer_size"]
        self.states = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(config.observation_dim),
        )
        self.actions = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(config.action_dim),
        )
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_states = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(config.observation_dim),
        )
        self.done = np.zeros(self.buffer_size, dtype=bool)
        self.index = 0
        self.number_of_stored_transitions = 0

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.done[self.index] = done
        self.index = (self.index + 1) % self.buffer_size
        self.number_of_stored_transitions = min(
            self.number_of_stored_transitions + 1, self.buffer_size
        )

    def get_number_of_stored_transitions(self) -> int:
        return self.number_of_stored_transitions

    def get_transition_data(
        self: "DQNBuffer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        transition_indices = np.random.choice(
            np.arange(len(self.states)), size=self.minibatch_size
        )
        return (
            self.states[transition_indices],
            self.actions[transition_indices],
            self.rewards[transition_indices],
            self.next_states[transition_indices],
            self.done[transition_indices],
        )
