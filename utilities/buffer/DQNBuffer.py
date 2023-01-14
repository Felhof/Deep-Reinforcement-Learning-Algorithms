from typing import Dict

import numpy as np
from utilities.config import Config
from utilities.types import ObservationDim
from utilities.utils import get_dimension_format_string


class DQNBuffer:
    def __init__(
        self: "DQNBuffer", config: Config, observation_dim: ObservationDim = 2
    ) -> None:
        self.minibatch_size = config.hyperparameters["DQN"]["minibatch_size"]
        self.buffer_size = config.hyperparameters["DQN"]["buffer_size"]
        self.states = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(observation_dim),
        )
        self.actions = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(1),
        )
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_states = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(observation_dim),
        )
        self.done = np.zeros(self.buffer_size, dtype=bool)
        self.index = 0
        self.number_of_stored_transitions = 0

    def add_transition(
        self: "DQNBuffer",
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

    def get_number_of_stored_transitions(self: "DQNBuffer") -> int:
        return self.number_of_stored_transitions

    def get_transition_data(
        self: "DQNBuffer",
    ) -> Dict[str, np.ndarray]:
        transition_indices = np.random.choice(
            np.arange(self.number_of_stored_transitions), size=self.minibatch_size
        )
        return dict(
            states=self.states[transition_indices],
            actions=self.actions[transition_indices],
            rewards=self.rewards[transition_indices],
            next_states=self.next_states[transition_indices],
            done=self.done[transition_indices],
        )
