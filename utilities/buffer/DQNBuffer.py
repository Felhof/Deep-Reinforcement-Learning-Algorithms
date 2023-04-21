from typing import Dict, List

import numpy as np
from utilities.types.environment import Observation
from utilities.utils import get_dimension_format_string


class DQNBuffer:
    def __init__(
        self: "DQNBuffer",
        minibatch_size: int = 256,
        buffer_size: int = 40000,
    ) -> None:
        self.minibatch_size = minibatch_size
        self.buffer_size = buffer_size
        self.states: List[Observation] = []
        self.actions = np.zeros(
            self.buffer_size,
            dtype=get_dimension_format_string(1),
        )
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_states: List[Observation] = []
        self.done = np.zeros(self.buffer_size, dtype=bool)
        self.index = 0
        self.number_of_stored_transitions = 0
        self.min_reward = 10e4
        self.max_reward = -10e4

    def add_transition(
        self: "DQNBuffer",
        state: Observation,
        action: np.ndarray,
        reward: float,
        next_state: Observation,
        done: bool,
    ) -> None:
        if reward < self.min_reward:
            self.min_reward = reward
        if reward > self.max_reward:
            self.max_reward = reward
        if self.number_of_stored_transitions < self.buffer_size:
            self.states.append(state)
        else:
            self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        if self.number_of_stored_transitions < self.buffer_size:
            self.next_states.append(next_state)
        else:
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
            np.arange(self.number_of_stored_transitions),
            size=self.minibatch_size,
        )
        scaled_rewards = (
            self.rewards[transition_indices] / sum(self.rewards[transition_indices])
            if sum(self.rewards[transition_indices]) != 0
            else self.rewards[transition_indices]
        )
        return dict(
            states=np.array(
                [
                    np.array(self.states[transition_index])
                    for transition_index in transition_indices
                ]
            ),
            actions=self.actions[transition_indices],
            rewards=scaled_rewards,
            next_states=np.array(
                [
                    np.array(self.next_states[transition_index])
                    for transition_index in transition_indices
                ]
            ),
            done=self.done[transition_indices],
        )
