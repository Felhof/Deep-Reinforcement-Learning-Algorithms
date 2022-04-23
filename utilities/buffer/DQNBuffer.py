from collections import deque
from typing import Deque, Tuple

import numpy as np

from utilities.config import Config


class DQNBuffer:
    def __init__(self: "DQNBuffer", config: Config, buffer_size: int) -> None:
        self.minibatch_size = config.hyperparameters["DQN"]["minibatch_size"]
        self.buffer_size = buffer_size
        self.states: Deque[np.ndarray] = deque()
        self.actions: Deque[np.ndarray] = deque()
        self.rewards: Deque[float] = deque()
        self.next_states: Deque[np.ndarray] = deque()

    def add_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        if len(self.states) >= self.buffer_size:
            self.states.popleft()
            self.actions.popleft()
            self.rewards.popleft()
            self.next_states.popleft()
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def get_transition_data(
        self: "DQNBuffer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        transition_indices = np.random.choice(
            np.arange(len(self.states)), size=self.minibatch_size
        )
        return (
            np.array(self.states[transition_indices]),
            np.array(self.actions[transition_indices]),
            np.array(self.rewards[transition_indices]),
            np.array(self.next_states[transition_indices]),
        )
