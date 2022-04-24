from typing import List, Union

import gym
import numpy as np
import torch
import torch.nn as nn

from utilities.buffer.DQNBuffer import DQNBuffer
from utilities.config import Config
from utilities.nn import create_nn
from utilities.types import AdamOptimizer, NNParameters


class DQN:
    def __init__(self: "DQN", config: Config) -> None:
        self.config = config
        self.environment: gym.Env[np.ndarray, Union[int, np.ndarray]] = gym.make(
            self.config.environment_name
        )
        self.episode_length: int = self.config.hyperparameters["DQN"]["episode_length"]
        self.gamma: float = self.config.hyperparameters["DQN"]["discount_rate"]
        q_net_parameters: NNParameters = self.config.hyperparameters["VPG"][
            "q_net_parameters"
        ]
        self.q_net: nn.Sequential = create_nn(
            q_net_parameters["sizes"],
            q_net_parameters["activations"],
        )
        self.q_net_optimizer: AdamOptimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=self.config.hyperparameters["DQN"]["q_net_learning_rate"],
        )
        self.replayBuffer = DQNBuffer(config)
        self.exploration_rate = self.config.hyperparameters["DQN"]["exploration_rate"]

    def train(self: "DQN") -> List[float]:
        def update_q_network() -> None:
            states, actions, rewards, next_states, done = self.replayBuffer.get_transition_data()
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)
            next_state_q_values = self.q_net(next_states)
            best_next_state_q_values = torch.max(next_state_q_values, dim=1)
            q_value_targets = rewards + done * self.gamma * best_next_state_q_values
            actual_q_values = self.q_net(states).gather(1, actions.unsqueeze(-1).type(torch.int64)).squeeze(-1)

        obs = self.environment.reset()
        for _step in range(self.config.training_steps_per_epoch):
            action = self._get_action(torch.tensor(obs, dtype=torch.float32))
            next_obs, reward, done, info = self.environment.step(action)
            self.replayBuffer.add_transition(obs, action, reward, next_obs, done)

            if (
                self.replayBuffer.get_number_of_stored_transitions()
                >= self.config.hyperparameters["DQN"]["minibatch_size"]
            ):


    def _get_action(self: "DQN", obs: torch.Tensor) -> np.ndarray:
        explore = np.random.binomial(1, p=self.exploration_rate)
        if explore:
            action = self.environment.action_space.sample()
        else:
            action = self._get_best_action(obs)
        return action

    def _get_best_action(self: "DQN", obs: torch.Tensor) -> np.ndarray:
        q_value_tensor = self.q_net.forward(obs)
        action = torch.argmax(q_value_tensor)
        return np.array([action])
