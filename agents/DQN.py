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
        episode_rewards: List[float] = []

        def update_q_network() -> None:
            data = self.replayBuffer.get_transition_data()
            states = torch.tensor(data["states"], dtype=torch.float32)
            actions = torch.tensor(data["actions"], dtype=torch.float32)
            rewards = torch.tensor(data["rewards"], dtype=torch.float32)
            next_states = torch.tensor(data["next_states"], dtype=torch.float32)
            done = torch.tensor(data["done"], dtype=torch.float32)

            next_state_q_values = self.q_net(next_states)
            best_next_state_q_values = torch.max(next_state_q_values, dim=1)
            q_value_targets = rewards + done * self.gamma * best_next_state_q_values
            actual_q_values = self.q_net(states).gather(1, actions.unsqueeze(-1).type(torch.int64)).squeeze(-1)

            self.q_net_optimizer.zero_grad()
            q_loss = nn.MSELoss()(q_value_targets, actual_q_values)
            q_loss.backward()
            self.q_net_optimizer.step()

        for episode in range(self.config.training_steps_per_epoch):
            episode_reward = 0
            obs = self.environment.reset()
            for _step in range(self.config.episode_length):
                action = self._get_action(torch.tensor(obs, dtype=torch.float32))
                next_obs, reward, done, info = self.environment.step(action)
                self.replayBuffer.add_transition(obs, action, reward, next_obs, done)
                episode_reward += reward
                if (
                    self.replayBuffer.get_number_of_stored_transitions()
                    >= self.config.hyperparameters["DQN"]["minibatch_size"]
                ):
                    update_q_network()
                    self.exploration_rate = self.exploration_rate / (episode + 1)
                if done:
                    break
            episode_rewards.append(episode_reward)
        return episode_rewards

    def _get_action(self: "DQN", obs: torch.Tensor) -> np.ndarray:
        explore = np.random.binomial(1, p=self.exploration_rate)
        if explore:
            action = self.environment.action_space.sample()
        else:
            action = self._get_best_action(obs)
        return action

    def _get_best_action(self: "DQN", obs: torch.Tensor) -> np.ndarray:
        self.q_net.eval()
        with torch.no_grad():
            q_value_tensor = self.q_net.forward(obs)
        action = torch.argmax(q_value_tensor)
        self.q_net.train()
        return np.array([action])
