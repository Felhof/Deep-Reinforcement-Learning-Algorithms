from typing import Union

import gym
import numpy as np
import torch
import torch.nn as nn
from utilities.buffer.DQNBuffer import DQNBuffer
from utilities.nn import create_nn
from utilities.types import AdamOptimizer, NNParameters


class DQN:
    def __init__(self: "DQN", **kwargs) -> None:
        self.config = kwargs["config"]
        self.environment: gym.Env[np.ndarray, Union[int, np.ndarray]] = gym.make(
            self.config.environment_name
        )
        self.episode_length: int = self.config.episode_length
        self.gamma: float = self.config.hyperparameters["DQN"]["discount_rate"]
        q_net_parameters: NNParameters = self.config.hyperparameters["DQN"][
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
        self.replayBuffer = DQNBuffer(self.config)
        self.exploration_rate = self.config.hyperparameters["DQN"][
            "initial_exploration_rate"
        ]
        self.random_episodes = self.config.hyperparameters["DQN"]["random_episodes"]
        self.gradient_clipping_norm = self.config.hyperparameters["DQN"][
            "gradient_clipping_norm"
        ]
        self.exploration_rate_divisor = 2
        self.result_storage = kwargs["result_storage"]

    def train(self: "DQN"):
        def update_q_network() -> None:
            data = self.replayBuffer.get_transition_data()
            states = torch.tensor(data["states"], dtype=torch.float32)
            actions = torch.tensor(data["actions"], dtype=torch.float32)
            rewards = torch.tensor(data["rewards"], dtype=torch.float32)
            next_states = torch.tensor(data["next_states"], dtype=torch.float32)
            done = torch.tensor(data["done"], dtype=torch.float32)

            with torch.no_grad():
                next_state_q_values = self.q_net(next_states)
                best_next_state_q_values = torch.max(next_state_q_values, dim=1).values
                q_value_targets = (
                    rewards + (1 - done) * self.gamma * best_next_state_q_values
                )
            actual_q_values = (
                self.q_net(states)
                .gather(1, actions.unsqueeze(-1).type(torch.int64))
                .squeeze(-1)
            )

            self.q_net_optimizer.zero_grad()
            q_loss = nn.MSELoss()(q_value_targets, actual_q_values)
            q_loss.backward()
            if self.gradient_clipping_norm is not None:
                torch.nn.utils.clip_grad_norm(
                    self.q_net.parameters(), self.gradient_clipping_norm
                )
            self.q_net_optimizer.step()

        for episode in range(self.config.training_steps_per_epoch):
            episode_reward = 0
            obs = self.environment.reset()
            for _step in range(self.config.episode_length):
                action = self._get_action(torch.tensor(obs, dtype=torch.float32))
                next_obs, reward, done, info = self.environment.step(action)
                reward /= self.config.episode_length
                self.replayBuffer.add_transition(obs, action, reward, next_obs, done)
                episode_reward += reward
                obs = next_obs
                learning = (
                    self.replayBuffer.get_number_of_stored_transitions()
                    >= self.config.hyperparameters["DQN"]["minibatch_size"]
                )
                if learning:
                    update_q_network()
                if done or _step == self.config.episode_length - 1:
                    if episode > self.random_episodes and learning:
                        self.exploration_rate = 1 / self.exploration_rate_divisor
                        self.exploration_rate_divisor += 1
                    break
            self.result_storage.add_average_training_step_reward(
                episode_reward * self.config.episode_length
            )
        self.result_storage.end_epoch()

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
        return np.array(action)
