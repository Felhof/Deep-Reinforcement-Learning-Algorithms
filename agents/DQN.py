from agents.BaseAgent import BaseAgent
import numpy as np
import torch
import torch.nn as nn
from utilities.buffer.DQNBuffer import DQNBuffer
from utilities.nn import create_q_net
from utilities.types import AdamOptimizer, NNParameters


class DQN(BaseAgent):
    def __init__(self: "DQN", **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma: float = self.config.hyperparameters["DQN"]["discount_rate"]
        q_net_parameters: NNParameters = self.config.hyperparameters["DQN"][
            "q_net_parameters"
        ]
        q_net_parameters["device"] = self.device
        self.q_net: nn.Sequential = create_q_net(
            observation_dim=self.environment.observation_dim,
            number_of_actions=self.environment.number_of_actions,
            parameters=q_net_parameters,
        )
        self.q_net_optimizer: AdamOptimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=self.config.hyperparameters["DQN"]["q_net_learning_rate"],
        )
        self.replay_buffer = DQNBuffer(
            minibatch_size=self.config.hyperparameters["DQN"]["minibatch_size"],
            buffer_size=self.config.hyperparameters["DQN"]["buffer_size"],
            observation_dim=self.environment.observation_dim,
        )
        self.exploration_rate = self.config.hyperparameters["DQN"][
            "initial_exploration_rate"
        ]
        self.pure_exploration_steps = self.config.hyperparameters["DQN"].get(
            "pure_exploration_steps", 0
        )
        self.gradient_clipping_norm = self.config.hyperparameters["DQN"][
            "gradient_clipping_norm"
        ]
        self.exploration_rate_divisor = 2

    def evaluate(self: "DQN", time_to_save: bool = False) -> float:
        self.q_net.eval()
        reward = super().evaluate(time_to_save=time_to_save)
        self.q_net.train()
        return reward

    def load(self: "DQN", filename) -> None:
        self.q_net.load_state_dict(torch.load(f"{filename}_q_net.pt"))
        if torch.cuda.is_available():
            self.q_net = self.q_net.cuda()

    def _update_q_network(self: "DQN") -> None:
        data = self.replay_buffer.get_transition_data()
        states = torch.tensor(data["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(data["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            data["next_states"], dtype=torch.float32, device=self.device
        )
        done = torch.tensor(data["done"], dtype=torch.float32, device=self.device)

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

    def _training_loop(self: "DQN") -> None:
        episode_reward: float = 0
        obs, _ = self.environment.reset()
        for step in range(self.config.episode_length):
            action = self._get_action(
                torch.tensor(obs, dtype=torch.float32, device=self.device)
            )
            next_obs, reward, terminated, truncated, info = self.environment.step(
                action
            )
            reward /= self.config.episode_length
            self.replay_buffer.add_transition(
                obs, action, float(reward), next_obs, terminated or truncated
            )
            episode_reward += float(reward)
            obs = next_obs
            can_learn = (
                self.replay_buffer.get_number_of_stored_transitions()
                >= self.config.hyperparameters["DQN"]["minibatch_size"]
            )
            is_exploration_step = self.current_timestep <= self.pure_exploration_steps
            if can_learn and not is_exploration_step:
                self._update_q_network()
            self.current_timestep += 1
            if self.max_timestep != -1 and self.current_timestep >= self.max_timestep:
                break
            if terminated or truncated or step == self.config.episode_length - 1:
                if can_learn and not is_exploration_step:
                    self.exploration_rate = 1 / self.exploration_rate_divisor
                    self.exploration_rate_divisor += 1
                break

    def _get_action(self: "DQN", obs: torch.Tensor) -> np.ndarray:
        explore = np.random.binomial(1, p=self.exploration_rate)
        if explore:
            action = self.environment.action_space.sample()
        else:
            action = self.get_best_action(obs)
        return action

    def get_best_action(self: "DQN", obs: torch.Tensor) -> np.ndarray:
        q_value_tensor = self.q_net.forward(obs)
        action = torch.argmax(q_value_tensor)
        return np.array(action)
