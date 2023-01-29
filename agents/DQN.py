import numpy as np
import torch
import torch.nn as nn

from agents.QLearningAgent import QLearningAgent
from utilities.nn import create_q_net, soft_update_nn
from utilities.types import AdamOptimizer, NNParameters


class DQN(QLearningAgent):
    def __init__(self: "DQN", **kwargs) -> None:
        super().__init__(key="DQN", **kwargs)
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
        self.target_q_net: nn.Sequential = create_q_net(
            observation_dim=self.environment.observation_dim,
            number_of_actions=self.environment.number_of_actions,
            parameters=q_net_parameters,
        )
        self._soft_update_target_networks(tau=1.0)
        self.exploration_rate = self.config.hyperparameters["DQN"][
            "initial_exploration_rate"
        ]
        self.gradient_clipping_norm = self.config.hyperparameters["DQN"][
            "gradient_clipping_norm"
        ]
        self.initial_exploration_rate = self.config.hyperparameters["DQN"][
            "initial_exploration_rate"
        ]
        self.final_exploration_rate = self.config.hyperparameters["DQN"][
            "final_exploration_rate"
        ]
        self.exploration_rate_annealing_period = self.config.hyperparameters["DQN"][
            "exploration_rate_annealing_period"
        ]
        self.tau = self.config.hyperparameters["DQN"][
            "soft_update_interpolation_factor"
        ]

    def evaluate(self: "DQN", time_to_save: bool = False) -> float:
        self.q_net.eval()
        reward = super().evaluate(time_to_save=time_to_save)
        self.q_net.train()
        return reward

    def load(self: "DQN", filename) -> None:
        self.q_net.load_state_dict(torch.load(f"{filename}_q_net.pt"))
        if torch.cuda.is_available():
            self.q_net = self.q_net.cuda()

    def _update(self: "DQN") -> None:
        data = self.replay_buffer.get_transition_data()
        states = torch.tensor(data["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(data["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            data["next_states"], dtype=torch.float32, device=self.device
        )
        done = torch.tensor(data["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_state_q_values = self.target_q_net(next_states)
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
        self._soft_update_target_networks(tau=self.tau)

    def _training_loop(self: "DQN") -> None:
        super()._training_loop()
        can_learn = self._can_learn()
        is_exploration_step = self._is_exploration_step()
        if can_learn and not is_exploration_step:
            exploration_ratio = min(
                (self.current_timestep - self.pure_exploration_steps)
                / self.exploration_rate_annealing_period,
                1.0,
            )
            self.exploration_rate = (
                (1.0 - exploration_ratio) * self.initial_exploration_rate
                + exploration_ratio * self.final_exploration_rate
            )

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
        return np.array(action.cpu())

    def _soft_update_target_networks(self: "DQN", tau: float = 0.01) -> None:
        soft_update_nn(self.target_q_net, self.q_net, tau)
