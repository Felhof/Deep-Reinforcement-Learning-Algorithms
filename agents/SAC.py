from typing import cast, Tuple

from agents.BaseAgent import BaseAgent
from agents.Policy import CategoricalPolicy, create_policy
import numpy as np
from torch import nn
import torch.optim
from utilities.buffer.DQNBuffer import DQNBuffer
from utilities.nn import create_q_net
from utilities.types import AdamOptimizer, PolicyParameters


def _soft_update(target_model, origin_model, tau):
    for target_param, local_param in zip(
        target_model.parameters(), origin_model.parameters()
    ):
        target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


class SAC(BaseAgent):
    def __init__(self: "SAC", **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma: float = self.config.hyperparameters["SAC"]["discount_rate"]

        actor_parameters: PolicyParameters = {
            "action_type": self.environment.action_type,
            "number_of_actions": self.environment.number_of_actions,
            "observation_dim": self.environment.observation_dim,
            "policy_net_parameters": self.config.hyperparameters["SAC"][
                "actor_parameters"
            ]
            | {"device": self.device},
        }
        self.actor: CategoricalPolicy = cast(
            CategoricalPolicy, create_policy(actor_parameters)
        )

        critic_parameters = self.config.hyperparameters["SAC"]["critic_parameters"] | {
            "device": self.device
        }
        self.critic1: nn.Sequential = create_q_net(
            observation_dim=self.environment.observation_dim,
            number_of_actions=self.environment.number_of_actions,
            parameters=critic_parameters,
        )
        self.critic2: nn.Sequential = create_q_net(
            observation_dim=self.environment.observation_dim,
            number_of_actions=self.environment.number_of_actions,
            parameters=critic_parameters,
        )
        self.critic1_optimizer: AdamOptimizer = torch.optim.Adam(
            self.critic1.parameters(),
            lr=critic_parameters["learning_rate"],
        )
        self.critic2_optimizer: AdamOptimizer = torch.optim.Adam(
            self.critic2.parameters(),
            lr=critic_parameters["learning_rate"],
        )

        self.critic_target1: nn.Sequential = create_q_net(
            observation_dim=self.environment.observation_dim,
            number_of_actions=self.environment.number_of_actions,
            parameters=critic_parameters,
        )
        self.critic_target2: nn.Sequential = create_q_net(
            observation_dim=self.environment.observation_dim,
            number_of_actions=self.environment.number_of_actions,
            parameters=critic_parameters,
        )
        self._soft_update_target_networks(tau=1.0)

        self.replay_buffer = DQNBuffer(self.config, self.environment.observation_dim)

        self.target_entropy = 0.98 * -np.log(1 / self.environment.action_space.n)
        self.log_alpha = torch.tensor(
            np.log(self.config.hyperparameters["SAC"]["initial_temperature"]),
            requires_grad=True,
        )
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.config.hyperparameters["SAC"]["temperature_learning_rate"],
        )
        self.tau = self.config.hyperparameters["SAC"][
            "soft_update_interpolation_factor"
        ]

    def _actor_loss(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs, log_action_probs = self.actor.get_action_probs(obs)
        q_values_local = self.critic1(obs)
        q_values_local2 = self.critic2(obs)
        inside_term = self.alpha * log_action_probs - torch.min(
            q_values_local, q_values_local2
        )
        policy_loss = (action_probs * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probs

    def _critic_loss(
        self: "SAC",
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards,
        next_states: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            action_probs, log_action_probs = self.actor.get_action_probs(next_states)
            next_q_values_target = self.critic_target1.forward(next_states)
            next_q_values_target2 = self.critic_target2.forward(next_states)
            soft_state_values = (
                action_probs
                * (
                    torch.min(next_q_values_target, next_q_values_target2)
                    - self.alpha * log_action_probs
                )
            ).sum(dim=1)

            next_q_values = rewards + ~done * self.gamma * soft_state_values

        soft_q_values = (
            self.critic1(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )
        soft_q_values2 = (
            self.critic2(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )
        critic_square_error = torch.nn.MSELoss(reduction="none")(
            soft_q_values, next_q_values
        )
        critic2_square_error = torch.nn.MSELoss(reduction="none")(
            soft_q_values2, next_q_values
        )
        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def _get_action(self: "SAC", obs: torch.Tensor) -> np.ndarray:
        return self.actor.get_action(obs).numpy()

    def _soft_update_target_networks(self: "SAC", tau=0.01):
        _soft_update(self.critic_target1, self.critic1, tau)
        _soft_update(self.critic_target2, self.critic2, tau)

    def _temperature_loss(self, log_action_probabilities: torch.Tensor) -> torch.Tensor:
        return -(
            self.log_alpha * (log_action_probabilities + self.target_entropy).detach()
        ).mean()

    def _update(self: "SAC") -> None:
        data = self.replay_buffer.get_transition_data()
        states = torch.tensor(data["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(data["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            data["next_states"], dtype=torch.float32, device=self.device
        )
        done = torch.tensor(data["done"], dtype=torch.float32, device=self.device)

        critic_loss, critic2_loss = self._critic_loss(
            states, actions, rewards, next_states, done
        )

        critic_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        actor_loss, log_action_probs = self._actor_loss(states)

        actor_loss.backward()
        self.actor.update()

        temperature_loss = self._temperature_loss(log_action_probs)

        temperature_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self._soft_update_target_networks(tau=self.tau)

    def evaluate(self: "SAC", time_to_save: bool = False) -> float:
        self.actor.policy_net.eval()
        reward = super().evaluate(time_to_save=time_to_save)
        self.actor.policy_net.train()
        return reward

    def get_best_action(self: "SAC", obs: torch.Tensor) -> np.ndarray:
        return self.actor.get_best_action(obs).numpy()

    def train(self: "SAC") -> None:
        for episode in range(self.config.training_steps_per_epoch):
            obs, _ = self.environment.reset()
            for _step in range(self.config.episode_length):
                self.critic1.zero_grad()
                self.critic2.zero_grad()
                self.actor.reset_gradients()
                self.alpha_optimizer.zero_grad()

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
                obs = next_obs

                can_learn = (
                    self.replay_buffer.get_number_of_stored_transitions()
                    >= self.config.hyperparameters["SAC"]["minibatch_size"]
                )
                if can_learn:
                    self._update()

            if episode % self.config.evaluation_interval == 0:
                evaluation_result = self.evaluate(
                    time_to_save=episode % self.config.save_interval == 0
                )
                self.logger.info(
                    f"During evaluation the policy achieves a score of {evaluation_result}"
                )

    def load(self: "SAC", filename: str) -> None:
        self.actor.policy_net.load_state_dict(torch.load(f"{filename}_actor.pt"))
        self.critic1.load_state_dict(torch.load(f"{filename}_critic1.pt"))
        self.critic2.load_state_dict(torch.load(f"{filename}_critic2.pt"))
        self.critic_target1.load_state_dict(torch.load(f"{filename}_critic_target1.pt"))
        self.critic_target2.load_state_dict(torch.load(f"{filename}_critic_target2.pt"))
        if torch.cuda.is_available() and self.config.device != "cpu":
            self.actor.policy_net = self.actor.policy_net.cuda()
            self.critic1 = self.critic1.cuda()
            self.critic2 = self.critic2.cuda()
            self.critic_target1 = self.critic_target1.cuda()
            self.critic_target2 = self.critic_target2.cuda()
