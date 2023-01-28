import abc
from abc import ABC

from agents.BaseAgent import BaseAgent
import numpy as np
import torch
from utilities.buffer.DQNBuffer import DQNBuffer


class QLearningAgent(BaseAgent, ABC):
    def __init__(self: "QLearningAgent", key: str = "DQN", **kwargs) -> None:
        super().__init__(**kwargs)
        self.key = key

        self.replay_buffer = DQNBuffer(
            minibatch_size=self.config.hyperparameters[self.key]["minibatch_size"],
            buffer_size=self.config.hyperparameters[self.key]["buffer_size"],
        )

        self.pure_exploration_steps = self.config.hyperparameters[self.key].get(
            "pure_exploration_steps", 0
        )

    @abc.abstractmethod
    def _get_action(self: "QLearningAgent", obs: torch.Tensor) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _update(self: "QLearningAgent") -> None:
        pass

    def _can_learn(self: "QLearningAgent") -> bool:
        return (
            self.replay_buffer.get_number_of_stored_transitions()
            >= self.config.hyperparameters[self.key]["minibatch_size"]
        )

    def _evaluate_if_evaluating_during_episodes(self: "QLearningAgent") -> None:
        if self.config.training_steps_per_epoch > 0:
            return
        if self.current_timestep % self.config.evaluate_every_n_timesteps == 0:
            self.evaluate(
                time_to_save=self.current_timestep
                % self.config.save_every_n_training_steps
                == 0
            )

    def _is_exploration_step(self: "QLearningAgent") -> bool:
        return self.current_timestep <= self.pure_exploration_steps

    def _is_time_to_update(self: "QLearningAgent") -> bool:
        return self.current_timestep % self.config.update_model_every_n_timesteps == 0

    def _training_loop(self: "QLearningAgent") -> None:
        self.logger.start_timer(scope="epoch", level="INFO", attribute="episode")
        obs, _ = self.environment.reset()
        for step in range(self.config.episode_length):
            self.current_timestep += 1
            action = self._get_action(
                torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
            )
            next_obs, reward, terminated, truncated, info = self.environment.step(
                action
            )
            self.replay_buffer.add_transition(
                obs, action, float(reward), next_obs, terminated or truncated
            )
            obs = next_obs

            can_learn = self._can_learn()
            is_exploration_step = self._is_exploration_step()
            time_to_update = self._is_time_to_update()
            if can_learn and time_to_update and not is_exploration_step:
                self.logger.info("Updating parameters.")
                self.logger.start_timer(scope="epoch", level="INFO", attribute="update")
                self._update()
                self.logger.stop_timer(scope="epoch", level="INFO", attribute="update")
            self._evaluate_if_evaluating_during_episodes()
            if self.has_reached_timestep_limit():
                break
            if terminated or truncated:
                self.logger.info(
                    f"This episode the agent lasted for {step + 1} frames before losing."
                )
                break
        self.logger.stop_timer(scope="epoch", level="INFO", attribute="episode")
