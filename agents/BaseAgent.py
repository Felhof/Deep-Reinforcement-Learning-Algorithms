from abc import ABC, abstractmethod

import numpy as np
import torch
from utilities.environments import BaseEnvironmentWrapper
from utilities.progress_logging import ProgressLogger


class BaseAgent(ABC):
    def __init__(self: "BaseAgent", **kwargs) -> None:
        self.config = kwargs["config"]
        assert not (
            self.config.use_cuda and not torch.cuda.is_available()
        ), "Cannot use cuda as it is not available on this machine"
        self.device = "cuda:0" if self.config.use_cuda else "cpu"
        self.environment: BaseEnvironmentWrapper = kwargs["environment"]
        self.episode_length: int = self.config.episode_length
        self.dtype_name = self.config.dtype_name
        if self.dtype_name == "float64":
            self.tensor_type = torch.float64
            torch.set_default_tensor_type("torch.DoubleTensor")
        else:
            self.tensor_type = torch.float32
            torch.set_default_tensor_type(torch.FloatTensor)
        self.logger = ProgressLogger(
            level=self.config.log_level,
            filename=self.config.log_filename,  # log_to_console=False
            directory=self.config.log_directory,
        )
        self.result_storage = kwargs["result_storage"]
        if self.config.save:
            self.model_saver = kwargs["model_saver"]
        self.max_timestep = self.config.train_for_n_environment_steps
        self.current_timestep = 0

    @abstractmethod
    def _training_loop(self: "BaseAgent") -> None:
        pass

    @abstractmethod
    def get_best_action(self: "BaseAgent", obs: torch.Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def load(self: "BaseAgent", filename) -> None:
        pass

    def _train_for_n_training_steps(self: "BaseAgent") -> None:
        for training_step in range(self.config.training_steps_per_epoch):
            self.logger.info(f"Training step {training_step}.")
            self._training_loop()
            if training_step % self.config.evaluate_every_n_training_steps == 0:
                self.evaluate(
                    time_to_save=training_step % self.config.save_every_n_training_steps
                    == 0
                )

    def _train_until_timestep_limit_reached(self: "BaseAgent") -> None:
        training_step_counter = 0
        while True:
            training_step_counter += 1
            self.logger.info(f"Training step {training_step_counter}.")
            self._training_loop()
            if self.has_reached_timestep_limit():
                self.logger.info(
                    "Stopping training as maximum number of training steps has been "
                    "reached."
                )
                break

    def evaluate(self: "BaseAgent", time_to_save: bool = False) -> float:
        self.logger.info("Evaluate agent.")
        with torch.no_grad():
            env = self.environment
            obs, _ = env.true_reset()
            total_reward: float = 0.0
            for step in range(self.episode_length):
                action = self.get_best_action(
                    torch.tensor(
                        np.array(obs), dtype=self.tensor_type, device=self.device
                    )
                )
                next_obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if env.is_really_done:
                    self.logger.info(
                        f"During evaluation the policy survives for {step + 1} frames."
                    )
                    break
                elif terminated:
                    next_obs, info = env.reset()
                obs = next_obs
            if time_to_save and self.config.save:
                self.model_saver.save_model_if_best(self, total_reward)
            self.result_storage.add_average_training_step_reward(total_reward)
        self.logger.info(
            f"During evaluation the policy achieves a score of {total_reward}."
        )
        return total_reward

    def has_reached_timestep_limit(self: "BaseAgent") -> bool:
        return self.max_timestep != 0 and self.current_timestep >= self.max_timestep

    def train(self: "BaseAgent") -> None:
        if self.config.training_steps_per_epoch > 0:
            self._train_for_n_training_steps()
        else:
            self._train_until_timestep_limit_reached()

        self.logger.log_table(scope="epoch", level="INFO")
        self.logger.clear(scope="epoch")
        self.logger.clear_handlers()
