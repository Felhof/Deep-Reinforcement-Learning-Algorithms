from abc import ABC, abstractmethod

import numpy as np
import torch
from utilities.progress_logging import ProgressLogger


class BaseAgent(ABC):
    def __init__(self: "BaseAgent", **kwargs) -> None:
        self.config = kwargs["config"]
        self.device = (
            "cuda:0" if torch.cuda.is_available() and self.config.use_cuda else "cpu"
        )
        self.environment = kwargs["environment"]
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

    @abstractmethod
    def get_best_action(self: "BaseAgent", obs: torch.Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def train(self: "BaseAgent") -> None:
        pass

    @abstractmethod
    def load(self: "BaseAgent", filename) -> None:
        pass

    def evaluate(self: "BaseAgent", time_to_save: bool = False) -> float:
        with torch.no_grad():
            obs, _ = self.environment.reset()
            total_reward: float = 0.0
            for step in range(self.episode_length):
                action = self.get_best_action(
                    torch.tensor(obs, dtype=self.tensor_type, device=self.device)
                )
                next_obs, reward, terminated, truncated, info = self.environment.step(
                    action
                )
                total_reward += reward
                if terminated or truncated:
                    break
                obs = next_obs
            if time_to_save and self.config.save:
                self.model_saver.save_model_if_best(self, total_reward)
            self.result_storage.add_average_training_step_reward(total_reward)
        return total_reward
