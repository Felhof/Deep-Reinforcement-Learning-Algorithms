from abc import ABC, abstractmethod

import numpy as np
import torch
from utilities.progress_logging import ProgressLogger


class BaseAgent(ABC):
    def __init__(self: "BaseAgent", **kwargs) -> None:
        self.config = kwargs["config"]
        self.environment = kwargs["environment"]
        self.episode_length: int = self.config.episode_length
        self.logger = ProgressLogger(
            level=self.config.log_level,
            filename=self.config.log_filename,  # log_to_console=False
            directory=self.config.log_directory,
        )
        self.result_storage = kwargs["result_storage"]
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
