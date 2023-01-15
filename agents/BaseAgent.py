from abc import ABC, abstractmethod

from utilities.progress_logging import ProgressLogger


class BaseAgent(ABC):
    def __init__(self: "BaseAgent", **kwargs) -> None:
        self.config = kwargs["config"]
        self.environment = kwargs["environment"]
        self.episode_length: int = self.config.episode_length
        self.gamma: float = self.config.hyperparameters["DQN"]["discount_rate"]
        self.logger = ProgressLogger(
            level=self.config.log_level,
            filename=self.config.log_filename,  # log_to_console=False
            directory=self.config.log_directory,
        )
        self.result_storage = kwargs["result_storage"]

    @abstractmethod
    def train(self: "BaseAgent") -> None:
        pass
