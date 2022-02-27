from dataclasses import dataclass

from utilities.types import HyperParameters  # type: ignore


@dataclass
class Config:
    environment_name: str = ""
    observation_dim: int = 1
    action_dim: int = 1
    number_of_actions: int = 1
    hyperparameters: HyperParameters = None
    training_steps_per_epoch: int = 0
    epochs: int = 0
