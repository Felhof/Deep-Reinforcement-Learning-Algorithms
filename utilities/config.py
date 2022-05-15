from dataclasses import dataclass, field

from utilities.types import HyperParameters


@dataclass
class Config:
    environment_name: str = ""
    observation_dim: int = 1
    action_dim: int = 1
    number_of_actions: int = 1
    hyperparameters: HyperParameters = field(default_factory=HyperParameters)
    training_steps_per_epoch: int = 0
    episode_length: int = 0
    epochs: int = 0
    target_score: int = 200
    plot_name: str = ""
