from dataclasses import dataclass, field

from utilities.types import HyperParameters


@dataclass
class Config:
    environment_name: str = ""
    action_type: str = "Categorical"
    number_of_actions: int = 2
    observation_dim: int = 2
    hyperparameters: HyperParameters = field(default_factory=HyperParameters)
    training_steps_per_epoch: int = 0
    episode_length: int = 0
    epochs: int = 0
    target_score: int = 200
    plot_name: str = ""
    log_level: str = "WARN"
    log_filename: str = ""
    results_filename: str = ""
