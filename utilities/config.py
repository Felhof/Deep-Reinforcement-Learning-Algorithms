from dataclasses import dataclass, field

from utilities.types import HyperParameters


@dataclass
class Config:
    hyperparameters: HyperParameters = field(default_factory=HyperParameters)
    training_steps_per_epoch: int = 0
    episode_length: int = 0
    epochs: int = 0
    target_score: int = 200
    plot_name: str = ""
    log_level: str = "WARN"
    log_filename: str = ""
    results_filename: str = ""
