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
    log_directory: str = ""
    results_filename: str = ""
    results_directory: str = ""
    model_filename: str = ""
    model_directory: str = ""
    save: bool = True
    save_every_n_training_steps: int = 5
    save_every_n_timesteps: int = 2500
    evaluate_every_n_training_steps: int = 1
    evaluate_every_n_timesteps: int = 2500
    use_cuda: bool = False
    dtype_name: str = "float32"
    train_for_n_environment_steps: int = 0
    update_model_every_n_timesteps: int = 1
