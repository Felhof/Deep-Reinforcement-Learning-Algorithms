from agents import trainer, TRPG
import gymnasium as gym
import torch.nn
from utilities.config import Config
from utilities.environments import BaseEnvironmentWrapper

NUMBER_OF_ACTIONS: int = 2
ACTION_DIM: int = 1
OBSERVATION_DIM: int = 4

config = Config(
    hyperparameters={
        "policy_gradient": {
            "episodes_per_training_step": 30,
            "value_updates_per_training_step": 20,
            "discount_rate": 0.99,
            "gae_exp_mean_discount_rate": 0.92,
            "policy_net_parameters": {
                "linear_layer_sizes": [128],
                "linear_layer_activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
                "learning_rate": 0.001,
            },
            "value_net_parameters": {
                "linear_layer_sizes": [128],
                "linear_layer_activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
                "learning_rate": 0.001,
            },
        },
        "TRPG": {
            "kl_divergence_limit": 0.01,
            "backtracking_coefficient": 0.5,
            "backtracking_iterations": 10,
            "damping_coefficient": 1e-8,
            "conjugate_gradient_iterations": 10,
        },
    },
    episode_length=200,
    training_steps_per_epoch=400,
    epochs=1,
    results_filename="TRPG_cartpole_rewards_05",
    log_level="INFO",
    log_filename="TRPG_cartpole_debug_05",
    dtype_name="float64",
)

env = BaseEnvironmentWrapper(gym.make("CartPole-v1"))

if __name__ == "__main__":
    trpg_trainer = trainer.Trainer(config)
    trpg_trainer.train_agents([TRPG.TRPG], environment=env)
    trpg_trainer.save_results_to_csv()
