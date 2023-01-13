from agents import trainer, PPO
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
                "hidden_layer_sizes": [128],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
                "learning_rate": 0.001,
            },
            "value_net_parameters": {
                "hidden_layer_sizes": [128],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
                "learning_rate": 0.001,
            },
        },
        "PPO": {
            "clip_range": 0.1
        }
    },
    episode_length=230,
    training_steps_per_epoch=400,
    epochs=5,
    results_filename="PPO_cartpole_rewards2",
    log_level="INFO",
    log_filename="PPO_cartpole_debug2",
)

env = BaseEnvironmentWrapper(gym.make("CartPole-v1"))

if __name__ == "__main__":
    ppo_trainer = trainer.Trainer(config)
    ppo_trainer.train_agents([PPO.PPO], environment=env)
    ppo_trainer.save_results_to_csv()
