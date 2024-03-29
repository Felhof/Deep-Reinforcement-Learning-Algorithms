from agents import trainer, VPG
import gymnasium as gym
import torch.nn
from utilities.config import Config
from utilities.environments import BaseEnvironmentWrapper

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
    },
    episode_length=200,
    training_steps_per_epoch=400,
    epochs=5,
    results_filename="VPG_cartpole_rewards",
    log_level="INFO",
    log_filename="VPG_cartpole_debug",
    model_filename="VPG_test",
)

env = BaseEnvironmentWrapper(gym.make("CartPole-v1"))

if __name__ == "__main__":
    vpg_trainer = trainer.Trainer(config)
    vpg_trainer.train_agents([VPG.VPG], environment=env)
    vpg_trainer.save_results_to_csv()
