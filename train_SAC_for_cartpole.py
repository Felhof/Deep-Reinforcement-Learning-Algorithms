from agents import trainer, SAC
import gymnasium as gym
import torch.nn
from utilities.config import Config
from utilities.environments import BaseEnvironmentWrapper

config = Config(
    hyperparameters={
        "SAC": {
            "discount_rate": 0.99,
            "actor_parameters": {
                "linear_layer_sizes": [128],
                "linear_layer_activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
                "learning_rate": 0.001,
            },
            "critic_parameters": {
                "linear_layer_sizes": [128],
                "linear_layer_activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
                "learning_rate": 0.001,
            },
            "initial_temperature": 0.05,
            "learn_temperature": False,
            "temperature_learning_rate": 0.001,
            "soft_update_interpolation_factor": 0.01,
            "minibatch_size": 256,
            "buffer_size": 40000,
        },
    },
    episode_length=200,
    training_steps_per_epoch=400,
    epochs=5,
    results_filename="SAC_cartpole_rewards2",
    log_level="INFO",
    log_filename="SAC_cartpole_debug2",
    model_filename="SAC_test",
)

env = BaseEnvironmentWrapper(gym.make("CartPole-v1"))

if __name__ == "__main__":
    sac_trainer = trainer.Trainer(config)
    sac_trainer.train_agents([SAC.SAC], environment=env)
    sac_trainer.save_results_to_csv()
