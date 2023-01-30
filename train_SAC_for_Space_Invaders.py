from agents import trainer, SAC, DQN
import gymnasium as gym
import torch.nn
from utilities.config import Config
from utilities.environments import AtariWrapper

network_parameters = {
        "convolutions": [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        "linear_layer_activations": [torch.nn.LeakyReLU(), torch.nn.Identity()],
        "linear_layer_sizes": [3136, 512],
        "learning_rate": 0.0003,
}

config = Config(
    hyperparameters={
        "SAC": {
            "discount_rate": 0.99,
            "actor_parameters": network_parameters,
            "critic_parameters": {
                    "convolutions": [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                    "linear_layer_activations": [torch.nn.LeakyReLU(), torch.nn.Identity()],
                    "linear_layer_sizes": [3136, 512],
                    "learning_rate": 0.0003,
            },
            "initial_temperature": 0.2,
            "learn_temperature": True,
            "temperature_learning_rate": 0.0003,
            "soft_update_interpolation_factor": 0.001,
            "minibatch_size": 64,
            "buffer_size": 5 * 10**5,
            "pure_exploration_steps": 20000
        },
    },
    episode_length=432000,
    train_for_n_environment_steps=5 * 10 ** 6,
    epochs=1,
    results_filename="SAC_Space_Invaders_rewards",
    log_level="INFO",
    log_filename="SAC_Space_Invaders_logs",
    model_filename="SAC_Space_Invaders_model",
    use_cuda=True,
    update_model_every_n_timesteps=4,
)

env = AtariWrapper(gym.make("ALE/SpaceInvaders-v5"))

if __name__ == "__main__":
    sac_trainer = trainer.Trainer(config)
    sac_trainer.train_agents([SAC.SAC], environment=env)
    sac_trainer.save_results_to_csv()
