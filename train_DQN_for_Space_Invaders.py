from agents import trainer, DQN
import gymnasium as gym
import torch.nn
from utilities.config import Config
from utilities.environments import AtariWrapper

config = Config(
    hyperparameters={
        "DQN": {
            "discount_rate": 0.99,
            "q_net_parameters": {
                    "convolutions": [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                    "linear_layer_activations": [torch.nn.LeakyReLU(), torch.nn.Identity()],
                    "linear_layer_sizes": [3136, 512],
                    "learning_rate": 0.0003,
            },
            "q_net_learning_rate": 0.0003,
            "minibatch_size": 64,
            "buffer_size": 5 * 10**5,
            "initial_exploration_rate": 1,
            "final_exploration_rate": 0.1,
            "exploration_rate_annealing_period": 10 ** 6,
            "pure_exploration_steps": 20000,
            "gradient_clipping_norm": 0.7,
            "soft_update_interpolation_factor": 0.001,
        },
    },
    episode_length=432000,
    train_for_n_environment_steps=5 * 10 ** 6,
    epochs=1,
    results_filename="DQN_Space_Invaders_rewards",
    log_level="INFO",
    log_filename="DQN_Space_Invaders_logs",
    model_filename="DQN_Space_Invaders_test",
    use_cuda=True,
    update_model_every_n_timesteps=4,
)

env = AtariWrapper(gym.make("ALE/SpaceInvaders-v5"))

if __name__ == "__main__":
    sac_trainer = trainer.Trainer(config)
    sac_trainer.train_agents([DQN.DQN], environment=env)
    sac_trainer.save_results_to_csv()