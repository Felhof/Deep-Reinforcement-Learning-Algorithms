from agents import trainer, SAC, DQN
import gymnasium as gym
import torch.nn
from utilities.config import Config
from utilities.environments import AtariWrapper

network_parameters = {
        "convolutions": [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        "linear_layer_activations": [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Tanh()],
        "linear_layer_sizes": [3136, 512],
        "learning_rate": 0.00025,
}

config = Config(
    hyperparameters={
        "SAC": {
            "discount_rate": 0.99,
            "actor_parameters": network_parameters,
            "critic_parameters": network_parameters,
            "initial_temperature": 1.,
            "learn_temperature": True,
            "temperature_learning_rate": 0.001,
            "soft_update_interpolation_factor": 0.01,
            "minibatch_size": 32,
            "buffer_size": 10**5,
            "pure_exploration_steps": 50000
        },
        "DQN": {
            "discount_rate": 0.99,
            "q_net_parameters": network_parameters,
            "q_net_learning_rate": 0.00025,
            "minibatch_size": 32,
            "buffer_size": 10**5,
            "initial_exploration_rate": 1,
            "pure_exploration_steps": 50000,
            "gradient_clipping_norm": 0.7,
        },
    },
    # episode_length=5 * 10**5,
    episode_length=1000,
    training_steps_per_epoch=500,
    epochs=1,
    results_filename="SAC_breakout_rewards",
    log_level="INFO",
    log_filename="SAC_breakout_debug",
    model_filename="SAC_test",
)

env = AtariWrapper(gym.make("ALE/Breakout-v5"))

if __name__ == "__main__":
    sac_trainer = trainer.Trainer(config)
    sac_trainer.train_agents([DQN.DQN], environment=env)
    sac_trainer.save_results_to_csv()
