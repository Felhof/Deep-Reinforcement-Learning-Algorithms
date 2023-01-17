from agents import trainer
from agents.DQN import DQN
import gymnasium as gym
import torch.nn
from utilities.config import Config
from utilities.environments import BaseEnvironmentWrapper

NUMBER_OF_ACTIONS: int = 2
ACTION_DIM: int = 1
OBSERVATION_DIM: int = 4

config = Config(
    hyperparameters={
        "DQN": {
            "discount_rate": 0.99,
            "q_net_parameters": {
                "linear_layer_sizes": [64],
                "linear_layer_activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
                "learning_rate": 0.001,
            },
            "q_net_learning_rate": 0.001,
            "minibatch_size": 256,
            "buffer_size": 40000,
            "initial_exploration_rate": 1,
            "pure_exploration_steps": 3,
            "gradient_clipping_norm": 0.7,
        },
    },
    episode_length=200,
    training_steps_per_epoch=400,
    epochs=5,
    target_score=200,
    results_filename="DQN_cartpole_rewards",
    log_level="INFO",
    log_filename="DQN_cartpole_debug",
)

env = BaseEnvironmentWrapper(gym.make("CartPole-v1"))

if __name__ == "__main__":
    dqn_trainer = trainer.Trainer(config)
    dqn_trainer.train_agents([DQN], environment=env)
    dqn_trainer.save_results_to_csv()
