from agents import trainer
from agents.DQN import DQN
import torch.nn
from utilities.config import Config

NUMBER_OF_ACTIONS: int = 2
ACTION_DIM: int = 1
OBSERVATION_DIM: int = 4

config = Config(
    environment_name="CartPole-v1",
    action_dim=ACTION_DIM,
    observation_dim=OBSERVATION_DIM,
    number_of_actions=NUMBER_OF_ACTIONS,
    hyperparameters={
        "DQN": {
            "discount_rate": 0.99,
            "q_net_parameters": {
                "sizes": [OBSERVATION_DIM, 64, NUMBER_OF_ACTIONS],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "q_net_learning_rate": 0.001,
            "minibatch_size": 256,
            "buffer_size": 40000,
            "initial_exploration_rate": 1,
            "random_episodes": 3,
            "gradient_clipping_norm": 0.7
        }
    },
    episode_length=250,
    training_steps_per_epoch=200,
    epochs=5,
    target_score=200,
)

if __name__ == "__main__":
    vpn_trainer = trainer.Trainer(config)
    vpn_trainer.train_agents([DQN])
    vpn_trainer.save_results()
