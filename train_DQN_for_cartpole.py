from agents import trainer
from agents.DQN import DQN
import torch.nn
from utilities.config import Config

NUMBER_OF_ACTIONS: int = 2
ACTION_DIM: int = 1
OBSERVATION_DIM: int = 4

config = Config(
    environment_name="CartPole-v1",
    action_type="Discrete",
    number_of_actions=2,
    observation_dim=4,
    hyperparameters={
        "DQN": {
            "discount_rate": 0.99,
            "q_net_parameters": {
                "hidden_layer_sizes": [64],
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
            "gradient_clipping_norm": 0.7,
        },
    },
    episode_length=230,
    training_steps_per_epoch=400,
    epochs=5,
    target_score=200,
    results_filename="VPG_cartpole_rewards3",
    log_level="INFO",
    log_filename="VPG_cartpole_debug3",
)

if __name__ == "__main__":
    dqn_trainer = trainer.Trainer(config)
    dqn_trainer.train_agents([DQN])
    dqn_trainer.save_results_to_csv()
