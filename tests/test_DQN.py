from agents.DQN import DQN
import torch.nn
from utilities.config import Config

cartpoleConfig = Config(
    environment_name="CartPole-v1",
    action_dim=1,
    observation_dim=4,
    number_of_actions=2,
    hyperparameters={
        "DQN": {
            "discount_rate": 0.99,
            "q_net_parameters": {
                "sizes": [4, 64, 2],
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
        }
    },
    episode_length=5,
    training_steps_per_epoch=5,
    epochs=1,
    target_score=200,
)

mountainCarConfig = Config(
    environment_name="MountainCar-v0",
    action_dim=1,
    observation_dim=2,
    number_of_actions=3,
    hyperparameters={
        "DQN": {
            "discount_rate": 0.99,
            "q_net_parameters": {
                "sizes": [2, 64, 3],
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
        }
    },
    episode_length=5,
    training_steps_per_epoch=5,
    epochs=1,
    target_score=200,
)


def test_can_train_with_different_environment_dimensions() -> None:
    agent = DQN(cartpoleConfig)
    avg_rewards = agent.train()
    assert len(avg_rewards) == 5

    agent = DQN(mountainCarConfig)
    avg_rewards = agent.train()
    assert len(avg_rewards) == 5
