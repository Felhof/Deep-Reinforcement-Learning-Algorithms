from agents.VPG import VPG
import torch.nn
from utilities.config import Config

cartpoleConfig = Config(
    environment_name="CartPole-v1",
    action_dim=1,
    observation_dim=4,
    number_of_actions=2,
    hyperparameters={
        "policy_gradient": {
            "episodes_per_training_step": 30,
            "value_updates_per_training_step": 20,
            "discount_rate": 0.99,
            "generalized_advantage_estimate_exponential_mean_discount_rate": 0.92,
            "policy_parameters": {
                "sizes": [4, 128, 2],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "value_net_parameters": {
                "sizes": [4, 128, 2],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "policy_learning_rate": 0.001,
            "value_net_learning_rate": 0.001,
        },
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
        "policy_gradient": {
            "episodes_per_training_step": 30,
            "value_updates_per_training_step": 20,
            "discount_rate": 0.99,
            "generalized_advantage_estimate_exponential_mean_discount_rate": 0.92,
            "policy_parameters": {
                "sizes": [2, 128, 3],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "value_net_parameters": {
                "sizes": [2, 128, 3],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "policy_learning_rate": 0.001,
            "value_net_learning_rate": 0.001,
        },
    },
    episode_length=5,
    training_steps_per_epoch=5,
    epochs=1,
    target_score=200,
)

# TODO: Implement non-categorical actions to make a test with this config work
mountainCarContinuousConfig = Config(
    environment_name="MountainCarContinuous-v0",
    action_dim=1,
    observation_dim=2,
    number_of_actions=1,
    hyperparameters={
        "policy_gradient": {
            "episodes_per_training_step": 30,
            "value_updates_per_training_step": 20,
            "discount_rate": 0.99,
            "generalized_advantage_estimate_exponential_mean_discount_rate": 0.92,
            "policy_parameters": {
                "sizes": [2, 128, 1],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "value_net_parameters": {
                "sizes": [2, 128, 1],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "policy_learning_rate": 0.001,
            "value_net_learning_rate": 0.001,
        },
    },
    episode_length=5,
    training_steps_per_epoch=5,
    epochs=1,
    target_score=200,
)


def test_can_train_with_different_environment_dimensions() -> None:
    agent = VPG(cartpoleConfig)
    avg_rewards = agent.train()
    assert len(avg_rewards) == 5

    agent = VPG(mountainCarConfig)
    avg_rewards = agent.train()
    assert len(avg_rewards) == 5
