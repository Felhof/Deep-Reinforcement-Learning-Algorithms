import os

from agents.PPO import PPO
import pytest
from tests.agent_test_helpers import (
    _assert_n_rows_where_stored,
    PATH_TO_TEST_RESULTS,
    _train_agent_and_store_result,
)
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
        "PPO" : {
            "clip_range": 0.1
        }
    },
    episode_length=5,
    training_steps_per_epoch=5,
    epochs=3,
    target_score=200,
    results_filename="cartpole_ppo",
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
        "PPO" : {
            "clip_range": 0.1
        }
    },
    episode_length=5,
    training_steps_per_epoch=5,
    epochs=3,
    target_score=200,
    results_filename="mountaincar_ppo",
)


@pytest.fixture
def cleanup_test_results() -> None:
    yield
    os.remove(f"{PATH_TO_TEST_RESULTS}cartpole_ppo.csv")
    os.remove(f"{PATH_TO_TEST_RESULTS}mountaincar_ppo.csv")


def test_can_train_with_different_environment_dimensions(cleanup_test_results) -> None:
    _train_agent_and_store_result(agent=PPO, config=cartpoleConfig)
    _assert_n_rows_where_stored(filepath=f"{PATH_TO_TEST_RESULTS}cartpole_ppo.csv", n=3)

    _train_agent_and_store_result(agent=PPO, config=mountainCarConfig)
    _assert_n_rows_where_stored(
        filepath=f"{PATH_TO_TEST_RESULTS}mountaincar_ppo.csv", n=3
    )