import os

from agents.DQN import DQN
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
    epochs=3,
    target_score=200,
    results_filename="cartpole_dqn",
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
    epochs=3,
    target_score=200,
    results_filename="mountaincar_dqn",
)


@pytest.fixture
def cleanup_test_results() -> None:
    yield
    os.remove(f"{PATH_TO_TEST_RESULTS}cartpole_dqn.csv")
    os.remove(f"{PATH_TO_TEST_RESULTS}mountaincar_dqn.csv")


def test_can_train_with_different_environment_dimensions() -> None:
    _train_agent_and_store_result(agent=DQN, config=cartpoleConfig)
    _assert_n_rows_where_stored(filepath=f"{PATH_TO_TEST_RESULTS}cartpole_dqn.csv", n=3)

    _train_agent_and_store_result(agent=DQN, config=mountainCarConfig)
    _assert_n_rows_where_stored(
        filepath=f"{PATH_TO_TEST_RESULTS}mountaincar_dqn.csv", n=3
    )
