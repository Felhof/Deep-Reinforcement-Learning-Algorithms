import os

from agents.DQN import DQN
import pytest
from tests.agent_test_helpers import (
    _assert_n_rows_where_stored,
    _train_agent_and_store_result,
    PATH_TO_TEST_RESULTS,
)

CARTPOLE_TEST_RESULTS = "cartpole_dqn_test"
MOUNTAIN_CAR_TEST_RESULTS = "mountain_car_dqn_test"


@pytest.fixture
def cleanup_test_results() -> None:
    yield
    os.remove(f"{PATH_TO_TEST_RESULTS}{CARTPOLE_TEST_RESULTS}.csv")
    os.remove(f"{PATH_TO_TEST_RESULTS}{MOUNTAIN_CAR_TEST_RESULTS}.csv")


def test_can_train_with_different_environment_dimensions(
    cartpole_environment,
    mountain_car_environment,
    cartpole_config,
    mountain_car_config,
    cleanup_test_results,
) -> None:
    config = cartpole_config(CARTPOLE_TEST_RESULTS)
    _train_agent_and_store_result(
        agent=DQN, config=config, environment=cartpole_environment
    )
    _assert_n_rows_where_stored(
        filepath=f"{PATH_TO_TEST_RESULTS}{CARTPOLE_TEST_RESULTS}.csv", n=3
    )

    config = mountain_car_config(MOUNTAIN_CAR_TEST_RESULTS)
    _train_agent_and_store_result(
        agent=DQN, config=config, environment=mountain_car_environment
    )
    _assert_n_rows_where_stored(
        filepath=f"{PATH_TO_TEST_RESULTS}{MOUNTAIN_CAR_TEST_RESULTS}.csv", n=3
    )
