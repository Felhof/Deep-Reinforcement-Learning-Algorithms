import os

from agents.PPO import PPO
import pytest
from tests.agent_test_helpers import (
    _assert_n_rows_where_stored,
    _train_agent_and_store_result,
    PATH_TO_TEST_RESULTS,
)

CARTPOLE_TEST_RESULTS = "cartpole_ppo_test"
MOUNTAIN_CAR_TEST_RESULTS = "mountain_car_ppo_test"


@pytest.fixture
def cleanup_test_results() -> None:
    yield
    os.remove(f"{PATH_TO_TEST_RESULTS}{CARTPOLE_TEST_RESULTS}.csv")
    os.remove(f"{PATH_TO_TEST_RESULTS}{MOUNTAIN_CAR_TEST_RESULTS}.csv")


def test_can_train_with_different_environment_dimensions(
    cartpole_config, cleanup_test_results, mountain_car_config
) -> None:
    config = cartpole_config(CARTPOLE_TEST_RESULTS)
    _train_agent_and_store_result(agent=PPO, config=config)
    _assert_n_rows_where_stored(
        filepath=f"{PATH_TO_TEST_RESULTS}{CARTPOLE_TEST_RESULTS}.csv", n=3
    )

    config = mountain_car_config(MOUNTAIN_CAR_TEST_RESULTS)
    _train_agent_and_store_result(agent=PPO, config=config)
    _assert_n_rows_where_stored(
        filepath=f"{PATH_TO_TEST_RESULTS}{MOUNTAIN_CAR_TEST_RESULTS}.csv", n=3
    )
