import csv
import os
from pathlib import Path

import pytest
from utilities.results import RESULT_DIRECTORY, ResultStorage

TEST_RESULTS_FILENAME = "test_results"
PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TEST_RESULTS = (
    f"{PACKAGE_DIRECTORY}/../{RESULT_DIRECTORY}/{TEST_RESULTS_FILENAME}.csv"
)


@pytest.fixture
def cleanup_test_results() -> None:
    yield
    os.remove(PATH_TO_TEST_RESULTS)


def test_result_storage_can_store_episode_data(cleanup_test_results):
    storage = ResultStorage(filename=TEST_RESULTS_FILENAME, evaluations_per_epoch=3)

    storage.add_average_training_step_reward(1.0)
    storage.add_average_training_step_reward(2.0)
    storage.add_average_training_step_reward(3.0)
    storage.end_epoch()
    storage.add_average_training_step_reward(4.0)
    storage.save_results_to_csv()

    results_file = Path(PATH_TO_TEST_RESULTS)
    assert results_file.is_file()

    stored_rows = []
    with open(PATH_TO_TEST_RESULTS) as results:
        reader = csv.reader(
            results, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            stored_rows.append(", ".join(row))

    assert stored_rows[0] == "1.0, 2.0, 3.0"
    assert stored_rows[1] == "4.0, 0.0, 0.0"
