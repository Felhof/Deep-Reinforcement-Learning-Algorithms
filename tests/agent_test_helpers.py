import csv
import os
from pathlib import Path

from utilities.results import RESULT_DIRECTORY, ResultStorage


def _train_agent_and_store_result(agent=None, config=None, environment=None):
    result_storage = ResultStorage(
        filename=config.results_filename,
        training_steps_per_epoch=config.training_steps_per_epoch,
        epochs=config.epochs,
    )
    agent = agent(environment, config=config, result_storage=result_storage)
    agent.train()
    result_storage.save_results_to_csv()


def _assert_n_rows_where_stored(filepath="", n=5):
    results_file = Path(filepath)
    assert results_file.is_file()

    with open(filepath) as results:
        reader = csv.reader(
            results, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        stored_rows = len([row for row in reader])

    assert stored_rows == n


PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TEST_RESULTS = f"{PACKAGE_DIRECTORY}/../{RESULT_DIRECTORY}/"
