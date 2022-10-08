import math
import os
from pathlib import Path
import time

import pytest
from utilities.progress_logging import LOG_DIRECTORY, ProgressLogger


TEST_LOGS_FILENAME = "test_logfile"
PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TEST_LOGS = f"{PACKAGE_DIRECTORY}/../{LOG_DIRECTORY}/{TEST_LOGS_FILENAME}.log"


@pytest.fixture
def cleanup_test_logs() -> None:
    yield
    os.remove(PATH_TO_TEST_LOGS)


def test_logger_logs_to_console_by_default(capfd):
    logger = ProgressLogger(log_to_file=False)
    logger.info("Hello World")
    _, err = capfd.readouterr()
    assert "Hello World" in err


def test_logger_logs_to_file_by_default(cleanup_test_logs):
    logger = ProgressLogger(filename=TEST_LOGS_FILENAME)
    logger.info("Hello World")

    logfile = Path(PATH_TO_TEST_LOGS)
    assert logfile.is_file()

    with open(PATH_TO_TEST_LOGS) as log:
        line = log.readline()

    assert "Hello World" in line


def test_logger_logs_statistics_correctly(cleanup_test_logs):
    logger = ProgressLogger(filename=TEST_LOGS_FILENAME)

    logger.store(scope="epoch", reward=1.0)
    logger.store(scope="epoch", reward=2.0)
    logger.store(scope="epoch", reward=3.0)
    logger.store(scope="epoch", other=1.0)
    logger.store(scope="epoch", other=2.0)
    logger.store(scope="epoch", other=3.0)

    logger.log_table(scope="epoch", level="INFO", attributes=["reward"])

    logfile = Path(PATH_TO_TEST_LOGS)
    assert logfile.is_file()

    with open(PATH_TO_TEST_LOGS) as log:
        lines = log.readlines()

    expected_table = [
        "+-----------+--------+--------+--------+--------+",
        "| attribute |  mean  |  std   |  max   |  min   |",
        "+-----------+--------+--------+--------+--------+",
        "|   reward  | 2.0000 | 0.8165 | 3.0000 | 1.0000 |",
        "+-----------+--------+--------+--------+--------+",
    ]

    assert len(lines) == 5
    assert expected_table[0] in lines[0]
    assert expected_table[1] in lines[1]
    assert expected_table[2] in lines[2]
    assert expected_table[3] in lines[3]
    assert expected_table[4] in lines[4]


def test_logger_can_clear_data():
    logger = ProgressLogger()

    logger.store(scope="episode", reward=1)
    logger.store(scope="training_step", reward=2)
    logger.store(scope="epoch", reward=3)
    logger.store(scope="training", reward=4)

    logger.clear(scope="episode")
    logger.clear(scope="training_step")
    logger.clear(scope="epoch")

    assert logger.data["episode"]["reward"] == []
    assert logger.data["training_step"]["reward"] == []
    assert logger.data["epoch"]["reward"] == []
    assert logger.data["training"]["reward"] == [4]


def test_logger_can_measure_time():
    logger = ProgressLogger(level="WARN")

    logger.start_timer(scope="epoch", level="WARN", attribute="wait")
    time.sleep(0.5)
    logger.stop_timer(scope="epoch", level="WARN", attribute="wait")

    wait_time = logger.data["epoch"]["wait_time"][0]

    assert math.isclose(wait_time, 0.5, rel_tol=1e-2)


def test_logger_does_not_measure_time_for_wrong_loglevel():
    logger = ProgressLogger(level="WARN")

    logger.start_timer(scope="epoch", level="INFO", attribute="wait")
    logger.stop_timer(scope="epoch", level="INFO", attribute="wait")

    assert logger.data["epoch"]["wait_time"] == []
