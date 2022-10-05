import os
from pathlib import Path

import pytest
from utilities.resultlogging import LOG_DIRECTORY, ResultLogger


PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TEST_LOGS = f"{PACKAGE_DIRECTORY}/../{LOG_DIRECTORY}/test_logfile.log"


@pytest.fixture
def cleanup_test_logs() -> None:
    yield
    os.remove(PATH_TO_TEST_LOGS)


def test_logger_logs_to_console_by_default(capfd):
    logger = ResultLogger(log_to_file=False)
    logger.info("Hello World")
    _, err = capfd.readouterr()
    assert "Hello World" in err


def test_logger_logs_to_file_by_default(cleanup_test_logs):
    logger = ResultLogger(filename="test_logfile")
    logger.info("Hello World")

    logfile = Path(PATH_TO_TEST_LOGS)
    assert logfile.is_file()

    with open(PATH_TO_TEST_LOGS) as log:
        line = log.readline()

    assert "Hello World" in line
