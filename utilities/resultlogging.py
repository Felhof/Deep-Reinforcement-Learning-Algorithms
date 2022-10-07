import logging
import os
from typing import Dict


PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_DIRECTORY = "logs"
LOG_DIRECTORY_PATH = f"{PACKAGE_DIRECTORY}/../{LOG_DIRECTORY}"

file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s -- module:%(module)s function:%(module)s"
)
console_formatter = logging.Formatter("%(levelname)s -- %(message)s")


levelmap: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARN,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ResultLogger:
    def __init__(
        self: "ResultLogger",
        level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        filename: str = "logfile",
    ) -> None:
        logger = logging.getLogger()
        logger.setLevel(levelmap[level])

        if log_to_file:
            file_handler = logging.FileHandler(f"{LOG_DIRECTORY_PATH}/{filename}.log")
            file_handler.setLevel(levelmap[level])
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(levelmap[level])
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        self.debug = logger.debug
        self.info = logger.info
        self.warn = logger.warning
        self.error = logger.error
        self.critical = logger.critical
        self.logger = logger
