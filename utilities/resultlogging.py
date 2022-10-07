from collections import defaultdict
import logging
import os
from typing import Dict, List

import numpy as np
from prettytable import PrettyTable


PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_DIRECTORY = "logs"
LOG_DIRECTORY_PATH = f"{PACKAGE_DIRECTORY}/../{LOG_DIRECTORY}"

file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s -- module:%(module)s function:%(module)s -- %(message)s "
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

        self.data: Dict[str, defaultdict[str, List[float]]] = {
            "episode": defaultdict(lambda: []),
            "training_step": defaultdict(lambda: []),
            "epoch": defaultdict(lambda: []),
            "training": defaultdict(lambda: []),
        }

    def store(self: "ResultLogger", scope: str = "epoch", **kwargs) -> None:
        for key, value in kwargs.items():
            self.data[scope][key].append(value)

    def clear(self: "ResultLogger", scope: str = "epoch") -> None:
        self.data[scope] = defaultdict(lambda: [])

    def log_table(
        self: "ResultLogger", scope: str = "epoch", level: str = "INFO"
    ) -> None:
        table = PrettyTable()
        table.field_names = ["attribute", "mean", "std", "max", "min"]

        for attribute_name, attribute_data in self.data[scope].items():
            mean = format(np.mean(attribute_data), ".4f")
            std = format(np.std(attribute_data), ".4f")
            attribute_max = format(max(attribute_data), ".4f")
            attribute_min = format(min(attribute_data), ".4f")
            table.add_row([attribute_name, mean, std, attribute_max, attribute_min])

        log_function = {
            "DEBUG": self.debug,
            "INFO": self.info,
            "WARN": self.warn,
            "ERROR": self.error,
            "CRITICAL": self.critical,
        }[level]

        table_strings = table.get_string().split("\n")

        for table_string in table_strings:
            log_function(table_string)
