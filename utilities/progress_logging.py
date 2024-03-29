from collections import defaultdict
import logging
import os
from timeit import default_timer as timer
from typing import Dict, List, Optional

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


class ProgressLogger:
    def __init__(
        self: "ProgressLogger",
        level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        filename: str = "logfile",
        directory: str = "",
    ) -> None:
        self.directory = LOG_DIRECTORY_PATH if directory == "" else directory
        self.level = levelmap[level]

        logger = logging.getLogger()
        logger.setLevel(self.level)

        if logger.hasHandlers():
            logger.handlers.clear()

        if log_to_file:
            file_handler = logging.FileHandler(f"{self.directory}/{filename}.log")
            file_handler.setLevel(self.level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
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
        self.timer_start_times: Dict[str, Dict[str, float]] = {
            "episode": {},
            "training_step": {},
            "epoch": {},
            "training": {},
        }

    def store(self: "ProgressLogger", scope: str = "epoch", **kwargs) -> None:
        for key, value in kwargs.items():
            self.data[scope][key].append(value)

    def clear(self: "ProgressLogger", scope: str = "epoch") -> None:
        self.data[scope] = defaultdict(lambda: [])

    def log_table(
        self: "ProgressLogger",
        scope: str = "epoch",
        level: str = "INFO",
        attributes: Optional[List[str]] = None,
    ) -> None:
        table = PrettyTable()
        table.field_names = ["attribute", "mean", "std", "max", "min"]

        attributes = list(self.data[scope].keys()) if attributes is None else attributes

        for attribute_name in attributes:
            attribute_data = self.data[scope][attribute_name]
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

    def start_timer(
        self: "ProgressLogger",
        scope: str = "epoch",
        level: str = "INFO",
        attribute: str = "",
    ) -> None:
        if levelmap[level] >= self.level:
            self.timer_start_times[scope][attribute] = timer()

    def stop_timer(
        self: "ProgressLogger",
        scope: str = "epoch",
        level: str = "INFO",
        attribute: str = "",
    ) -> None:
        if levelmap[level] >= self.level:
            stop_time = timer()
            start_time = self.timer_start_times[scope][attribute]
            self.store(scope=scope, **{f"{attribute}_time": stop_time - start_time})

    def clear_handlers(self: "ProgressLogger") -> None:
        self.logger.handlers.clear()
