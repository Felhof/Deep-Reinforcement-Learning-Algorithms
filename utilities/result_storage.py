import csv
import os

import numpy as np
from utilities.utils import get_dimension_format_string


PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RESULT_DIRECTORY = "results"
RESULT_DIRECTORY_PATH = f"{PACKAGE_DIRECTORY}/../{RESULT_DIRECTORY}"


class ResultStorage:
    def __init__(
        self: "ResultStorage", training_step_length: int = 30, training_steps=100
    ) -> None:
        self.rewards = np.zeros(
            training_steps,
            dtype=get_dimension_format_string(training_step_length, dtype="float32"),
        )
        self.episode_idx = 0
        self.training_step_idx = 0

    def add_average_episode_reward(self: "ResultStorage", reward: float) -> None:
        self.rewards[self.training_step_idx][self.episode_idx] = reward
        self.episode_idx += 1

    def end_training_step(self: "ResultStorage") -> None:
        self.episode_idx = 0
        self.training_step_idx += 1

    def store_results(self: "ResultStorage", filename: str = "results") -> None:
        with open(f"{RESULT_DIRECTORY_PATH}/{filename}.csv", mode="w") as result_file:
            result_writer = csv.writer(
                result_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for r in self.rewards:
                result_writer.writerow(r)
