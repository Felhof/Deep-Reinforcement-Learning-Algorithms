import csv
import os

import numpy as np
from utilities.utils import get_dimension_format_string

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RESULT_DIRECTORY = "results"
RESULT_DIRECTORY_PATH = f"{PACKAGE_DIRECTORY}/../{RESULT_DIRECTORY}"


class ResultStorage:
    def __init__(
        self: "ResultStorage",
        filename: str = "VPG",
        training_steps_per_epoch: int = 400,
        epochs=5,
    ) -> None:
        self.rewards = np.zeros(
            epochs,
            dtype=get_dimension_format_string(
                training_steps_per_epoch, dtype="float32"
            ),
        )
        self.episode_idx = 0
        self.training_step_idx = 0
        self.filename = filename

    def add_average_training_step_reward(self: "ResultStorage", reward: float) -> None:
        self.rewards[self.training_step_idx][self.episode_idx] = reward
        self.episode_idx += 1

    def end_epoch(self: "ResultStorage") -> None:
        self.episode_idx = 0
        self.training_step_idx += 1

    def save_results_to_csv(self: "ResultStorage") -> None:
        with open(
            f"{RESULT_DIRECTORY_PATH}/{self.filename}.csv", mode="w"
        ) as result_file:
            result_writer = csv.writer(
                result_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for r in self.rewards:
                result_writer.writerow(r)
