import csv
import os

from utilities.result_data import agent_label_to_type, EpochRewards, ResultData

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RESULT_DIRECTORY = "results"
RESULT_DIRECTORY_PATH = f"{PACKAGE_DIRECTORY}/../{RESULT_DIRECTORY}"


def _read_results_from_csv(filename: str) -> EpochRewards:
    results: EpochRewards = []
    with open(f"{RESULT_DIRECTORY_PATH}/{filename}.csv", mode="r") as result_file:
        reader = csv.reader(
            result_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            results.append(
                [float(training_step_reward) for training_step_reward in row]
            )

    return results


def get_result_data_for_agent(
    agent_label: str = "VPG", number_of_files: int = 1
) -> ResultData:
    agent_type = agent_label_to_type[agent_label]
    average_epoch_rewards: EpochRewards = []

    if number_of_files == 1:
        average_epoch_rewards = _read_results_from_csv(agent_label)
    else:
        for n in range(1, number_of_files + 1):
            filename = f"{agent_label}_0{n}" if n < 10 else f"{agent_label}_{n}"
            average_epoch_rewards += _read_results_from_csv(filename)

    return ResultData(
        average_epoch_rewards=average_epoch_rewards, agent_type=agent_type
    )
