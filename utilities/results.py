import csv
import os
from typing import cast

from agents.BaseAgent import BaseAgent
from agents.BasePG import BasePG
from agents.DQN import DQN
from agents.SAC import SAC
import numpy as np
import torch
from utilities.utils import get_dimension_format_string

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RESULT_DIRECTORY = "results"
RESULT_DIRECTORY_PATH = f"{PACKAGE_DIRECTORY}/../{RESULT_DIRECTORY}"


class ResultStorage:
    def __init__(
        self: "ResultStorage",
        filename: str = "VPG_results",
        training_steps_per_epoch: int = 400,
        epochs=5,
        directory="",
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
        self.directory = RESULT_DIRECTORY_PATH if directory == "" else directory

    def add_average_training_step_reward(self: "ResultStorage", reward: float) -> None:
        self.rewards[self.training_step_idx][self.episode_idx] = reward
        self.episode_idx += 1

    def end_epoch(self: "ResultStorage") -> None:
        self.episode_idx = 0
        self.training_step_idx += 1

    def save_results_to_csv(self: "ResultStorage") -> None:
        with open(f"{self.directory}/{self.filename}.csv", mode="w") as result_file:
            result_writer = csv.writer(
                result_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for r in self.rewards:
                result_writer.writerow(r)


class ModelSaver:
    def __init__(self: "ModelSaver", filename: str = "VPG_model", directory="") -> None:
        self.filename = filename
        self.directory = RESULT_DIRECTORY_PATH if directory == "" else directory
        self.best_score_so_far: float = -(10**6)

    def _save_dqn(self: "ModelSaver", agent: DQN) -> None:
        torch.save(
            agent.q_net.state_dict(), f"{self.directory}/{self.filename}_q_net.pt"
        )

    def _save_pg(self: "ModelSaver", agent: BasePG) -> None:
        torch.save(
            agent.policy.policy_net.state_dict(),
            f"{self.directory}/{self.filename}_policy_model.pt",
        )
        torch.save(
            agent.value_net.state_dict(),
            f"{self.directory}/{self.filename}_value_model.pt",
        )

    def _save_sac(self: "ModelSaver", agent: SAC) -> None:
        torch.save(
            agent.actor.policy_net.state_dict(),
            f"{self.directory}/{self.filename}_policy_model.pt",
        )
        torch.save(
            agent.critic1.state_dict(),
            f"{self.directory}/{self.filename}_critic1_model.pt",
        )
        torch.save(
            agent.critic2.state_dict(),
            f"{self.directory}/{self.filename}_critic2_model.pt",
        )
        torch.save(
            agent.critic_target1.state_dict(),
            f"{self.directory}/{self.filename}_target1_model.pt",
        )
        torch.save(
            agent.critic_target2.state_dict(),
            f"{self.directory}/{self.filename}_target2_model.pt",
        )

    def save_model_if_best(self: "ModelSaver", agent: BaseAgent, score: float) -> None:
        if score <= self.best_score_so_far:
            return
        self.best_score_so_far = score
        match agent.__class__.__name__:
            case "VPG" | "TRPG" | "PPO":
                self._save_pg(cast(BasePG, agent))
            case "DQN":
                self._save_dqn(cast(DQN, agent))
            case "SAC":
                self._save_sac(cast(SAC, agent))
