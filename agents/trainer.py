from typing import List

from utilities.config import Config
from utilities.environments import BaseEnvironmentWrapper
from utilities.results import ResultStorage


class Trainer:
    def __init__(self: "Trainer", config: Config) -> None:
        self.config = config
        self.result_storage = ResultStorage(
            filename=config.results_filename,
            training_steps_per_epoch=config.training_steps_per_epoch,
            epochs=config.epochs,
        )

    def _train_agent(self: "Trainer", agent_type: type, environment: BaseEnvironmentWrapper) -> None:
        assert self.config is not None
        assert self.config.epochs >= 1

        for _epoch in range(self.config.epochs):
            agent = agent_type(environment, config=self.config, result_storage=self.result_storage)
            agent.train()
            self.result_storage.end_epoch()

    def train_agents(self: "Trainer", agent_types: List[type], environment: BaseEnvironmentWrapper) -> None:
        for agent_type in agent_types:
            self._train_agent(agent_type, environment)

    def save_results_to_csv(self: "Trainer") -> None:
        self.result_storage.save_results_to_csv()
