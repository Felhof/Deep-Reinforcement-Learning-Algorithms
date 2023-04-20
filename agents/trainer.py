from typing import List

from utilities.config import Config
from utilities.environments import BaseEnvironmentWrapper
from utilities.results import ModelSaver, ResultStorage


class Trainer:
    def __init__(self: "Trainer", config: Config) -> None:
        assert (
            config.training_steps_per_epoch >= 1
            and config.evaluate_every_n_training_steps >= 1
        ) or (
            config.train_for_n_environment_steps >= 1
            and config.evaluate_every_n_timesteps >= 1
        ), (
            "Either both training_steps_per_epoch and evaluate_every_n_training_steps "
            "or train_for_n_environment_steps and evaluate_every_n_environment_steps "
            "must be positive."
        )

        self.config = config
        if config.training_steps_per_epoch >= 1:
            evaluations_per_epoch = int(
                config.training_steps_per_epoch / config.evaluate_every_n_training_steps
            )
        else:
            evaluations_per_epoch = int(
                config.train_for_n_environment_steps / config.evaluate_every_n_timesteps
            )
        self.result_storage = ResultStorage(
            filename=config.results_filename,
            directory=config.results_directory,
            evaluations_per_epoch=evaluations_per_epoch,
            epochs=config.epochs,
        )

    def _train_agent(
        self: "Trainer", agent_type: type, environment: BaseEnvironmentWrapper
    ) -> None:
        assert self.config is not None
        assert self.config.epochs >= 1

        for _epoch in range(self.config.epochs):
            agent = agent_type(
                environment=environment,
                config=self.config,
                result_storage=self.result_storage,
                model_saver=ModelSaver(
                    filename=self.config.model_filename,
                    directory=self.config.model_directory,
                ),
            )
            agent.train()
            self.result_storage.end_epoch()

    def train_agents(
        self: "Trainer", agent_types: List[type], environment: BaseEnvironmentWrapper
    ) -> None:
        for agent_type in agent_types:
            self._train_agent(agent_type, environment)

    def save_results_to_csv(self: "Trainer") -> None:
        self.result_storage.save_results_to_csv()
