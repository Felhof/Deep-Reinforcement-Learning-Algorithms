import csv
from typing import List

from utilities.config import Config
from utilities.plotter import Plotter


class Trainer:
    def __init__(self: "Trainer", config: Config) -> None:
        self.config = config
        self.plotter = Plotter(target_score=config.target_score)

    def _train_agent(
        self: "Trainer",
        agent_type: type,
        result_filename: str = "",
        save_to_file: bool = False,
    ) -> None:
        assert self.config is not None
        assert self.config.epochs >= 1

        avg_overall_rewards: List[List[float]] = []
        for epoch in range(self.config.epochs):
            print("Epoch: ", epoch)
            agent = agent_type(self.config)
            avg_epoch_rewards = agent.train()
            print("Average rewards on each training step in this epoch:")
            print(avg_epoch_rewards)
            avg_overall_rewards.append(avg_epoch_rewards)
        self.plotter.plot_average_agent_overall_results(
            avg_overall_rewards, agent_type, show_std=True
        )
        if save_to_file:
            with open(f"results/{result_filename}.csv", mode="w") as result_file:
                result_writer = csv.writer(
                    result_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                for r in avg_overall_rewards:
                    result_writer.writerow(r)

    def train_agents(
        self: "Trainer",
        agent_types: List[type],
        result_filename: str = "",
        save_to_file: bool = False,
    ) -> None:
        for agent_type in agent_types:
            self._train_agent(
                agent_type, result_filename=result_filename, save_to_file=save_to_file
            )

    def save_results(self: "Trainer", filename: str = "results.png") -> None:
        self.plotter.create_plot(title=self.config.plot_name, filename=filename)
