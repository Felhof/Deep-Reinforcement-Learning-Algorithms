from typing import List

from utilities.config import Config
from utilities.plotter import Plotter


class Trainer:
    def __init__(self: "Trainer", config: Config) -> None:
        self.config = config
        self.plotter = Plotter()

    def _train_agent(self: "Trainer", agent_type: type) -> None:
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
            avg_overall_rewards, "VPN", show_std=True
        )

    def train_agents(self: "Trainer", agent_types: List[type]) -> None:
        for agent_type in agent_types:
            self._train_agent(agent_type)
        self.plotter.show_plot(self.config.environment_name)
