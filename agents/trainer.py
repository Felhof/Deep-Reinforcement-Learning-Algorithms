from typing import List

import numpy as np
from utilities.config import Config  # type: ignore


class Trainer:
    def __init__(self: "Trainer", config: Config = None) -> None:
        self.config = config

    def set_configuration(self: "Trainer", config: Config) -> None:
        self.config = config

    def train_agents(self: "Trainer", agent_types: List[type]) -> None:
        for agent_type in agent_types:
            self._train_agent(agent_type)

    def _train_agent(self: "Trainer", agent_type: type) -> None:
        assert self.config is not None
        assert self.config.epochs >= 1

        avg_rewards = np.empty(
            self.config.epochs,
            dtype=f"({self.config.epochs},{self.config.training_steps_per_epoch})float32",
        )
        for epoch in range(self.config.epochs):
            print("Epoch: ", epoch)
            agent = agent_type(self.config)
            avg_reward_per_step = agent.train()
            print("Average rewards on each training step in this epoch:")
            print(avg_reward_per_step)
            avg_rewards[epoch] = avg_reward_per_step
        print(avg_rewards.sum(axis=1).mean(axis=0))
