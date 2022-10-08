from typing import List

from utilities.config import Config


class Trainer:
    def __init__(self: "Trainer", config: Config) -> None:
        self.config = config

    def _train_agent(self: "Trainer", agent_type: type) -> None:
        assert self.config is not None
        assert self.config.epochs >= 1

        for _epoch in range(self.config.epochs):
            agent = agent_type(self.config)
            agent.train()

    def train_agents(self: "Trainer", agent_types: List[type]) -> None:
        for agent_type in agent_types:
            self._train_agent(agent_type)
