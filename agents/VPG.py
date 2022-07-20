import torch

from agents.AbstractPG import AbstractPG
from utilities.config import Config


class VPG(AbstractPG):
    def __init__(self: "VPG", config: Config) -> None:
        super().__init__(config)

    def _update_policy(self: "VPG", obs: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor) -> None:
        policy_loss = self._compute_policy_loss(obs, actions, advantages)
        policy_loss.backward()
        self.policy_optimizer.step()
