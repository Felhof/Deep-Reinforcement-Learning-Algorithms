import torch
import torch.nn as nn
from agents.AbstractPG import AbstractPG
from utilities.config import Config


class VPG(AbstractPG):
    def __init__(self: "VPG", config: Config) -> None:
        super().__init__(config)

    def _update_q_net(
        self: "AbstractPG",
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards_to_go: torch.Tensor,
    ) -> None:
        for _ in range(
            self.config.hyperparameters["policy_gradient"]["value_updates_per_training_step"]
        ):
            state_action_values = self._get_state_action_values(obs, actions)
            self.q_net_optimizer.zero_grad()
            q_loss = nn.MSELoss()(state_action_values, rewards_to_go)
            q_loss.backward()
            self.q_net_optimizer.step()