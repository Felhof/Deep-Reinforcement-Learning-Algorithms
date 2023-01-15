from agents.BasePG import BasePG
import torch


class VPG(BasePG):
    def __init__(self: "VPG", **kwargs) -> None:
        super().__init__(**kwargs)

    def _update_policy(
        self: "VPG", obs: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor
    ) -> None:
        policy_loss = self.policy.compute_loss(obs, actions, advantages)
        policy_loss.backward()
        self.policy.update_gradients()
