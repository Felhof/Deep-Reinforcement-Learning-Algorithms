from agents.AbstractPG import AbstractPG
import torch
from utilities.config import Config


class PPO(AbstractPG):
    def __init__(self: "PPO", **kwargs) -> None:
        super().__init__(**kwargs)
        config: Config = kwargs["config"]
        self.clip_range = config.hyperparameters["PPO"]["clip_range"]

    def _update_policy(
        self: "PPO", obs: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor
    ) -> None:
        ppo_clip_objective = self._ppo_clip_objective(obs, actions, advantages)
        ppo_clip_objective_grad = torch.autograd.grad(
            ppo_clip_objective, [param for param in self.policy.parameters()]
        )
        for param, grad in zip(self.policy.parameters(), ppo_clip_objective_grad):
            param.grad = grad
        self.policy_optimizer.step()

    def _ppo_clip_objective(
        self: "PPO", obs: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor
    ) -> torch.Tensor:
        action_log_probs = torch.exp(self._log_probs_from_actions(obs, actions))
        fixed_action_log_probs = action_log_probs.detach()
        action_log_prob_ratio = action_log_probs / fixed_action_log_probs
        advantages_scaled_by_log_probs = action_log_prob_ratio * advantages
        clipped_objectives_at_timesteps = torch.min(
            advantages_scaled_by_log_probs, self._g(advantages)
        )
        return -torch.mean(clipped_objectives_at_timesteps)

    def _g(self: "PPO", advantages: torch.Tensor) -> torch.Tensor:
        return (advantages >= 0) * (1 + self.clip_range) * advantages + (
            advantages < 0
        ) * (1 - self.clip_range) * advantages
