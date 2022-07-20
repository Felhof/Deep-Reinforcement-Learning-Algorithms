import numpy as np
import torch
from torch.autograd.functional import hvp
from agents.AbstractPG import AbstractPG
from utilities.config import Config


class TRPG(AbstractPG):

    DELTA = 0.01
    ALPHA = 0.5
    BACKTRACKING_ITERATIONS = 10

    def __init__(self: "TRPG", config: Config) -> None:
        super().__init__(config)

    def _update_policy(
            self: "TRPG", obs: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor
    ) -> None:
        loss = self._compute_policy_loss(obs, actions, advantages)
        loss_grad = torch.autograd.grad(loss, self.policy.parameters())
        flat_loss_grad = torch.cat([grad.view(-1) for grad in loss_grad]).data
        flat_loss_grad *= -1

        # Use conjugate gradient algorithm to compute x = H^-1 * g
        x = torch.zeros_like(flat_loss_grad)
        r = flat_loss_grad.clone()
        p = flat_loss_grad.clone()
        r_dot_old = torch.dot(r, r)

        def kl_hessian_vector_product(v):
            log_probs = self._log_probs(obs)
            # log_probs = self._log_probs_from_actions(obs, actions)

            kl_loss = torch.nn.functional.kl_div(log_probs, log_probs.clone().detach(), log_target=True, reduction='batchmean')
            kl_loss = torch.mean(kl_loss)
            kl_loss_grad = torch.autograd.grad(kl_loss, self.policy.parameters(), create_graph=True)
            flat_kl_loss_grad = torch.cat([grad.view(-1) for grad in kl_loss_grad])

            kl_v = torch.dot(flat_kl_loss_grad, v.detach())
            kl_v_grad = torch.autograd.grad(kl_v, self.policy.parameters())
            hvp = torch.cat([grad.contiguous().view(-1) for grad in kl_v_grad]).data

            return hvp

        for _ in range(20):
            hvp = kl_hessian_vector_product(p)

            alpha = r_dot_old / (torch.dot(p, hvp) + 1e-8)
            x += alpha * p
            r -= alpha * hvp
            r_dot_new = torch.dot(r, r)
            p = r + (r_dot_new / r_dot_old * p)
            r_dot_old = r_dot_new

        # Update policy by backtracking line search
        step_direction = torch.sqrt(2 * self.DELTA / torch.dot(x, kl_hessian_vector_product(x))) * x

        def backtracking_line_search(original_params):
            log_probs = self._log_probs(obs)
            # log_probs = self._log_probs_from_actions(obs, actions)
            for step_fraction in self.ALPHA ** np.arange(self.BACKTRACKING_ITERATIONS):
                new_params = original_params + step_fraction * step_direction
                set_flat_params_to(self.policy, new_params)
                new_log_probs = self._log_probs(obs)
                # new_actions = self.policy(obs)
                # new_log_probs = self._log_probs_from_actions(obs, new_actions)
                kl = torch.nn.functional.kl_div(new_log_probs, log_probs, log_target=True, reduction='batchmean')
                kl = torch.mean(kl)
                if kl < self.DELTA:
                    return
            set_flat_params_to(self.policy, original_params)

        flat_params = get_flat_params_from(self.policy)
        backtracking_line_search(flat_params)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size