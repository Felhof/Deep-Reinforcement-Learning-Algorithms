from typing import Callable

import numpy as np
import torch
from agents.AbstractPG import AbstractPG
from utilities.config import Config


class TRPG(AbstractPG):
    def __init__(self: "TRPG", config: Config) -> None:
        super().__init__(config)
        self.tensor_type = torch.float64
        self.delta = config.hyperparameters["TRPG"]["kl_divergence_limit"]
        self.alpha = config.hyperparameters["TRPG"]["backtracking_coefficient"]
        self.backtracking_iters = config.hyperparameters["TRPG"][
            "backtracking_iterations"
        ]
        self.damping_coeff = config.hyperparameters["TRPG"]["damping_coefficient"]
        self.cg_iters = config.hyperparameters["TRPG"]["conjugate_gradient_iterations"]

    def _update_policy(
        self: "TRPG", obs: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor
    ) -> None:
        loss = self._compute_policy_loss(obs, actions, advantages)
        loss_grad = torch.autograd.grad(loss, self.policy.parameters())
        flat_loss_grad = torch.cat([grad.view(-1) for grad in loss_grad]).data
        flat_loss_grad *= -1

        def kl_hessian_vector_product(v):
            log_probs = self._log_probs(obs)

            kl_loss = torch.nn.functional.kl_div(
                log_probs,
                log_probs.clone().detach(),
                log_target=True,
                reduction="batchmean",
            )
            kl_loss_grad = torch.autograd.grad(
                kl_loss, self.policy.parameters(), create_graph=True
            )
            flat_kl_loss_grad = torch.cat([grad.view(-1) for grad in kl_loss_grad])

            kl_v = torch.dot(flat_kl_loss_grad, v.detach())
            kl_v_grad = torch.autograd.grad(kl_v, self.policy.parameters())
            hvp = torch.cat([grad.contiguous().view(-1) for grad in kl_v_grad]).data

            return hvp

        x = self._cga(flat_loss_grad, kl_hessian_vector_product)
        step_direction = (
            torch.sqrt(
                2
                * self.delta
                / (torch.dot(x, kl_hessian_vector_product(x)) + self.damping_coeff)
            )
            * x
        )

        def backtracking_line_search(original_params: torch.Tensor) -> None:
            log_probs = self._log_probs(obs)
            for i, step_fraction in enumerate(
                self.alpha ** np.arange(self.backtracking_iters)
            ):
                new_params = original_params + step_fraction * step_direction
                self._set_flat_params(new_params)
                new_log_probs = self._log_probs(obs)
                kl = torch.nn.functional.kl_div(
                    new_log_probs, log_probs, log_target=True, reduction="batchmean"
                )
                if kl < self.delta:
                    return
            self._set_flat_params(original_params)

        flat_params = self._get_flat_params()
        backtracking_line_search(flat_params)

    def _cga(
        self: "TRPG",
        g: torch.Tensor,
        hessian_vector_product: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        x = torch.zeros_like(g)
        r = g.clone()
        p = g.clone()
        r_dot_old = torch.dot(r, r)

        for _ in range(self.cg_iters):
            hvp = hessian_vector_product(p)
            alpha = r_dot_old / (torch.dot(p, hvp) + self.damping_coeff)
            x += alpha * p
            r -= alpha * hvp
            r_dot_new = torch.dot(r, r)
            p = r + (r_dot_new / r_dot_old * p)
            r_dot_old = r_dot_new

        return x

    def _get_flat_params(self: "TRPG") -> torch.Tensor:
        params = []
        for param in self.policy.parameters():
            params.append(param.data.view(-1))

        flat_params = torch.cat(params)
        return flat_params

    def _set_flat_params(self: "TRPG", flat_params: torch.Tensor) -> None:
        prev_ind = 0
        state_dict = self.policy.state_dict()
        for key, params in self.policy.state_dict().items():
            flat_size = int(np.prod(list(params.size())))
            state_dict[key] = flat_params[prev_ind : prev_ind + flat_size].view(
                params.size()
            )
            prev_ind += flat_size
        self.policy.load_state_dict(state_dict)
