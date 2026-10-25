"""Research AdamW variants for noise-dominated recurrent training (strict FP32, default off).

- ``CautiousAdamW``: AdamW whose update is masked to coordinates agreeing in sign with the current
  gradient, rescaled by the kept fraction (Liang et al. 2024, arXiv 2411.16085).
- ``AdEMAMix``: AdamW with an additional slow gradient EMA (beta3) mixed in with weight alpha;
  alpha and beta3 warm up over ``warmup`` steps (Pagliardini et al. 2024, arXiv 2409.03137).

Both keep ``betas[0]`` as the fast momentum so OneCycleLR's momentum cycling still applies, and use
decoupled weight decay exactly like ``torch.optim.AdamW``.
"""

from __future__ import annotations

import math

import torch


class CautiousAdamW(torch.optim.Optimizer):
    """AdamW with the cautious sign-agreement mask on each parameter's update."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one masked AdamW update and return the closure loss when given."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v, t = state["m"], state["v"], state["step"]
                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                denom = (v / (1 - beta2**t)).sqrt_().add_(group["eps"])
                update = (m / (1 - beta1**t)) / denom
                mask = (update * g > 0).to(update.dtype)
                update.mul_(mask).mul_(mask.numel() / mask.sum().clamp_min(1.0))
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss


class AdEMAMix(torch.optim.Optimizer):
    """AdamW plus a slow, bias-uncorrected gradient EMA with warmed-up mixing weight."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        alpha=4.0,
        beta3=0.999,
        warmup=0,
    ):
        if alpha < 0 or not 0 <= beta3 < 1 or warmup < 0:
            raise ValueError("AdEMAMix needs alpha >= 0, beta3 in [0,1) and warmup >= 0")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            beta3=beta3,
            warmup=warmup,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _schedules(group: dict, t: int) -> tuple[float, float]:
        """Linear alpha warmup and the paper's beta3 schedule from beta1 to beta3."""
        alpha, beta3, warmup = group["alpha"], group["beta3"], group["warmup"]
        if not warmup or t >= warmup or beta3 == 0:
            return alpha, beta3
        frac = t / warmup
        log_start, log_end = math.log(group["betas"][0]), math.log(beta3)
        scheduled = math.exp(log_start * log_end / ((1 - frac) * log_end + frac * log_start))
        return alpha * frac, min(scheduled, beta3)

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one AdEMAMix update and return the closure loss when given."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                alpha, beta3 = self._schedules(group, t)
                state["m1"].lerp_(g, 1 - beta1)
                state["m2"].lerp_(g, 1 - beta3)
                state["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                denom = (state["v"] / (1 - beta2**t)).sqrt_().add_(group["eps"])
                numer = state["m1"] / (1 - beta1**t) + alpha * state["m2"]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.addcdiv_(numer, denom, value=-group["lr"])
        return loss
