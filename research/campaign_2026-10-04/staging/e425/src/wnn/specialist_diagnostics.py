"""Low-overhead training diagnostics for the E425 soft expert mixture."""

from __future__ import annotations

import torch
from torch import Tensor

from wnn.specialist import SpecialistGRUModel


class SoftRouteDiagnostics:
    """Aggregate soft-route entropy, expert gradients, and actual updates.

    Gradient norms are measured after AMP unscaling and configured gradient
    dynamics, immediately before the optimizer step. They include the full
    training objective flowing through predictions: main supervision,
    auxiliary targets, distillation, and EMA consistency where enabled. They
    do not isolate the main-label loss. Nonfinite expert gradients are counted
    and excluded from the finite-gradient mean. AMP skip and completed-update
    counts remain the trainer's separate ``EpochUpdateCounter`` diagnostics.
    Parameter deltas are read after each optimizer call, so skipped calls
    contribute zero movement.
    """

    def __init__(self, model: SpecialistGRUModel) -> None:
        if model.route != "soft":
            raise ValueError("soft-route diagnostics require route='soft'")
        parameter = next(model.parameters())
        self.device = parameter.device
        self.experts = model.num_experts
        self.route_entropy_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self.route_rows = 0
        self.route_probability_sum = torch.zeros(
            self.experts, dtype=torch.float64, device=self.device
        )
        self.gradient_norm_sum = torch.zeros(self.experts, dtype=torch.float64, device=self.device)
        self.gradient_nonzero_steps = torch.zeros(
            self.experts, dtype=torch.int64, device=self.device
        )
        self.gradient_finite_steps = torch.zeros(
            self.experts, dtype=torch.int64, device=self.device
        )
        self.gradient_nonfinite_steps = torch.zeros(
            self.experts, dtype=torch.int64, device=self.device
        )
        self.update_norm_sum = torch.zeros(self.experts, dtype=torch.float64, device=self.device)
        self.update_nonzero_steps = torch.zeros(self.experts, dtype=torch.int64, device=self.device)
        self.optimizer_attempts = 0

    @torch.no_grad()
    def before_optimizer_step(self, model: SpecialistGRUModel) -> list[list[Tensor]]:
        """Record this chunk's route and gradients; snapshot expert parameters."""
        probabilities = model.last_route_probabilities
        if probabilities is None:
            raise RuntimeError("soft route probabilities are absent for the optimizer step")
        probabilities = probabilities.float()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        self.route_entropy_sum.add_(entropy.sum().to(torch.float64))
        self.route_probability_sum.add_(probabilities.sum(dim=(0, 1)).to(torch.float64))
        self.route_rows += entropy.numel()

        snapshots: list[list[Tensor]] = []
        for index in range(self.experts):
            parameters = [
                *model.experts[index].parameters(),
                *model.expert_heads[index].parameters(),
            ]
            squared_norm = torch.zeros((), dtype=torch.float64, device=self.device)
            finite = torch.ones((), dtype=torch.bool, device=self.device)
            for parameter in parameters:
                if parameter.grad is not None:
                    gradient = parameter.grad.detach()
                    finite_values = torch.isfinite(gradient)
                    finite.logical_and_(finite_values.all())
                    safe_gradient = torch.where(finite_values, gradient, 0).to(torch.float64)
                    squared_norm.add_(safe_gradient.square().sum())
            norm = squared_norm.sqrt()
            finite_count = finite.to(torch.int64)
            self.gradient_finite_steps[index].add_(finite_count)
            self.gradient_nonfinite_steps[index].add_((~finite).to(torch.int64))
            self.gradient_norm_sum[index].add_(norm * finite.to(torch.float64))
            self.gradient_nonzero_steps[index].add_((finite & norm.gt(0)).to(torch.int64))
            snapshots.append([parameter.detach().clone() for parameter in parameters])
        self.optimizer_attempts += 1
        return snapshots

    @torch.no_grad()
    def after_optimizer_step(
        self, model: SpecialistGRUModel, snapshots: list[list[Tensor]]
    ) -> None:
        """Accumulate measured expert parameter deltas after an optimizer call."""
        if len(snapshots) != self.experts:
            raise ValueError("expert snapshot count does not match the model")
        for index, before in enumerate(snapshots):
            parameters = [
                *model.experts[index].parameters(),
                *model.expert_heads[index].parameters(),
            ]
            if len(before) != len(parameters):
                raise ValueError("expert parameter topology changed during fitting")
            squared_delta = torch.zeros((), dtype=torch.float64, device=self.device)
            for previous, parameter in zip(before, parameters, strict=True):
                delta = parameter.detach().float() - previous.float()
                squared_delta.add_(delta.square().sum().to(torch.float64))
            norm = squared_delta.sqrt()
            self.update_norm_sum[index].add_(norm)
            self.update_nonzero_steps[index].add_(norm.gt(0).to(torch.int64))

    @torch.no_grad()
    def finish(self) -> dict[str, object]:
        """Return epoch-level values, synchronizing the device only at finish."""
        if self.optimizer_attempts == 0 or self.route_rows == 0:
            raise ValueError("cannot summarize an empty soft-route epoch")
        finite_gradient_steps = self.gradient_finite_steps.cpu().tolist()
        gradient_means = (
            self.gradient_norm_sum.div(self.gradient_finite_steps.clamp_min(1)).cpu().tolist()
        )
        return {
            "route_entropy_mean_nats": float((self.route_entropy_sum / self.route_rows).item()),
            "route_probability_mean": (
                self.route_probability_sum.div(self.route_rows).cpu().tolist()
            ),
            "route_rows": self.route_rows,
            "optimizer_attempts": self.optimizer_attempts,
            "expert_prediction_gradient_l2_mean": [
                mean if count else None
                for mean, count in zip(gradient_means, finite_gradient_steps, strict=True)
            ],
            "expert_prediction_gradient_nonzero_steps": self.gradient_nonzero_steps.cpu().tolist(),
            "expert_prediction_gradient_finite_steps": finite_gradient_steps,
            "expert_prediction_gradient_nonfinite_steps": (
                self.gradient_nonfinite_steps.cpu().tolist()
            ),
            "expert_parameter_update_l2_mean": (
                self.update_norm_sum.div(self.optimizer_attempts).cpu().tolist()
            ),
            "expert_parameter_nonzero_update_steps": self.update_nonzero_steps.cpu().tolist(),
        }
