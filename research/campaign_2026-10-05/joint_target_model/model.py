"""Compact paired-codebook heads for the clipped two-target objective."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def tilted_bayes_readout(
    probabilities: torch.Tensor,
    codebook: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Decode each target as its own absolute-value-tilted conditional mean.

    ``probabilities`` has shape ``(..., J)`` and ``codebook`` is ``(J, 2)``.
    If a target's denominator is zero, use that target's parent prediction.
    """
    if probabilities.shape[-1] != codebook.shape[0] or codebook.shape[-1] != 2:
        raise ValueError("incompatible probability/codebook shape")
    magnitude = codebook.abs()
    numerator = torch.einsum("...j,jk->...k", probabilities, magnitude * codebook)
    denominator = torch.einsum("...j,jk->...k", probabilities, magnitude)
    decoded = numerator / denominator.clamp_min(eps)
    return torch.where(denominator > eps, decoded, fallback)


def ordinary_readout(probabilities: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Return the ordinary (unweighted) mean of paired codebook atoms."""
    return torch.einsum("...j,jk->...k", probabilities, codebook)


class JointTargetHead(nn.Module):
    """Matched heads sharing one feature trunk and an exact parent anchor."""

    def __init__(
        self,
        state_width: int,
        paired_codebook: torch.Tensor,
        independent_codebook: torch.Tensor,
        shuffled_codebook: torch.Tensor,
        hidden_width: int = 128,
    ) -> None:
        super().__init__()
        if paired_codebook.ndim != 2 or paired_codebook.shape[1] != 2:
            raise ValueError("paired codebook must have shape (J, 2)")
        if shuffled_codebook.shape != paired_codebook.shape:
            raise ValueError("shuffled codebook shape differs")
        if independent_codebook.ndim != 2 or independent_codebook.shape[0] != 2:
            raise ValueError("independent codebook must have shape (2, K)")
        self.register_buffer("paired_codebook", paired_codebook.float().clamp(-2, 2))
        self.register_buffer("shuffled_codebook", shuffled_codebook.float().clamp(-2, 2))
        self.register_buffer("independent_codebook", independent_codebook.float().clamp(-2, 2))
        input_width = state_width * 2 + 4
        self.features = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
        )
        self.joint_logits = nn.Linear(hidden_width, paired_codebook.shape[0])
        self.independent_logits = nn.Linear(hidden_width, independent_codebook.shape[1] * 2)
        self.residual = nn.Linear(hidden_width, 2)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.metric_gate = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        states: torch.Tensor,
        member_predictions: torch.Tensor,
        arm: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return clipped prediction, joint logits, and independent logits."""
        if states.ndim != 3 or states.shape[1] != 2:
            raise ValueError("states must have shape (N, 2, H)")
        if member_predictions.shape != (len(states), 2, 2):
            raise ValueError("member predictions must have shape (N, 2, 2)")
        parent = member_predictions.mean(dim=1)
        joined = torch.cat(
            (states.flatten(start_dim=1), member_predictions.flatten(start_dim=1)), dim=1
        )
        feature = self.features(joined)
        joint_logits = self.joint_logits(feature)
        independent_logits = self.independent_logits(feature).reshape(
            -1, 2, self.independent_codebook.shape[1]
        )
        residual = self.residual(feature)
        if arm == "regression_anchor":
            prediction = parent + residual
        elif arm == "independent_tilted":
            probabilities = independent_logits.softmax(dim=-1)
            codebook = self.independent_codebook
            magnitude = codebook.abs()
            numerator = torch.einsum("nkm,km->nk", probabilities, magnitude * codebook)
            denominator = torch.einsum("nkm,km->nk", probabilities, magnitude)
            decoded = torch.where(
                denominator > 1e-12, numerator / denominator.clamp_min(1e-12), parent
            )
            prediction = parent + torch.tanh(self.metric_gate) * (decoded - parent) + residual
        else:
            codebook = self.shuffled_codebook if arm == "shuffled_tilted" else self.paired_codebook
            probabilities = joint_logits.softmax(dim=-1)
            if arm in ("joint_tilted", "shuffled_tilted"):
                decoded = tilted_bayes_readout(probabilities, codebook, parent)
            elif arm == "joint_ordinary":
                decoded = ordinary_readout(probabilities, codebook)
            else:
                raise ValueError(f"unknown arm: {arm}")
            prediction = parent + torch.tanh(self.metric_gate) * (decoded - parent) + residual
        return prediction.clamp(-2, 2), joint_logits, independent_logits


def weighted_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pooled clipped absolute-target-weighted MSE surrogate for the official WP metric."""
    clipped_target = target.clamp(-2, 2)
    clipped_prediction = prediction.clamp(-2, 2)
    weights = clipped_target.abs()
    numerator = (weights * (clipped_prediction - clipped_target).square()).sum()
    denominator = weights.sum()
    return numerator / denominator.clamp_min(1e-12)


def require_finite_training_state(
    loss: torch.Tensor,
    components: tuple[torch.Tensor, ...],
    parameters: Iterable[nn.Parameter],
    phase: str,
) -> None:
    """Fail closed on nonfinite losses, gradients, or updated parameters."""
    checks = [torch.isfinite(loss).all()]
    checks.extend(torch.isfinite(value).all() for value in components)
    parameter_list = list(parameters)
    if phase == "gradients":
        gradients = [parameter.grad for parameter in parameter_list if parameter.grad is not None]
        if not gradients:
            raise FloatingPointError("training step produced no parameter gradients")
        checks.extend(torch.isfinite(gradient).all() for gradient in gradients)
    elif phase == "parameters":
        checks.extend(torch.isfinite(parameter).all() for parameter in parameter_list)
    else:
        raise ValueError(f"unknown finite-check phase: {phase}")
    if not bool(torch.stack(checks).all().item()):
        raise FloatingPointError(f"nonfinite loss, {phase}, or model state during training")


def arm_loss(
    prediction: torch.Tensor,
    joint_logits: torch.Tensor,
    independent_logits: torch.Tensor,
    joint_labels: torch.Tensor,
    independent_labels: torch.Tensor,
    arm: str,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed joint/marginal CE plus identical metric loss for all arms."""
    if arm == "independent_tilted":
        ce = torch.stack(
            [F.cross_entropy(independent_logits[:, k], independent_labels[:, k]) for k in range(2)]
        ).mean()
    else:
        ce = F.cross_entropy(joint_logits, joint_labels)
    wmse_surrogate = weighted_mse(prediction, target)
    return ce + wmse_surrogate, ce.detach(), wmse_surrogate.detach()
