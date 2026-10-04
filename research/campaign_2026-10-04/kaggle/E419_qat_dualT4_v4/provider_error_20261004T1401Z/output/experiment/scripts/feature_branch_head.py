"""Statistical code heads and fit-only empirical moments, without oracle inputs."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wnn.outcomes import N_CODES, N_SP, N_U

PARAMETER_BUDGET = 65_536
ARMS = ("wmse", "joint_ce", "joint_tilted_wmse", "candidate", "factorized_tilted_wmse")


def factorized_log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """P(class|h) P(spread-code|h) P(branch|h) P(U-bin|branch,h);32 logits,504 codes."""
    if logits.shape[-1] != 32:
        raise ValueError("Factorized output requires32 logits")
    market, spread, branch = (
        F.log_softmax(logits[..., start:end], -1) for start, end in ((0, 4), (4, 11), (11, 14))
    )
    bins = torch.cat(
        [
            F.log_softmax(logits[..., start:end], -1) + branch[..., index : index + 1]
            for index, (start, end) in enumerate(((14, 19), (19, 27), (27, 32)))
        ],
        -1,
    )
    return (
        market[..., :, None, None] + spread[..., None, :, None] + bins[..., None, None, :]
    ).flatten(-3)


def empirical_moments(
    codes: np.ndarray, targets: np.ndarray, row_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Smooth sparse code moments through class/branch then branch/global FIT-only means."""
    y = np.clip(np.asarray(targets, dtype=np.float64), -2, 2)
    weight = np.asarray(row_weight, dtype=np.float64)
    labels = np.asarray(codes)
    if (
        y.shape != (len(labels), 2)
        or weight.shape != labels.shape
        or labels.dtype.kind not in "iu"
        or np.any((labels < 0) | (labels >= N_CODES))
        or not np.isfinite(y).all()
        or not np.isfinite(weight).all()
        or np.any(weight < 0)
        or weight.sum() <= 0
    ):
        raise ValueError("Invalid fitting code/moment data")
    # Moment order: |y|y then |y|, separately for both targets.
    values = np.concatenate((np.abs(y) * y, np.abs(y)), axis=1)
    all_codes = np.arange(N_CODES)
    branch = np.where(all_codes % N_U < 5, 0, np.where(all_codes % N_U < 13, 1, 2))
    classes = all_codes // (N_SP * N_U)
    global_mean = np.average(values, axis=0, weights=weight)
    if np.any(global_mean[2:] <= 0):
        raise ValueError("Fitting moment targets require positive magnitude mass")
    branch_mean = np.tile(global_mean, (3, 1))
    class_branch_mean = np.empty((4, 3, 4))
    for b in range(3):
        selected = branch[labels] == b
        if weight[selected].sum() > 0:
            branch_mean[b] = np.average(values[selected], axis=0, weights=weight[selected])
        for c in range(4):
            selected = (branch[labels] == b) & (classes[labels] == c)
            class_branch_mean[c, b] = branch_mean[b]
            if weight[selected].sum() > 0:
                class_branch_mean[c, b] = np.average(
                    values[selected], axis=0, weights=weight[selected]
                )
    total = np.zeros((N_CODES, 4))
    support = np.bincount(labels, weights=weight, minlength=N_CODES)
    np.add.at(total, labels, values * weight[:, None])
    # Eight fitting-equivalent rows per code is fixed, including never-seen codes.
    table = (total + 8 * class_branch_mean[classes, branch]) / (support[:, None] + 8)
    return (
        table[:, :2].astype(np.float32),
        table[:, 2:].astype(np.float32),
        {
            "pseudo_rows": 8,
            "unknown_codes": int(np.sum(support == 0)),
            "codes_below_eight_weighted_rows": int(np.sum(support < 8)),
            "weighted_support": support.tolist(),
            "backoff": "FIT class/branch mean, else FIT branch mean, else FIT global mean",
        },
    )


def rational_readout(
    log_probability: torch.Tensor, m1: torch.Tensor, m0: torch.Tensor, tilted: bool = False
) -> torch.Tensor:
    """Metric-clipped empirical rational mean; no target-derived values at prediction."""
    probability = log_probability.exp()
    if tilted:
        probability = probability / m0.mean(-1).clamp_min(1e-3)
    return (probability @ m1) / (probability @ m0).clamp_min(1e-6)


class Head(nn.Module):
    """One fixed-budget head on fit-normalized frozen state coordinates."""

    def __init__(self, width: int, arm: str, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Unknown branch-screen arm")
        output = (
            2
            if arm == "wmse"
            else 32
            if arm in ("candidate", "factorized_tilted_wmse")
            else N_CODES
        )
        hidden = (PARAMETER_BUDGET - output) // (width + output + 1)
        if hidden < 1:
            raise ValueError("State width exceeds the prespecified head budget")
        self.arm = arm
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        self.layers = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, output))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        normalized = ((state - self.mean) / self.scale).clamp(-8, 8)
        return self.layers(normalized)

    def log_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        logits = self(state)
        if self.arm == "wmse":
            raise ValueError("The direct regression head has no categorical probabilities")
        return (
            factorized_log_probabilities(logits)
            if logits.shape[-1] == 32
            else F.log_softmax(logits, -1)
        )

    def prediction(self, state: torch.Tensor, m1: torch.Tensor, m0: torch.Tensor) -> torch.Tensor:
        if self.arm == "wmse":
            return self(state)
        return rational_readout(self.log_probabilities(state), m1, m0, "tilted" in self.arm)
