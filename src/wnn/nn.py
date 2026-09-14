"""Recurrent models and metric-aligned losses (research side, torch)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

METRIC_CLIP = 2.0


class RNNModel(nn.Module):
    """Input projection -> stacked GRU -> head fed by GRU output and the projection skip."""

    def __init__(
        self, n_in: int, hidden: int = 256, layers: int = 2, dropout: float = 0.0, n_out: int = 2
    ):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out)
        )

    def forward(self, x, h=None):
        z = F.gelu(self.inp((x - self.mean) / self.std))
        o, h = self.gru(z, h)
        return self.head(torch.cat([o, z], dim=-1)), h


def weighted_corr_loss(pred, target, valid):
    """1 - mean weighted Pearson over (batch, target) with metric weights |clip(y)|.

    pred, target: (B, T, K); valid: (B, T) bool. Predictions are not clipped
    (keeps gradients); submission output scale is set separately.
    """
    pred = pred.float()
    y = target.float().clamp(-METRIC_CLIP, METRIC_CLIP)
    w = y.abs() * valid[..., None]
    wsum = w.sum(dim=1).clamp_min(1e-8)
    my = (w * y).sum(1) / wsum
    mp = (w * pred).sum(1) / wsum
    yc = y - my[:, None]
    pc = pred - mp[:, None]
    cov = (w * yc * pc).sum(1) / wsum
    vy = (w * yc * yc).sum(1) / wsum
    vp = (w * pc * pc).sum(1) / wsum
    rho = cov / torch.sqrt(vy * vp + 1e-12)
    ok = (wsum > 1e-6) & (vy > 1e-8)
    return 1.0 - (rho * ok).sum() / ok.sum().clamp_min(1)


def weighted_mse_loss(pred, target, valid):
    y = target.float().clamp(-METRIC_CLIP, METRIC_CLIP)
    w = y.abs() * valid[..., None]
    return (w * (pred.float() - y) ** 2).sum() / w.sum().clamp_min(1e-8)
