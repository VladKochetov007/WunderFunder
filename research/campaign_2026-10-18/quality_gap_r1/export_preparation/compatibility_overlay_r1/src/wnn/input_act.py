"""Quantile-normalised activation copies of the raw columns (research flag --input-act).

Each raw column is mapped through its training-fit empirical CDF (mid-rank for ties) to a standard normal score,
then passed through sigmoid/tanh/silu/gelu. The map is a pure function of the current row, so it is causal.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

INPUT_ACTS = ("none", "sigmoid", "tanh", "silu", "gelu")
N_KNOTS = 257
Z_CLIP = 4.0


def fit_quantile_table(rows: np.ndarray, knots: int = N_KNOTS) -> np.ndarray:
    """(columns, knots) sorted quantile levels from training rows (N, columns); inputs only."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < knots or not np.isfinite(rows).all():
        raise ValueError("need finite (N, columns) rows with N >= knots")
    return np.quantile(rows, np.linspace(0.0, 1.0, knots), axis=0).T.astype(np.float32)


def quantile_normal(raw: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Standard-normal score of each column value under its fitted table; shape of ``raw``."""
    columns, knots = table.shape
    flat = raw.reshape(-1, columns).T.contiguous()
    table = table.to(flat.device, flat.dtype).contiguous()
    left = torch.searchsorted(table, flat, right=False).to(flat.dtype)
    right = torch.searchsorted(table, flat, right=True).to(flat.dtype)
    u = (0.5 * (left + right) + 0.5) / (knots + 1)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return z.clamp(-Z_CLIP, Z_CLIP).T.reshape(raw.shape)


def input_act_features(raw: torch.Tensor, table: torch.Tensor, act: str) -> torch.Tensor:
    """Activation of the quantile-normal copy of the raw columns."""
    z = quantile_normal(raw, table)
    if act == "sigmoid":
        return torch.sigmoid(z)
    if act == "tanh":
        return torch.tanh(z)
    if act == "silu":
        return F.silu(z)
    if act == "gelu":
        return F.gelu(z)
    raise ValueError(f"unknown input activation {act!r}")
