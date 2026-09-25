"""Outcome codes for a distributional head built on the target law (research H-042).

Both targets are a deterministic function of the market class (tick quantum),
the current spread ``sp`` in ticks and one integer ``U = F - bid``:

    U < 0        t0 = -(sp + 2)   t1 = -U
    0 <= U <= sp t0 = U - sp      t1 = -U
    U > sp       t0 = U - sp      t1 = -(sp + 2)

A row's outcome code is (class, spread bin, U bin). A softmax over codes is
trained with cross-entropy; the metric-optimal prediction is the |y|-tilted
mean  sum_j pi_j E[|y| y | j] / sum_j pi_j E[|y| | j],  with the per-code
moments accumulated from training rows (exact inside non-tail bins).
"""

from __future__ import annotations

import torch

QUANTA = (0.12543, 0.24133, 0.46854, 0.65687)
N_CLASS = 4
N_SP = 7  # spread 1..6 ticks, 7 = 7 or more
N_BELOW = 5  # -U = 1..4, 5 = 5 or more
N_INSIDE = 8  # U = 0..6 inside the spread, 7 = 7 or more
N_ABOVE = 5  # U - sp = 1..4, 5 = 5 or more
N_U = N_BELOW + N_INSIDE + N_ABOVE
N_CODES = N_CLASS * N_SP * N_U
METRIC_CLIP = 2.0


def sequence_class(y: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) targets -> (B,) class index: the quantum the targets are integer multiples of."""
    errs = []
    for q in QUANTA:
        errs.append((torch.round(y / q) * q - y).abs().amax(dim=(1, 2)))
    return torch.stack(errs, 1).argmin(1)


def invert(t: torch.Tensor):
    """Integer targets (..., 2) in ticks -> spread (>= 1) and U."""
    t0, t1 = t[..., 0], t[..., 1]
    up, dn = t0 >= 1, t1 >= 1
    sp = torch.where(up, -t1 - 2, torch.where(dn, -t0 - 2, -(t0 + t1))).clamp_min(1)
    u = torch.where(up, t0 + sp, -t1)
    return sp, u


def outcome_codes(y: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) float targets and (B,) classes -> (B, T) long codes in [0, N_CODES)."""
    q = torch.tensor(QUANTA, device=y.device, dtype=y.dtype)[cls][:, None, None]
    t = torch.round(y / q).long()
    sp, u = invert(t)
    spb = sp.clamp(max=N_SP) - 1
    below = (-u).clamp(1, N_BELOW) - 1
    inside = N_BELOW + u.clamp(0, N_INSIDE - 1)
    above = N_BELOW + N_INSIDE + (u - sp).clamp(1, N_ABOVE) - 1
    ub = torch.where(u < 0, below, torch.where(u > sp, above, inside))
    return (cls[:, None] * N_SP + spb) * N_U + ub


class ReadoutMoments:
    """Running per-code sums of w*|y|*y, w*|y| and w (metric-clipped y, both targets)."""

    def __init__(self, device):
        self.m1 = torch.zeros(N_CODES, 2, device=device, dtype=torch.float64)
        self.m0 = torch.zeros(N_CODES, 2, device=device, dtype=torch.float64)
        self.n = torch.zeros(N_CODES, device=device, dtype=torch.float64)

    def update(self, codes, y, weight):
        yc = y.double().clamp(-METRIC_CLIP, METRIC_CLIP)
        w = weight.double()
        idx = codes.reshape(-1)
        a = (w[..., None] * yc.abs())
        self.m1.index_add_(0, idx, (a * yc).reshape(-1, 2))
        self.m0.index_add_(0, idx, a.reshape(-1, 2))
        self.n.index_add_(0, idx, w.reshape(-1))

    def tables(self):
        n = self.n.clamp_min(1e-12)[:, None]
        return (self.m1 / n).float(), (self.m0 / n).float()


def readout(logits: torch.Tensor, m1: torch.Tensor, m0: torch.Tensor, tilted: bool = False) -> torch.Tensor:
    """(..., N_CODES) logits -> (..., 2) tilted-mean predictions.

    With ``tilted`` the head was trained with |y|-weighted cross-entropy, so its softmax is
    pi_j * mbar_j (mbar = mean |y| of the code over both targets); dividing by mbar recovers pi.
    """
    pi = torch.softmax(logits.float(), dim=-1)
    if tilted:
        pi = pi / (0.5 * (m0[:, 0] + m0[:, 1])).clamp_min(1e-3)
    return (pi @ m1) / (pi @ m0).clamp_min(1e-6)


N_SP_AUX = 12  # auxiliary spread head: 1..11 ticks, 12 = 12 or more


def structure_labels(y: torch.Tensor, cls: torch.Tensor):
    """(B, T, 2) targets, (B,) classes -> spread bin (0..11), branch (0 below, 1 inside, 2 above), class per row."""
    q = torch.tensor(QUANTA, device=y.device, dtype=y.dtype)[cls][:, None, None]
    sp, u = invert(torch.round(y / q).long())
    branch = torch.where(u < 0, 0, torch.where(u > sp, 2, 1))
    return sp.clamp(max=N_SP_AUX) - 1, branch, cls[:, None].expand_as(sp)
