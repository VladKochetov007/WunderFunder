"""Optional spectral-norm cap on recurrent gate blocks (research flag, default off).

Motivated by the matched AdamW-vs-Muon diagnostics of 2026-10-25: AdamW lets single directions of
the GRU update-gate blocks grow to singular values of 25-29, saturating ~45% of gate activations,
while Muon's bounded per-step spectral movement keeps them near 2-3. Every ``every`` optimizer
steps this projection clamps each gate block's singular values at ``tau`` (exact SVD, FP32).
"""

from __future__ import annotations

import torch
from torch import nn

GATE_PREFIX = ("gru.weight_ih_l", "gru.weight_hh_l")


@torch.no_grad()
def cap_gate_blocks_(model: nn.Module, tau: float, gates: int = 3) -> float:
    """Clamp singular values of every recurrent gate block at ``tau``; return the max pre-cap value."""
    largest = 0.0
    for name, p in model.named_parameters():
        if not name.startswith(GATE_PREFIX):
            continue
        rows = p.shape[0] // gates
        for g in range(gates):
            block = p[g * rows : (g + 1) * rows]
            u, s, vh = torch.linalg.svd(block.float(), full_matrices=False)
            largest = max(largest, float(s[0]))
            if float(s[0]) > tau:
                block.copy_(((u * s.clamp_max(tau)) @ vh).to(p.dtype))
    return largest


def maybe_cap_(model: nn.Module, args, step_count: int) -> None:
    """Apply the cap when ``--spectral-cap`` is positive and the step is on the cadence."""
    tau = float(getattr(args, "spectral_cap", 0.0) or 0.0)
    every = int(getattr(args, "spectral_cap_every", 10))
    if tau > 0 and step_count % every == 0:
        cap_gate_blocks_(model, tau)
