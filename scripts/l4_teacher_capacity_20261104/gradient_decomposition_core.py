"""Focused checkpoint validation and gradient calculations for the L4 control diagnostic."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from wnn.nn import weighted_mse_loss
from wnn.outcomes import N_SP_AUX
from wnn.spread_aux import structural_loss
from wnn.training.forward import detach_state, forward_chunk

SEEDS, EPOCHS = (1001, 1002), (2, 3)
BATCH, CHUNK = 32, 1000
STARTS = (1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000)
GPU_LIMIT = 12 * 1024**3


def pack_gradients(
    names: list[str], grads: Sequence[torch.Tensor | None]
) -> dict[str, torch.Tensor]:
    """Pack gradients for shared layers and the three requested output row ranges."""
    by_name = dict(zip(names, grads, strict=True))

    def pack(selected: Sequence[str], rows: slice | None = None) -> torch.Tensor:
        pieces = []
        for name in selected:
            grad = by_name[name]
            if grad is not None:
                pieces.append(
                    (grad if rows is None else grad[rows]).detach().reshape(-1).cpu().double()
                )
        return torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.float64)

    inputs = [n for n in names if n.startswith("inp.")]
    gru = [n for n in names if n.startswith("gru.")]
    hidden = [n for n in names if n.startswith(("head.0.", "head.extra."))]
    output = [n for n in names if n.startswith("head.2.")]
    return {
        "input": pack(inputs),
        "gru": pack(gru),
        "head_hidden": pack(hidden),
        "shared": pack(inputs + gru + hidden),
        "head_rows_0_2": pack(output, slice(0, 2)),
        "head_rows_2_21": pack(output, slice(2, 21)),
        "head_rows_21_23": pack(output, slice(21, 23)),
    }


def vnorm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value))


def vcos(left: torch.Tensor, right: torch.Tensor) -> float | None:
    scale = vnorm(left) * vnorm(right)
    return None if scale == 0 else float(torch.dot(left, right) / scale)


def measure(
    record: dict[str, Any],
    inputs: torch.Tensor,
    y: torch.Tensor,
    labels: tuple[torch.Tensor, ...],
    start: int,
    deadline: float,
) -> dict[str, Any]:
    """Replay a detached causal prefix and measure gradients for one C=1000 chunk."""
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic exceeded the 300-second bound")
    model, state = record["model"], None
    with torch.no_grad():
        for prefix in range(0, start, CHUNK):
            _, state = forward_chunk(
                model, inputs, slice(prefix, min(prefix + CHUNK, start)), state
            )
            state = detach_state(state)
    out, _ = forward_chunk(model, inputs, slice(start, start + CHUNK), state)
    valid, spread, branch, classes, nx_rows, nx_up, nx_ok = labels
    sl, target = slice(start, start + CHUNK), y[:, start : start + CHUNK]
    mask = valid[:, sl]
    main = weighted_mse_loss(out[..., :2], target, mask)
    struct = 0.3 * structural_loss(
        out, spread[:, sl], branch[:, sl], classes[:, sl], mask.float(), 2, N_SP_AUX
    )
    if not bool(mask.all()):
        raise ValueError("chosen window unexpectedly includes warmup rows")
    nw = nx_ok[:, sl].float() * mask.float()
    nmse = (out[..., 21].float() - (nx_rows[:, sl] - 3.0) / 2.0).square()
    nbce = F.binary_cross_entropy_with_logits(out[..., 22].float(), nx_up[:, sl], reduction="none")
    nxt = 0.3 * ((nmse + nbce) * nw).sum() / nw.sum().clamp_min(1e-8)
    losses = (main, struct, nxt, main + struct + nxt)
    named = [
        (n, p)
        for n, p in model.named_parameters()
        if n.startswith(("inp.", "gru.", "head.0.", "head.extra.", "head.2."))
    ]
    if len(named) != sum(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("gradient groups do not cover every trainable model parameter")
    names, params = [x[0] for x in named], [x[1] for x in named]
    gradients = [
        pack_gradients(
            names, torch.autograd.grad(loss, params, retain_graph=i < 3, allow_unused=True)
        )
        for i, loss in enumerate(losses)
    ]
    group_stats = {}
    for group in gradients[0]:
        g0, g1, g2, gc = (g[group] for g in gradients)
        aux = g1 + g2
        group_stats[group] = {
            "norm": {
                name: vnorm(g[group])
                for name, g in zip(
                    ("main_wmse", "weighted_struct", "weighted_next", "combined"),
                    gradients,
                    strict=True,
                )
            },
            "cosine": {
                "main_vs_struct": vcos(g0, g1),
                "main_vs_next": vcos(g0, g2),
                "struct_vs_next": vcos(g1, g2),
                "main_vs_aux_sum": vcos(g0, aux),
            },
            "aux_to_main_norm_ratio": vnorm(aux) / vnorm(g0) if vnorm(g0) else None,
        }
    residual = (
        gradients[3]["shared"]
        - gradients[0]["shared"]
        - gradients[1]["shared"]
        - gradients[2]["shared"]
    )
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("autograd.grad unexpectedly populated parameter .grad")
    clipped = target.float().clamp(-2, 2)
    return {
        "loss": {
            "main_wmse": float(main.detach()),
            "weighted_struct": float(struct.detach()),
            "weighted_next": float(nxt.detach()),
            "combined": float(losses[3].detach()),
            "main_weight_denominator": float((clipped.abs() * mask[..., None]).sum()),
            "structural_row_denominator": int(mask.sum()),
            "next_row_denominator": int(nw.sum()),
            "main_valid_rows": int(mask.sum()),
            "next_valid_rows": int(nw.sum()),
            "scored_mask_used": False,
        },
        "gradient": {
            "groups": group_stats,
            "shared_sum_relative_residual": vnorm(residual)
            / max(vnorm(gradients[3]["shared"]), 1e-30),
        },
    }
