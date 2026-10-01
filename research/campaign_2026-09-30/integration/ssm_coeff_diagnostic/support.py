"""Diagnostic literal recurrence with explicit rounded transition tensors."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_HASH = "227a5b8fe8ada7b34d61463ad2f022a0d802011aa985b215ef880d5a469c1756"
ATOL, RTOL = 2e-5, 2e-4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coefficients(model) -> list[dict[str, torch.Tensor]]:
    """Read actual frozen model operations; preserve exactly the returned FP32 bits."""
    result = []
    for block in model.blocks:
        scalar, a, b = block.transition_coefficients()
        log_rho, scale = block.dynamics()
        result.append({"scalar_rho": scalar, "rotation_a": a, "rotation_b": b, "scale": scale,
                       "half_life": block.half_lives(), "log_rho": log_rho,
                       "rho": log_rho.exp(), "phase_cos": block.phase.cos(), "phase_sin": block.phase.sin()})
    return result


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: cpu_tree(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    error = (left - right).abs()
    bound = ATOL + RTOL * right.abs()
    ratio = error / bound
    indices = (ratio > 1).nonzero()
    return {"max_abs": float(error.max()), "max_bound_ratio": float(ratio.max()), "violations": len(indices),
            "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
            "violation_coordinates": [{"index": index.tolist(), "left": float(left[tuple(index)]), "right": float(right[tuple(index)]), "ratio": float(ratio[tuple(index)])} for index in indices]}


def literal(model, x: torch.Tensor, values: list[dict[str, torch.Tensor]], fixed_drives=None):
    """Use explicit scalar/real2x2 A and scale, with the frozen nonlinear maps."""
    z = F.gelu(F.linear((x - model.mean) / model.std, model.inp.weight, model.inp.bias))
    u, final, traces = z, [], []
    for i, (block, coeff) in enumerate(zip(model.blocks, values, strict=True)):
        scale = torch.cat([coeff["scale"][:64], coeff["scale"][64:].repeat_interleave(2)])
        drive = F.linear(u, block.inject.weight) * scale if fixed_drives is None else fixed_drives[i]
        state, states = model.reset(x.shape[0])[i], []
        for row in drive.unbind(1):
            pair = state[:, 64:].reshape(-1, 32, 2)
            re, im = pair.unbind(-1)
            a, b = coeff["rotation_a"], coeff["rotation_b"]
            rotated = torch.stack([re * a - im * b, re * b + im * a], dim=-1)
            state = torch.cat([state[:, :64] * coeff["scalar_rho"], rotated.flatten(1)], 1) + row
            states.append(state)
        timeline = torch.stack(states, 1)
        direct = F.linear(u, block.direct.weight)
        read = F.linear(timeline, block.readout.weight)
        gate = torch.sigmoid(F.linear(u, block.gate.weight, block.gate.bias))
        traces.append({"input": u, "drive": drive, "timeline": timeline})
        u = u + gate * F.gelu(read + direct)
        final.append(state)
    hidden = torch.cat([u, z], -1)
    output = F.linear(F.gelu(F.linear(hidden, model.head[0].weight, model.head[0].bias)), model.head[2].weight, model.head[2].bias)
    return output, tuple(final), traces
