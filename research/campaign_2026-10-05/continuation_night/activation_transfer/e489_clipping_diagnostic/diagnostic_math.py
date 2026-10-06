"""Small, non-scoring arithmetic helpers for the E489 clipping diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CLIP_NORM = 1.0
DELTA_RELATIVE_LIMIT, DELTA_MAX_ABS = 0.02, 2e-5
OUTPUT_ATOL, OUTPUT_RTOL = 2e-5, 2e-4
GRAD_ATOL, GRAD_RTOL = 5e-5, 5e-4


def sha256_file(path: Path) -> str:
    """Hash a file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash an array's identity, dtype, shape, and C-order bytes."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_state(state: dict[str, torch.Tensor]) -> str:
    """Hash named initialization tensors in a stable order."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        shape = json.dumps(list(tensor.shape), separators=(",", ":"))
        digest.update(shape.encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _finite_pair(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(f"comparison shape mismatch: {reference.shape} != {candidate.shape}")
    if not (torch.isfinite(reference).all() and torch.isfinite(candidate).all()):
        raise ValueError("nonfinite value in E489 diagnostic comparison")


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    """Record tensor error magnitudes and the frozen elementwise tolerance result."""
    _finite_pair(reference, candidate)
    ref = reference.detach().float()
    delta = candidate.detach().float() - ref
    ref_rms = float(ref.square().mean().sqrt().cpu())
    error_rms = float(delta.square().mean().sqrt().cpu())
    return {
        "reference_rms": ref_rms,
        "error_rms": error_rms,
        "relative_error_rms": error_rms / max(ref_rms, 1e-10),
        "max_abs_error": float(delta.abs().max().cpu()),
        "allclose": bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def raw_global_norm(named_grads: dict[str, torch.Tensor]) -> torch.Tensor:
    """Match E489's L2 norm over the vector of per-parameter L2 norms."""
    if not named_grads:
        raise ValueError("cannot measure an empty gradient set")
    values = list(named_grads.values())
    if not all(bool(torch.isfinite(value).all().item()) for value in values):
        raise ValueError("nonfinite raw gradient")
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(value.detach(), 2) for value in values]), 2
    )


def clipping_record(norm: torch.Tensor, max_norm: float = CLIP_NORM) -> dict[str, float | bool]:
    """Describe PyTorch's exact registered global clipping gate and scale."""
    if not bool(torch.isfinite(norm).item()) or max_norm <= 0:
        raise ValueError("nonfinite norm or invalid clip threshold")
    value = float(norm.detach().cpu())
    return {
        "preclip_global_norm": value,
        "clip_threshold": max_norm,
        "clipping_active": value > max_norm,
        "clip_factor": min(1.0, max_norm / (value + 1e-6)),
    }


def compare_delta_groups(
    reference: dict[str, list[torch.Tensor]], candidate: dict[str, list[torch.Tensor]]
) -> dict[str, dict[str, float | bool]]:
    """Compare optimizer deltas by the original four-matrix plus AdamW groups."""
    if reference.keys() != candidate.keys():
        raise ValueError("optimizer delta families differ")
    report: dict[str, dict[str, float | bool]] = {}
    for family in reference:
        if len(reference[family]) != len(candidate[family]):
            raise ValueError(f"parameter count differs in {family}")
        ref_parts, err_parts = [], []
        max_abs = 0.0
        for ref, got in zip(reference[family], candidate[family], strict=True):
            _finite_pair(ref, got)
            ref_parts.append(ref.detach().float().reshape(-1))
            err = (got - ref).detach().float().reshape(-1)
            err_parts.append(err)
            max_abs = max(max_abs, float(err.abs().max().cpu()))
        ref_all = torch.cat(ref_parts)
        err_all = torch.cat(err_parts)
        reference_rms = float(ref_all.square().mean().sqrt().cpu())
        error_rms = float(err_all.square().mean().sqrt().cpu())
        relative = error_rms / max(reference_rms, 1e-10)
        report[family] = {
            "reference_update_rms": reference_rms,
            "error_update_rms": error_rms,
            "relative_error_rms": relative,
            "max_abs_error": max_abs,
            "relative_limit": DELTA_RELATIVE_LIMIT,
            "max_abs_limit": DELTA_MAX_ABS,
            "original_delta_gate_passed": relative <= DELTA_RELATIVE_LIMIT
            and max_abs <= DELTA_MAX_ABS,
        }
    return report
