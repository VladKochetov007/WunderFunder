from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any


def muon_precision_error(args: argparse.Namespace) -> str | None:
    """Validate explicit Muon orthogonalization and strict-FP32 recipe flags."""
    ortho_dtype = getattr(args, "muon_ortho_dtype", "auto")
    if ortho_dtype not in {"auto", "fp32"}:
        return "--muon-ortho-dtype must be auto or fp32"

    optim = getattr(args, "optim", "adamw")
    switch_target = (getattr(args, "optim_switch", "") or "").partition(":")[2]
    muon_active = optim == "muon" or switch_target == "muon"
    strict = bool(getattr(args, "strict_muon_fp32", False))

    if not muon_active:
        if strict:
            return "--strict-muon-fp32 requires --optim muon or an --optim-switch target of muon"
        if ortho_dtype != "auto":
            return "--muon-ortho-dtype applies only when Muon is active"
        return None

    if strict and getattr(args, "precision", None) != "fp32":
        return "--strict-muon-fp32 requires --precision fp32"
    if strict and ortho_dtype != "fp32":
        return "--strict-muon-fp32 requires --muon-ortho-dtype fp32"
    return None


def muon_resume_policy_error(
    args: argparse.Namespace,
    current_groups: Sequence[Mapping[str, Any]],
    saved_optimizer: Mapping[str, Any],
) -> str | None:
    """Reject a strict-FP32 resume checkpoint that would downgrade Muon NS arithmetic."""
    if not bool(getattr(args, "strict_muon_fp32", False)):
        return None
    saved_groups = saved_optimizer.get("param_groups")
    if not isinstance(saved_groups, list) or len(saved_groups) != len(current_groups):
        return "strict Muon resume optimizer groups are missing or changed"
    for index, (current, saved) in enumerate(zip(current_groups, saved_groups, strict=True)):
        if not isinstance(saved, Mapping):
            return "strict Muon resume optimizer group metadata is malformed"
        planned_muon = bool(current.get("muon", False)) or bool(current.get("switch_muon", False))
        if planned_muon and (
            current.get("muon_fp32_ortho") is not True or saved.get("muon_fp32_ortho") is not True
        ):
            return (
                "strict Muon resume requires FP32 orthogonalization in optimizer group "
                f"{index}; checkpoint policy differs"
            )
    return None
