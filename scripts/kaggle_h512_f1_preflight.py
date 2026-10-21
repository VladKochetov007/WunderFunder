from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from kaggle_h512_f1_recipe import validate_admission, validate_recipe
from kaggle_h512_f1_worker_support import (
    ADMISSION,
    DERIVED,
    EXPORT_RESERVE_SECONDS,
    FINALIZER_RESERVE_SECONDS,
    FIT_BATCHES,
    OUTER_WALL_CAP_SECONDS,
    PROFILE_SLOWDOWN,
    PROPOSAL,
    WORKER_WALL_CAP_SECONDS,
    emit,
    read_object,
    require_profile,
    run_cli,
    validate_small_inputs,
)


def run_profile_preflight(root: Path, started: float) -> dict[str, Any]:
    """Verify inputs, load the canonical split once, and run the finite profile."""
    proposal = read_object(root / PROPOSAL)
    profile_argv, fit_argv = validate_recipe(proposal)
    identities = validate_small_inputs(root, started)
    admission = validate_admission(root / ADMISSION, identities["proposal_sha256"])
    emit("canonical_split_preflight_started", started, torch_loaded="torch" in sys.modules)
    from wnn.splits import load_campaign_split

    if "torch" in sys.modules:
        raise RuntimeError("Torch loaded before source-identity preflight")
    preflight_started = time.monotonic()
    split = load_campaign_split(root / DERIVED, root)
    preflight_seconds = time.monotonic() - preflight_started
    if (
        split.fold != 1
        or split.n_folds != 4
        or len(split.train_groups) != 10_533
        or len(split.valid_groups) != 1_416
        or len(split.evaluation_groups) != 457
    ):
        raise ValueError("Canonical split preflight returned unexpected F1 membership")
    evaluation_order = list(map(int, split.evaluation_groups))
    evaluation_groups = set(evaluation_order)
    evaluation_components = {
        component
        for section, group, component in split.components
        if section == "valid" and group in evaluation_groups
    }
    emit(
        "canonical_split_preflight_completed",
        started,
        preflight_seconds=round(preflight_seconds, 3),
        fold=split.fold,
        train_groups=len(split.train_groups),
        fit_valid_groups=len(split.valid_groups),
        evaluation_groups=len(split.evaluation_groups),
        evaluation_components=len(evaluation_components),
    )
    for identity in identities["raw_sources"].values():
        identity["sha256_check"] = (
            "passed canonical load_campaign_split portable SHA verification before Torch import"
        )
    raw_mount_stats = {name: (root / name).stat().st_mtime_ns for name in identities["raw_sources"]}
    deadline_value = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
    try:
        kernel_deadline = float(deadline_value) if deadline_value is not None else math.nan
    except ValueError as error:
        raise ValueError("Kernel deadline must be a finite monotonic timestamp") from error
    if not math.isfinite(kernel_deadline) or kernel_deadline <= time.monotonic():
        raise TimeoutError("A future finite provider kernel deadline is required")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE=0 is required for strict FP32")
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    emit("profile_started", started, argv=profile_argv, trainer_imported=False)
    profile_elapsed, records = run_cli(root, profile_argv, capture_profile=True)
    step_seconds = require_profile(records)
    tf32_after_profile = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    if any(tf32_after_profile.values()):
        raise RuntimeError("Canonical profile enabled TF32 after strict-FP32 setup")
    if any(
        (root / name).stat().st_mtime_ns != raw_mount_stats[name]
        for name in identities["raw_sources"]
    ):
        raise RuntimeError("Raw source metadata changed during profile")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.device_count() != 1 or not visible or "," in visible:
        raise RuntimeError("Worker must expose exactly one assigned GPU")
    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name:
        raise RuntimeError(f"Requested NvidiaTeslaT4 but received {device_name}")
    device = {
        "requested_machine_shape": "NvidiaTeslaT4",
        "actual_device_name": device_name,
        "visible_count": torch.cuda.device_count(),
        "cuda_visible_devices": visible,
        "provider_deadline_monotonic": os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC"),
    }
    projection = PROFILE_SLOWDOWN * (profile_elapsed + step_seconds * FIT_BATCHES)
    elapsed_after_profile = time.monotonic() - started
    predicted_worker = elapsed_after_profile + projection + EXPORT_RESERVE_SECONDS
    predicted_outer = predicted_worker + FINALIZER_RESERVE_SECONDS
    deadline_remaining = kernel_deadline - time.monotonic()
    predicted_remaining_outer = projection + EXPORT_RESERVE_SECONDS + FINALIZER_RESERVE_SECONDS
    provider_limit = float(admission["remaining_unreserved_provider_gpu_seconds"]) - int(
        admission["safety_reserve_seconds"]
    )
    gate = {
        "profile_elapsed_seconds": round(profile_elapsed, 3),
        "elapsed_worker_seconds_at_profile_gate": round(elapsed_after_profile, 3),
        "profile_step_wall_seconds": step_seconds,
        "full_fit_batches": FIT_BATCHES,
        "fit_projection_seconds_with_1_25_margin": round(projection, 3),
        "export_reserve_seconds_for_CP5_and_SWA345": EXPORT_RESERVE_SECONDS,
        "preflight_seconds": round(preflight_seconds, 3),
        "runner_finalizer_reserve_seconds": FINALIZER_RESERVE_SECONDS,
        "kernel_deadline_remaining_seconds_at_gate": round(deadline_remaining, 3),
        "predicted_remaining_outer_seconds": round(predicted_remaining_outer, 3),
        "fits_actual_kernel_deadline": predicted_remaining_outer <= deadline_remaining,
        "predicted_worker_seconds": round(predicted_worker, 3),
        "predicted_outer_wall_seconds": round(predicted_outer, 3),
        "provider_quota_available_after_safety_reserve": provider_limit,
        "provider_charge_estimate_1x_session_wall_seconds": round(predicted_outer, 3),
        "dual_T4_allocated_device_seconds_conservative_bound": round(2 * predicted_outer, 3),
        "outer_wall_cap_seconds": OUTER_WALL_CAP_SECONDS,
        "worker_wall_cap_seconds": WORKER_WALL_CAP_SECONDS,
    }
    emit(
        "profile_and_conditional_cost_gate",
        started,
        device=device,
        profile=records["profile"],
        update_evidence=records["updates"],
        gate=gate,
    )
    return {
        "proposal": proposal,
        "fit_argv": fit_argv,
        "identities": identities,
        "admission": admission,
        "split": split,
        "evaluation_order": evaluation_order,
        "records": records,
        "tf32_flags_after_profile": tf32_after_profile,
        "gate": gate,
        "device": device,
        "preflight_seconds": preflight_seconds,
        "raw_mount_stats": raw_mount_stats,
        "predicted_worker_exceeds_cap": predicted_worker > WORKER_WALL_CAP_SECONDS,
        "predicted_outer_exceeds_cap": predicted_outer > OUTER_WALL_CAP_SECONDS,
        "predicted_outer_exceeds_quota": predicted_outer > provider_limit,
        "predicted_outer_exceeds_kernel_deadline": predicted_remaining_outer > deadline_remaining,
    }
