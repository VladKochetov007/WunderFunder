"""Small B300-only engineering checks for PR81 model components."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from impact81.bindings import (
    RUNTIME_COMMIT,
    SPLIT_SHA256,
    canonical,
    digest,
    source_identity,
    write_new,
)
from impact81.receiver_probes import (
    CHUNK_ATOL,
    CHUNK_RTOL,
    _check_gradients,
    _check_sequence_behavior,
    _check_state_roundtrip,
    _max_abs,
    _predict,
    activate_branches,
)

# JSON configs and receipts are heterogeneous runtime-boundary metadata.
SEED = 204101
VARIANTS = ("plain", "ple", "groups", "target_adapters")


def _config(artifact: dict[str, Any], artifact_sha256: str, variant: str) -> dict[str, Any]:
    cell = {"variant": variant, "seed": SEED}
    return {
        "model": {
            "architecture": "transformer",
            "hidden": 48,
            "layers": 2,
            "heads": 4,
            "ff_multiplier": 2.0,
            "context_rows": 16,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": 0,
            "precision_mode": "bf16_bf16x9",
            "attention_backend": "efficient",
            "compile_blocks": False,
        },
        "impact81": {
            "schema": "impact81.model.v1",
            "cell": cell,
            "preprocessing": artifact,
            "preprocessing_sha256": artifact_sha256,
        },
    }


def _build(
    builder: Callable[[dict[str, Any], torch.device], nn.Module],
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    torch.manual_seed(SEED)
    model = builder(config, device)
    model.eval()
    return model


def run_checks(preprocessing_path: Path, output: Path, runtime_root: Path) -> None:
    """Check model, gradients, causal streaming, and state reload on an actual B300.

    Args:
        preprocessing_path: Prepared PR81 train-only normalization/knot JSON.
        output: New path for the immutable JSON check receipt.
        runtime_root: Exact external runtime checkout admitted by the receiver worker.

    Raises:
        FileExistsError: If a receipt already exists at ``output``.
        RuntimeError: If the device is not B300 or any engineering check fails.
        ValueError: If preprocessing metadata or output paths are invalid.
    """
    if sys.platform != "linux":
        raise RuntimeError("PR81 model checks are restricted to the Linux receiver")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"receiver check receipt already exists: {output}")
    if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") != "1":
        raise RuntimeError("CUBLAS_EMULATE_SINGLE_PRECISION=1 is required for BF16/BF16x9")
    if not torch.cuda.is_available():
        raise RuntimeError("receiver checks require an actual CUDA B300 device")
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    device_name = torch.cuda.get_device_name(device)
    if "B300" not in device_name.upper():
        raise RuntimeError(f"receiver checks require NVIDIA B300; found {device_name!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected B300 runtime does not report BF16 support")
    capability = list(torch.cuda.get_device_capability(device))

    if preprocessing_path.is_symlink() or not preprocessing_path.is_file():
        raise ValueError("preprocessing must be a regular, nonsymlink JSON file")
    preprocessing_path = preprocessing_path.resolve(strict=True)
    from impact81.preprocessing import load_preprocessing

    artifact = load_preprocessing(preprocessing_path)
    fit_identity = artifact.get("fit_identity")
    if not isinstance(fit_identity, dict) or fit_identity.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("preprocessing artifact is not bound to the pinned official split")
    members_sha256 = fit_identity.get("fit_members_sha256")
    if (
        not isinstance(members_sha256, str)
        or len(members_sha256) != 64
        or any(character not in "0123456789abcdef" for character in members_sha256)
    ):
        raise ValueError("preprocessing fit membership hash is missing or malformed")
    artifact_sha256 = digest(preprocessing_path)
    from impact81.modeling import build_model

    runtime_root = runtime_root.resolve(strict=True)
    if not (runtime_root / "src/wnn/training/sequence_models.py").is_file():
        raise RuntimeError("source identity root is not the admitted WNN runtime")
    identity = source_identity(runtime_root)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    probe = torch.randn((2, 48, 112), generator=generator, dtype=torch.float32).to(device)

    configurations = {variant: _config(artifact, artifact_sha256, variant) for variant in VARIANTS}
    models = {variant: _build(build_model, configurations[variant], device) for variant in VARIANTS}
    checks: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        reference, _ = _predict(models["plain"], probe)
        if not bool(torch.isfinite(reference).all()):
            raise RuntimeError("plain model produced non-finite receiver predictions")
        for variant in VARIANTS[1:]:
            prediction, _ = _predict(models[variant], probe)
            max_abs = _max_abs(reference, prediction)
            if not torch.equal(reference, prediction):
                raise RuntimeError(
                    f"zero-initialized {variant} changed the common prediction: max_abs={max_abs}"
                )
            checks[variant] = {"initial_prediction_equal": True, "initial_max_abs": max_abs}

    for variant in VARIANTS[1:]:
        checks[variant]["branch_gradients"] = _check_gradients(models[variant], variant, probe)
    for variant in VARIANTS:
        checks.setdefault(variant, {})["active_branch_probe"] = activate_branches(
            models[variant], variant
        )
        checks[variant]["sequence_behavior"] = _check_sequence_behavior(models[variant], probe)
        checks[variant]["strict_state_roundtrip"] = _check_state_roundtrip(
            models[variant], configurations[variant], build_model, probe, device
        )
        checks[variant]["config_sha256"] = canonical(configurations[variant])

    from impact81.receiver_depth_checks import run_depth_checks
    from impact81.receiver_future_checks import run_future_checks
    from impact81.receiver_loop_checks import run_loop_checks
    from impact81.receiver_objective_checks import run_objective_checks

    depth_checks = run_depth_checks(artifact, artifact_sha256, device)
    objective_checks = run_objective_checks(device)
    loop_checks = run_loop_checks(artifact, artifact_sha256, device)
    future_checks = run_future_checks(artifact, artifact_sha256, device)
    torch.cuda.synchronize(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "impact81.receiver-checks.v1",
        "status": "passed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "scope": "small-model engineering updates; no full profile, scientific fit, or quality score",
        "runtime_commit": RUNTIME_COMMIT,
        "source_identity": identity,
        "preprocessing_sha256": artifact_sha256,
        "precision_mode": "bf16_bf16x9",
        "model_config": configurations["plain"]["model"],
        "preprocessing": {
            "schema": artifact["schema"],
            "split_sha256": SPLIT_SHA256,
            "fit_members_sha256": members_sha256,
        },
        "device": {
            "name": device_name,
            "index": device_index,
            "capability": capability,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "bf16_supported": True,
            "cublas_emulate_single_precision": os.environ["CUBLAS_EMULATE_SINGLE_PRECISION"],
        },
        "probe": {"shape": list(probe.shape), "dtype": str(probe.dtype)},
        "tolerances": {
            "initial_prediction": "bitwise equal",
            "suffix_invariance": "bitwise equal",
            "fresh_state_repeat": "bitwise equal",
            "chunk_carry": {"atol": CHUNK_ATOL, "rtol": CHUNK_RTOL},
            "strict_state_roundtrip": "bitwise equal",
        },
        "variants": checks,
        "extension_cards": ["04", "05", "06", "07"],
        "depth_checks": depth_checks,
        "objective_checks": objective_checks,
        "control_engine_parity": loop_checks,
        "future_checks": future_checks,
    }
    write_new(output, receipt)
