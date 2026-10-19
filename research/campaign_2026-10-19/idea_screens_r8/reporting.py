"""Compact scalar comparisons and B-only model archive for R8."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from idea_screens_r7.model_math import (
    ADAM_EPS,
    BATCH_SIZE,
    BETAS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

from .model import ConstantVarianceMLP, constant_head_state

TRAINING_RECIPE = {
    "seed": 19_031,
    "input_dim": 224,
    "hidden_dim": 64,
    "activation": "GELU",
    "head_widths": [64, 2],
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "batches_per_epoch": 63,
    "expected_updates": 756,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "betas": list(BETAS),
    "epsilon": ADAM_EPS,
    "weight_decay": WEIGHT_DECAY,
    "foreach": False,
    "fused": False,
    "global_gradient_clip_norm": 1.0,
    "scheduler": None,
    "early_stopping": False,
    "objective": "sum(w*(exp(-s)*(mu-yclip)^2+s))/sum(w)",
    "target_clip": [-2.0, 2.0],
    "logvariance_clip": [-4.0, 4.0],
    "weight": "abs(yclip)",
    "reduction": "one shared denominator across both targets and all rows",
}


def archive_one_model(
    path: Path,
    model: ConstantVarianceMLP,
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    """Write the final single-model state and fit normalizer to a private NPZ."""
    arrays: dict[str, np.ndarray] = {
        "normalizer_mean": mean,
        "normalizer_scale": scale,
    }
    arrays.update(
        {
            f"model__{name.replace('.', '__')}": tensor.detach().cpu().numpy()
            for name, tensor in model.state_dict().items()
        }
    )
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path_name": path.name,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "array_names": sorted(arrays),
        "predictions_in_archive": False,
    }


def reference_contrasts(
    score: dict[str, object], reference: dict[str, object]
) -> dict[str, object]:
    """Compare the one R8 score with stored R7 scalar and drop-one evidence."""
    endpoint = score["official_metric"]
    wp = float(endpoint["wp"])
    stored_candidate_wp = float(reference["heteroskedastic_wp"])
    stored_control_wp = float(reference["control_wp"])
    current_drop = score["component_drop_one_scores"]
    candidate_drop = reference["heteroskedastic_drop_one"]
    control_drop = reference["control_drop_one"]
    if set(current_drop) != set(candidate_drop) or set(current_drop) != set(control_drop):
        raise ValueError("R7 and R8 component drop-one membership differs")
    per_component: dict[str, dict[str, float]] = {}
    for component in sorted(current_drop, key=int):
        r8_drop_wp = float(current_drop[component]["wp"])
        per_component[component] = {
            "primary_stored_r7_heteroskedastic_minus_r8_constant_wp": (
                float(candidate_drop[component]["wp"]) - r8_drop_wp
            ),
            "secondary_r8_constant_minus_stored_r7_control_wp": (
                r8_drop_wp - float(control_drop[component]["wp"])
            ),
        }
    return {
        "stored_r7_result_sha256": str(reference["result_sha256"]),
        "stored_r7_control_wp": stored_control_wp,
        "stored_r7_heteroskedastic_wp": stored_candidate_wp,
        "r8_constant_variance_wp": wp,
        "primary_stored_r7_heteroskedastic_minus_r8_constant_wp": stored_candidate_wp - wp,
        "secondary_r8_constant_minus_stored_r7_control_wp": wp - stored_control_wp,
        "drop_one_contrasts": per_component,
    }


def runtime_record(cuda_visibility: str) -> dict[str, object]:
    """Capture fixed execution settings after enforcing CPU FP32 and one thread."""
    if cuda_visibility not in ("", "-1") or torch.cuda.is_initialized():
        raise RuntimeError("R8 requires hidden CUDA and an uninitialized CUDA runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("R8 requires the default FP32 dtype")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise RuntimeError("R8 does not permit autocast")
    blas = {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    if any(value != "1" for value in blas.values()):
        raise RuntimeError("R8 requires all BLAS/OpenMP thread limits to equal one")
    return {
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "model_dtype": "float32",
        "cuda_visible_devices": cuda_visibility,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_autocast_enabled": bool(torch.is_autocast_enabled()),
        "cpu_autocast_enabled": bool(torch.is_autocast_enabled("cpu")),
        "blas_thread_environment": blas,
    }


def result_head_record(model: ConstantVarianceMLP) -> dict[str, object]:
    """Return compact constant-head evidence for the result payload."""
    return constant_head_state(model)
