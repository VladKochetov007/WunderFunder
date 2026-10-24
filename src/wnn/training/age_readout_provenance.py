"""Source, cache, parent and endpoint identities for age-readout artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from .age_readout_artifacts import file_sha256
from .age_readout_runtime import CUDA_ALLOCATOR_LIMIT_BYTES

SOURCE_FILES = (
    "src/wnn/conditional_readout.py",
    "src/wnn/features.py",
    "src/wnn/research_features.py",
    "src/wnn/splits.py",
    "src/wnn/training/checkpoints.py",
    "src/wnn/training/online_relationship_cache.py",
    "src/wnn/training/online_relationship_cache_validation.py",
    "src/wnn/training/online_relationship_sources.py",
    "src/wnn/training/age_readout_artifacts.py",
    "src/wnn/training/age_readout_cache.py",
    "src/wnn/training/age_readout_checkpoint.py",
    "src/wnn/training/age_readout_cohort.py",
    "src/wnn/training/age_readout_features.py",
    "src/wnn/training/age_readout_fit.py",
    "src/wnn/training/age_readout_model.py",
    "src/wnn/training/age_readout_provenance.py",
    "src/wnn/training/age_readout_runtime.py",
    "src/wnn/training/age_readout_runner.py",
    "src/wnn/training/age_readout_portable.py",
    "src/wnn/training/age_readout_portable_builder.py",
    "src/wnn/training/age_readout_portable_identity.py",
    "src/wnn/training/age_readout_portable_paths.py",
    "src/wnn/training/age_readout_portable_sources.py",
    "src/wnn/training/age_readout_portable_validate.py",
    "src/wnn/training/age_readout_recipe.py",
    "src/wnn/training/age_readout_cli.py",
    "src/wnn/data.py",
    "scripts/age_readout_screen.py",
    "src/wnn/block_memory_attention.py",
    "src/wnn/event_content.py",
    "src/wnn/gmetric.py",
    "src/wnn/i1_history.py",
    "src/wnn/i1_history_scan.py",
    "src/wnn/i1_history_streaming.py",
    "src/wnn/kalman_group_features.py",
    "src/wnn/metric.py",
    "src/wnn/moment_geometry/analysis.py",
    "src/wnn/moment_geometry/features.py",
    "src/wnn/moment_geometry/online.py",
    "src/wnn/nn.py",
    "src/wnn/numeric_ple.py",
    "src/wnn/optim.py",
    "src/wnn/outcomes.py",
    "src/wnn/privileged.py",
    "src/wnn/propensity.py",
    "src/wnn/quantization.py",
    "src/wnn/sentinel.py",
    "src/wnn/specialist.py",
    "src/wnn/structured_state.py",
    "src/wnn/training/averaging.py",
    "src/wnn/training/cache_identity.py",
    "src/wnn/training/environment.py",
    "src/wnn/training/evaluation.py",
    "src/wnn/training/forward.py",
    "src/wnn/training/i1_inputs.py",
    "src/wnn/training/inputs.py",
    "src/wnn/training/membership.py",
    "src/wnn/training/model_validation.py",
    "src/wnn/training/models.py",
    "src/wnn/training/online_relationship_data_writer.py",
    "src/wnn/training/online_relationship_parent_batch.py",
    "src/wnn/training/targets.py",
    "src/wnn/training_dynamics.py",
    "src/wnn/training_factorized.py",
    "src/wnn/training_qat.py",
    "src/wnn/training_state.py",
)


def build_identity(
    repo_root: Path,
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    fit_cache: Any,
    receipt: dict[str, object],
    device: str,
) -> dict[str, object]:
    """Bind resumable checkpoints to recipe, source, cache and device identities."""
    mean = np.asarray(fit_cache.metadata["parent_mean"], dtype="<f4")
    std = np.asarray(fit_cache.metadata["parent_std"], dtype="<f4")
    sources = {path: file_sha256(repo_root / path) for path in SOURCE_FILES}
    return {
        "experiment_id": recipe["experiment_id"],
        "recipe_sha256": recipe_sha,
        "source_cache_recipe_sha256": receipt["source_cache_recipe_sha256"],
        "fold": int(fold["fold"]),
        "seed": int(fold["seed"]),
        "fit_sidecar_sha256": receipt["fit_cache"]["sidecar_sha256"],
        "fit_payload_sha256": receipt["fit_cache"]["payload_sha256"],
        "evaluation_sidecar_sha256": receipt["evaluation_cache"]["sidecar_sha256"],
        "evaluation_payload_sha256": receipt["evaluation_cache"]["payload_sha256"],
        "parent_manifest_sha256": receipt["parent_manifest_sha256"],
        "parent_prediction_sha256": receipt["parent_prediction_sha256"],
        "parent_mean_sha256": array_sha(mean),
        "parent_std_sha256": array_sha(std),
        "head_device": device,
        "source_sha256": sources,
    }


def build_provenance(
    *,
    recipe: dict[str, Any],
    cell: dict[str, Any],
    fold: dict[str, Any],
    arm: str,
    fit_cache: Any,
    eval_cache: Any,
    cache_receipt: dict[str, object],
    checkpoint: dict[str, object],
    prediction: dict[str, object],
    identity: dict[str, object],
    runtime: dict[str, object],
    fit_records: list[dict[str, object]],
) -> dict[str, object]:
    """Build one independent candidate lineage record without grade values."""
    parent = fold["parent"]
    return {
        "schema": "wnn.age-readout-provenance.v1",
        "run": cell["run"],
        "experiment_id": recipe["experiment_id"],
        "recipe_sha256": identity["recipe_sha256"],
        "source_cache_recipe": {
            "path": cache_receipt["source_cache_recipe_path"],
            "sha256": cache_receipt["source_cache_recipe_sha256"],
        },
        "fold": int(fold["fold"]),
        "seed": int(fold["seed"]),
        "arm": arm,
        "fit_cache": {**cache_receipt["fit_cache"], "row_refs": list(fit_cache.row_refs)},
        "evaluation_cache": {
            **cache_receipt["evaluation_cache"],
            "row_refs": list(eval_cache.row_refs),
        },
        "parent_endpoint": {
            "backend": "cpu",
            "precision": "strict_fp32",
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "provenance_sha256": parent["provenance_sha256"],
            "prediction_path": cache_receipt["parent_prediction_path"],
            "prediction_sha256": cache_receipt["parent_prediction_sha256"],
            "manifest_path": cache_receipt["parent_manifest_path"],
            "manifest_sha256": cache_receipt["parent_manifest_sha256"],
            "ancestry": parent,
        },
        "parent_normalization": cache_receipt["parent_normalization"],
        "checkpoint": checkpoint,
        "prediction": prediction,
        "source_sha256": identity["source_sha256"],
        "fit_source_sha256": identity["source_sha256"],
        "features": recipe["features"],
        "readout": recipe["readout"],
        "runtime": runtime,
        "fit_records": fit_records,
        "evaluation_read_contract": evaluation_read_contract(),
        "target_cache_opened": False,
        "cache_payload_hash_verified": cache_receipt["payload_hash_verified"],
    }


def build_prediction_sidecar(
    *,
    recipe: dict[str, Any],
    cell: dict[str, Any],
    fold: dict[str, Any],
    arm: str,
    recipe_sha: str,
    cache_receipt: dict[str, object],
    eval_cache: Any,
    prediction: dict[str, object],
    checkpoint: dict[str, object],
    provenance_path: Path,
    provenance_sha: str,
) -> dict[str, object]:
    """Bind a prediction array to complete ordered coverage and its lineage file."""
    return {
        "schema": "wnn.age-readout-prediction.v1",
        "run": cell["run"],
        "experiment_id": recipe["experiment_id"],
        "recipe_sha256": recipe_sha,
        "source_cache_recipe_sha256": cache_receipt["source_cache_recipe_sha256"],
        "fold": int(fold["fold"]),
        "seed": int(fold["seed"]),
        "arm": arm,
        "prediction_path": prediction["path"],
        "prediction_sha256": prediction["sha256"],
        "prediction_shape": prediction["shape"],
        "prediction_dtype": prediction["dtype"],
        "evaluation_row_refs": list(eval_cache.row_refs),
        "evaluation_membership_sha256": eval_cache.metadata["membership_sha256"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "provenance_path": str(provenance_path),
        "provenance_sha256": provenance_sha,
        "parent_prediction_path": cache_receipt["parent_prediction_path"],
        "parent_prediction_sha256": cache_receipt["parent_prediction_sha256"],
        "parent_manifest_path": cache_receipt["parent_manifest_path"],
        "parent_manifest_sha256": cache_receipt["parent_manifest_sha256"],
        "evaluation_read_contract": evaluation_read_contract(),
    }


def evaluation_read_contract() -> dict[str, bool]:
    """Return the exact declared fields read while building held-out predictions."""
    return {
        "raw_inputs": True,
        "cached_parent_predictions": True,
        "targets": False,
        "row_weights": False,
        "scoring_masks": False,
        "prediction_request_flags": False,
    }


def runtime_identity(torch_module: Any, device: str) -> dict[str, object]:
    """Describe selected readout device and strict FP32 policy."""
    return {
        "device": device,
        "precision": "strict_fp32",
        "torch_version": torch_module.__version__,
        "torch_threads": torch_module.get_num_threads(),
        "interop_threads": torch_module.get_num_interop_threads(),
        "tf32_matmul": torch_module.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch_module.backends.cudnn.allow_tf32,
        "autocast": False,
        "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_memory": _cuda_memory_identity(torch_module, device),
        "dtype": "float32",
    }


def _cuda_memory_identity(torch_module: Any, device: str) -> dict[str, int | None]:
    if not device.startswith("cuda:"):
        return {
            "allocator_cap_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    index = int(device.partition(":")[2])
    torch_module.cuda.synchronize(index)
    total_bytes = int(torch_module.cuda.get_device_properties(index).total_memory)
    return {
        "allocator_cap_bytes": min(CUDA_ALLOCATOR_LIMIT_BYTES, total_bytes),
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(index)),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(index)),
    }


def array_sha(array: Any) -> str:
    """Hash exact little-endian float32 normalization bytes."""
    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest()
