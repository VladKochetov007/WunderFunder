"""Compact R13 source, result, and private model-archive records."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .model_arms import ARM_NAMES, R13MeanVarianceMLP


def archive_models(
    path: Path,
    models: dict[str, R13MeanVarianceMLP],
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    """Write all three final states and fit-only normalization to a private NPZ."""
    arrays: dict[str, np.ndarray] = {
        "normalizer_mean": np.asarray(mean, dtype=np.float32),
        "normalizer_scale": np.asarray(scale, dtype=np.float32),
    }
    for arm in ARM_NAMES:
        if arm not in models:
            raise ValueError("R13 archive needs exactly the three fixed model arms")
        for name, tensor in models[arm].state_dict().items():
            arrays[f"{arm}__{name.replace('.', '__')}"] = tensor.detach().cpu().numpy()
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


def build_source_record(
    *,
    preparation_sha256: str,
    source_pins: dict[str, str],
    runtime_imports: dict[str, str],
    sample_manifest: dict,
    sample_sha256: str,
    membership_sha256: str,
    sample_array_count: int,
    panel_stats: dict,
) -> dict[str, object]:
    """Describe bound code and sample inputs without including parameter arrays."""
    eval_records = [item for item in sample_manifest["records"] if item["role"] == "eval"]
    return {
        "schema": "nf19.r13.source.v1",
        "preparation_manifest_sha256": preparation_sha256,
        "source_pins": source_pins,
        "runtime_imports": runtime_imports,
        "sample_sha256": sample_sha256,
        "membership_sha256": membership_sha256,
        "split_sha256": sample_manifest["split_sha256"],
        "sample_records": len(sample_manifest["records"]),
        "sample_arrays_hashed_and_mapped_for_integrity": sample_array_count,
        "evaluation_records": len(eval_records),
        "evaluation_target_files_hashed_and_mapped_for_integrity": True,
        "evaluation_target_values_used_for_fit_normalization_or_selection": False,
        "evaluation_target_values_used_for_final_scoring": True,
        "panel_stats": panel_stats,
    }


def build_result(
    *,
    fit: dict[str, object],
    initialization: dict[str, object],
    training: dict[str, dict[str, object]],
    scores: dict[str, object],
    archive: dict[str, object],
    runtime: dict[str, object],
    source_sha256: str,
) -> dict[str, object]:
    """Build the bounded scalar comparison record for the fixed R13 screen."""
    return {
        "schema": "nf19.r13.result.v1",
        "design": "immutable first-99-row mean context vs current and zero side controls",
        "source_sha256": source_sha256,
        "precision": "FP32 model/features/targets; fit-only normalizer uses FP64 moments then FP32",
        "fit_components": 64,
        "evaluation_components": 32,
        "fit_rows_per_arm": int(len(fit["targets"])),
        "evaluation_scored_rows": int(scores["eval_scored_rows"]),
        "updates_per_arm": 756,
        "total_updates": 2268,
        "row_exposures_per_arm": 381_696,
        "total_row_exposures": 1_145_088,
        "initialization": initialization,
        "training": training,
        "scores": scores,
        "prospective_gate": {
            "definition": "warmup99 minus current_bypass and warmup99 minus zero each exceed +0.005 pooled WP",
            "margin_wp": 0.005,
            "passed": bool(scores["prospective_exploratory_gate_passed"]),
            "is_significance_test": False,
        },
        "model_archive": archive,
        "runtime": runtime,
        "limits": {
            "single_exposed_R3_panel": True,
            "evaluation_used_once_after_all_training": True,
            "no_epoch_selection_from_evaluation": True,
            "no_retries_or_hyperparameter_search": True,
            "not_a_GRU_memory_or_population_claim": True,
        },
    }
