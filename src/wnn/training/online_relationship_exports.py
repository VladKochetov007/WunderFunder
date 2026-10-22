"""Serialize online relationship readout checkpoints and grader endpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.splits import file_sha256
from wnn.training.online_relationship_cache import SequenceCache
from wnn.training.online_relationship_sources import read_json, resolve


def write_candidate(
    root: Path,
    recipe: dict[str, Any],
    recipe_path: Path,
    recipe_sha: str,
    fold: dict[str, Any],
    cell: dict[str, Any],
    fit_cache: SequenceCache,
    eval_cache: SequenceCache,
    parent_manifest: dict[str, Any],
    normalizer: dict[str, object],
    normalizer_sha: str,
    result: Any,
) -> dict[str, object]:
    """Write checkpoint, provenance and the grader's frozen prediction contract."""
    output_dir = Path(cell["output"])
    if not output_dir.is_relative_to(Path(recipe["resource_contract"]["scratch"])):
        raise ValueError("candidate endpoints must be under physical scratch")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_ep2.pt"
    prediction_path = output_dir / "prediction.npy"
    provenance_path = output_dir / "provenance.json"
    sidecar_path = output_dir / "prediction.json"
    if any(
        path.exists() for path in (checkpoint_path, prediction_path, provenance_path, sidecar_path)
    ):
        raise FileExistsError(f"candidate output already exists: {output_dir}")
    checkpoint = {
        "schema": "wnn.online-relationships.readout-checkpoint.v1",
        "run": cell["run"],
        "epoch": 2,
        "model": result.state_dict,
        "metadata": result.metadata,
    }
    checkpoint_sha = _write_torch_create_only(checkpoint_path, checkpoint)
    prediction_sha = _write_predictions_create_only(
        prediction_path, result.evaluation_predictions, len(eval_cache.row_refs)
    )
    fit_sidecar = fit_cache.sidecar_path.resolve()
    eval_sidecar = eval_cache.sidecar_path.resolve()
    provenance = {
        "schema": "wnn.online-relationships.readout-provenance.v1",
        "run": cell["run"],
        "fold": int(fold["fold"]),
        "seed": int(fold["seed"]),
        "lag": int(cell["lag"]),
        "arm": cell["arm"],
        "epoch": 2,
        "recipe": {"path": str(recipe_path), "sha256": recipe_sha},
        "source_manifest": {
            "path": str(resolve(root, Path(fold["split_manifest"]))),
            "sha256": fold["split_manifest_sha256"],
        },
        "fit_membership": {
            "groups": list(fit_cache.row_refs),
            "membership_sha256": fit_cache.metadata["membership_sha256"],
            "selected_component_sha256": _subset_component_sha(root, fold),
            "subset_manifest": str(resolve(root, Path(fold["subset"]["path"]))),
            "subset_manifest_sha256": fold["subset"]["sha256"],
        },
        "caches": {
            "fit": {
                "path": str(fit_cache.data_path),
                "sidecar": str(fit_sidecar),
                "sidecar_sha256": file_sha256(fit_sidecar),
                "payload_sha256": fit_cache.metadata["data_sha256"],
                "scope": "in_sample_fit",
            },
            "evaluation": {
                "path": str(eval_cache.data_path),
                "sidecar": str(eval_sidecar),
                "sidecar_sha256": file_sha256(eval_sidecar),
                "payload_sha256": eval_cache.metadata["data_sha256"],
                "scope": "feature_only_eval",
            },
            "parent_evaluation": {
                "prediction_path": parent_manifest["prediction_path"],
                "prediction_sha256": parent_manifest["prediction_sha256"],
                "sidecar": parent_manifest["manifest_path"],
                "sidecar_sha256": parent_manifest["manifest_sha256"],
                "scope": "feature_only_eval",
            },
        },
        "parent": {
            **fold["parent"],
            "endpoint_precision": fit_cache.metadata["parent"]["endpoint_precision"],
            "endpoint_backend": fit_cache.metadata["parent"]["endpoint_backend"],
            "fit_prediction_scope": "in_sample_fit",
        },
        "propensity": fit_cache.metadata["propensity"],
        "row_weight_contract": {
            "args": recipe["readout"]["row_weight_args"],
            "batch_semantics": recipe["readout"]["row_weight_batch_semantics"],
        },
        "normalization": {
            "parent_mean": result.metadata["parent_mean"],
            "parent_std": result.metadata["parent_std"],
            "relationship_normalizer": normalizer,
            "relationship_normalizer_sha256": normalizer_sha,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "epoch": 2,
        },
        "prediction": {
            "path": str(prediction_path),
            "sha256": prediction_sha,
            "sidecar": str(sidecar_path),
        },
        "source_sha256": source_hashes(root),
        "fit": {"epoch_records": list(result.epoch_records), "metadata": result.metadata},
    }
    write_json_create_only(provenance_path, provenance)
    provenance_sha = file_sha256(provenance_path)
    sidecar = {
        "schema_version": 1,
        "role": "online_residual_evaluation",
        "run": cell["run"],
        "fold": int(fold["fold"]),
        "seed": int(fold["seed"]),
        "lag": int(cell["lag"]),
        "arm": cell["arm"],
        "epoch": 2,
        "recipe_sha256": recipe_sha,
        "source_manifest_sha256": fold["split_manifest_sha256"],
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "normalizer_sha256": normalizer_sha,
        "parent_cache_manifest_sha256": parent_manifest["manifest_sha256"],
        "groups": list(eval_cache.row_refs),
        "shape": [len(eval_cache.row_refs), 20_000, 2],
        "dtype": "<f4",
        "sha256": prediction_sha,
        "provenance_sha256": provenance_sha,
    }
    write_json_create_only(sidecar_path, sidecar)
    return {
        "run": cell["run"],
        "fold": int(fold["fold"]),
        "lag": int(cell["lag"]),
        "arm": cell["arm"],
        "checkpoint_sha256": checkpoint_sha,
        "prediction_sha256": prediction_sha,
        "provenance_sha256": provenance_sha,
        "sidecar": str(sidecar_path),
    }


def _write_predictions_create_only(
    path: Path, predictions: tuple[np.ndarray, ...], expected_count: int
) -> str:
    """Stream ordered sequence predictions to a strict little-endian float32 NPY."""
    if len(predictions) != expected_count:
        raise ValueError("candidate prediction count differs from evaluation cache membership")
    temporary = _temporary_path(path)
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype="<f4", shape=(expected_count, 20_000, 2)
    )
    try:
        for index, prediction in enumerate(predictions):
            value = np.asarray(prediction, dtype=np.float32)
            if value.shape != (20_000, 2) or not np.isfinite(value).all():
                raise ValueError("candidate produced a malformed or nonfinite sequence prediction")
            output[index] = value
        output.flush()
        output._mmap.close()
        digest = file_sha256(temporary)
        _publish(temporary, path)
        return digest
    except BaseException:
        output._mmap.close()
        temporary.unlink(missing_ok=True)
        raise


def _write_torch_create_only(path: Path, value: dict[str, object]) -> str:
    """Save and publish one model checkpoint without replacing an existing run."""
    temporary = _temporary_path(path)
    try:
        torch.save(value, temporary)
        digest = file_sha256(temporary)
        _publish(temporary, path)
        return digest
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _subset_component_sha(root: Path, fold: dict[str, Any]) -> str:
    path = resolve(root, Path(fold["subset"]["path"]))
    return str(read_json(path)["selected_component_sha256"])


def source_hashes(root: Path) -> dict[str, str]:
    """Hash the candidate and data-generation implementation captured by provenance."""
    paths = (
        "src/wnn/online_relationship.py",
        "src/wnn/online_relationship_numba.py",
        "src/wnn/training/online_relationship.py",
        "src/wnn/training/online_relationship_fit.py",
        "src/wnn/training/online_relationship_data.py",
        "src/wnn/training/online_relationship_data_writer.py",
        "src/wnn/training/online_relationship_sources.py",
        "src/wnn/training/online_relationship_cache.py",
        "src/wnn/training/online_relationship_cache_validation.py",
        "src/wnn/training/online_relationship_delivery.py",
        "src/wnn/training/online_relationship_exports.py",
        "src/wnn/training/online_relationship_panel_profile.py",
        "src/wnn/training/online_relationship_panel_profile_artifacts.py",
        "src/wnn/training/membership.py",
        "src/wnn/splits.py",
        "src/wnn/training/inputs.py",
        "src/wnn/training/targets.py",
        "src/wnn/training/models.py",
        "src/wnn/training/evaluation.py",
        "src/wnn/training/environment.py",
        "src/wnn/data.py",
        "src/wnn/features.py",
        "src/wnn/propensity.py",
        "src/wnn/conditional_readout.py",
        "src/wnn/training/checkpoints.py",
        "scripts/online_relationships.py",
    )
    return {name: file_sha256(root / name) for name in paths}


def canonical_json_sha256(value: dict[str, object]) -> str:
    """Hash the grader's stable ASCII canonical JSON representation."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def write_json_create_only(path: Path, value: dict[str, object]) -> None:
    """Create a JSON artifact once, preserving existing evidence on conflict."""
    temporary = _temporary_path(path)
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        _publish(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _temporary_path(path: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    Path(name).unlink()
    return Path(name)


def _publish(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    os.link(source, destination)
    source.unlink(missing_ok=True)
