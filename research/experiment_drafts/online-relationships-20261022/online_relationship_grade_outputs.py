"""Verify candidate prediction bytes and their training provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any  # JSON metadata is validated at runtime.

import numpy as np
from online_relationship_grade_contract import (
    ROWS,
    CandidateArtifacts,
    ParentCacheArtifacts,
    _resolve,
    read_json,
    row_refs_sha256,
)
from online_relationship_grade_core import prediction_memmap, sha256_file
from online_relationship_grade_provenance import verify_candidate_provenance


def verify_candidate(
    root: Path,
    recipe: dict[str, Any],
    recipe_path: Path,
    recipe_sha: str,
    fold: dict[str, Any],
    cell: dict[str, Any],
    subset: dict[str, Any],
    parent_cache: ParentCacheArtifacts,
) -> CandidateArtifacts:
    """Validate one candidate checkpoint, provenance, sidecar and exact eval NPY."""
    run = str(cell["run"])
    output_dir = _resolve(root, cell["output"])
    prediction_path = output_dir / "prediction.npy"
    sidecar_path = output_dir / "prediction.json"
    provenance_path = output_dir / "provenance.json"
    sidecar = read_json(sidecar_path)
    provenance_sha = sha256_file(provenance_path)
    provenance = read_json(provenance_path)
    fold_id = int(cell["fold"])
    groups = [{"split": "valid", "row_group": int(g)} for g in fold["evaluation_groups"]]
    checkpoint_path = output_dir / "model_ep2.pt"
    checkpoint_sha = sha256_file(checkpoint_path)
    split_path = _resolve(root, fold["split_manifest"])
    split_sha = str(fold["split_manifest_sha256"])
    parent_cache_dir = Path(fold["parent_cache"]).resolve()
    parent_cache_manifest = parent_cache_dir / "evaluation_predictions.json"
    parent_cache_manifest_sha = sha256_file(parent_cache_manifest)
    expected_sidecar = {
        "schema_version": 1,
        "role": "online_residual_evaluation",
        "run": run,
        "fold": fold_id,
        "seed": int(cell["seed"]),
        "lag": int(cell["lag"]),
        "arm": str(cell["arm"]),
        "epoch": 2,
        "recipe_sha256": recipe_sha,
        "source_manifest_sha256": split_sha,
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "parent_cache_manifest_sha256": parent_cache_manifest_sha,
        "groups": groups,
        "shape": [len(groups), ROWS, 2],
        "dtype": "<f4",
        "provenance_sha256": provenance_sha,
    }
    for key, expected in expected_sidecar.items():
        if sidecar.get(key) != expected:
            raise ValueError(
                f"{run}: prediction sidecar field {key} differs from the frozen contract"
            )
    normalizer_sha = sidecar.get("normalizer_sha256")
    if not isinstance(normalizer_sha, str) or not normalizer_sha:
        raise ValueError(f"{run}: prediction sidecar lacks normalizer identity")
    prediction_sha = sidecar.get("sha256")
    if not isinstance(prediction_sha, str) or not prediction_sha:
        raise ValueError(f"{run}: prediction sidecar lacks its payload SHA256")
    prediction, coverage = prediction_memmap(
        prediction_path,
        (len(groups), ROWS, 2),
        prediction_sha,
    )
    for index in range(len(groups)):
        if not np.isfinite(prediction[index]).all():
            raise ValueError(f"{run}: prediction contains nonfinite values")
    verify_candidate_provenance(
        root,
        recipe,
        recipe_path,
        recipe_sha,
        fold,
        cell,
        subset,
        parent_cache,
        parent_cache_manifest,
        provenance,
        provenance_path,
        provenance_sha,
        checkpoint_path,
        checkpoint_sha,
        prediction_path,
        sidecar_path,
        prediction_sha,
        normalizer_sha,
        groups,
        split_path,
    )
    identity = {
        "run": run,
        "fold": fold_id,
        "seed": int(cell["seed"]),
        "lag": int(cell["lag"]),
        "arm": str(cell["arm"]),
        "checkpoint_sha256": checkpoint_sha,
        "provenance_sha256": provenance_sha,
        "prediction_sha256": prediction_sha,
        "prediction_sidecar_sha256": sha256_file(sidecar_path),
        "normalizer_sha256": normalizer_sha,
        "parent_cache_manifest_sha256": parent_cache_manifest_sha,
        "evaluation_membership_sha256": row_refs_sha256(groups),
        "fit_membership_sha256": provenance["fit_membership"]["membership_sha256"],
        "source_sha256": dict(sorted(provenance["source_sha256"].items())),
        "prediction_npy_coverage": coverage,
        "prediction_path": str(prediction_path),
        "prediction_sidecar_path": str(sidecar_path),
        "checkpoint_path": str(checkpoint_path),
        "provenance_path": str(provenance_path),
    }
    return CandidateArtifacts(prediction, identity)
