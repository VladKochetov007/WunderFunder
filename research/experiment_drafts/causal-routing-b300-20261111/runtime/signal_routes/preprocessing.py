"""Train-only normalization identity for the Signal Routes model adapters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from signal_routes.bindings import RUNTIME_COMMIT, SPLIT_SHA256, canonical, read_object, write_new
from wnn.data import FEATURE_COLUMNS
from wnn.training.sequence_data import SequenceData


def fit_identity(data: SequenceData) -> dict[str, Any]:
    """Bind the full official training roster and canonical train-only moments."""
    if (
        data.split.sha256 != SPLIT_SHA256
        or data.split.valid_groups
        or len(data.fit.members) != 10_325
        or len(data.evaluation.members) != 1_873
        or any(source != "train" for source, _ in data.fit.members)
    ):
        raise ValueError("Signal Routes requires the complete pinned train-only split")
    if len(set(data.fit.members)) != 10_325:
        raise ValueError("fit membership contains duplicate sequences")
    data.normalization()
    members = [list(member) for member in data.fit.members]
    return {
        "split_sha256": data.split.sha256,
        "fit_members_sha256": canonical(members),
        "normalization": data.provenance["normalization"],
        "feature_columns": list(FEATURE_COLUMNS),
    }


def prepare(data: SequenceData, output: Path) -> dict[str, Any]:
    """Write one immutable moment artifact derived only from official train members."""
    identity = fit_identity(data)
    mean, std = data.normalization()
    mean_values, std_values = mean.tolist(), std.tolist()
    artifact = {
        "schema": "signal-routes.preprocessing.v1",
        "runtime_commit": RUNTIME_COMMIT,
        "fit_identity": identity,
        "mean": mean_values,
        "std": std_values,
    }
    if output.is_symlink() or output.exists():
        raise ValueError("preprocessing output must be a new path")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "preprocessing.json", artifact)
    return artifact


def load_preprocessing(path: Path, data: SequenceData | None = None) -> dict[str, Any]:
    """Validate artifact schema, dimensions, finite moments, and optional data identity."""
    artifact = read_object(path)
    if artifact.get("schema") != "signal-routes.preprocessing.v1":
        raise ValueError("unsupported Signal Routes preprocessing schema")
    if artifact.get("runtime_commit") != RUNTIME_COMMIT:
        raise ValueError("preprocessing runtime differs from the pinned PR75 source")
    identity = artifact.get("fit_identity")
    expected_identity_fields = {
        "split_sha256",
        "fit_members_sha256",
        "normalization",
        "feature_columns",
    }
    if not isinstance(identity, dict) or set(identity) != expected_identity_fields:
        raise ValueError("preprocessing fit identity has unexpected fields")
    if identity.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("preprocessing split differs from the official split pin")
    if identity.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("preprocessing feature order differs from the pinned schema")
    mean, std = artifact.get("mean"), artifact.get("std")
    if (
        not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 112
        or len(std) != 112
    ):
        raise ValueError("preprocessing moments must contain exactly 112 values")
    for values, label in ((mean, "mean"), (std, "std")):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"preprocessing {label} contains a nonfinite value")
    if any(value <= 0 for value in std):
        raise ValueError("preprocessing scales must be strictly positive")
    if data is not None:
        if identity != fit_identity(data):
            raise ValueError("preprocessing membership or normalization ancestry mismatch")
        actual_mean, actual_std = data.normalization()
        if actual_mean.tolist() != mean or actual_std.tolist() != std:
            raise ValueError("preprocessing moments differ from the canonical train normalizer")
    return artifact
