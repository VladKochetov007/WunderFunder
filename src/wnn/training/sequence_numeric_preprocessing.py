from __future__ import annotations

import hashlib
import json
from pathlib import Path

# JSON artifact fields are heterogeneous until the schema checks at the file boundary.
from typing import Any

import numpy as np

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, SequenceFile
from wnn.splits import CampaignSplit, load_campaign_split
from wnn.training.sequence_numeric import (
    NUMERIC_INPUT_WIDTH,
    NUMERIC_MISSING_OFFSET,
    numeric_inputs,
)
from wnn.training.sequence_numeric_artifact import (
    ARTIFACT_SCHEMA,
    FEATURE_SCHEMA,
    PREPARATION_SCHEMA,
    _atomic_create,
    _canonical_bytes,
    _canonical_sha256,
    _module_hashes,
    _validate_fit_arrays,
)
from wnn.training.sequence_numeric_artifact import load_numeric_artifact as load_numeric_artifact

DEFAULT_RESERVOIR_ROWS = 131_072
DEFAULT_CALIBRATION_SEED = 221_022
KNOT_PROBABILITIES = np.linspace(0.0, 1.0, 17, dtype=np.float64)


def _fit_members(split: CampaignSplit) -> list[tuple[str, int]]:
    """Return every source-qualified fitting member in canonical split order."""
    return [("train", group) for group in split.train_groups] + [
        ("valid", group) for group in split.valid_groups
    ]


def _dataset_paths(root: Path, sources: dict[str, object]) -> dict[str, Path]:
    """Resolve train and valid parquet paths from the validated source manifest."""
    paths: dict[str, Path] = {}
    for split_name in ("train", "valid"):
        matches = [
            name
            for name in sources
            if Path(name).suffix == ".parquet" and Path(name).stem == split_name
        ]
        if len(matches) != 1:
            raise ValueError(f"campaign sources must pin exactly one {split_name}.parquet")
        paths[split_name] = root / matches[0]
    return paths


def _update_moments(
    values: np.ndarray, mean: np.ndarray, m2: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Combine a finite block into stable float64 population moments."""
    block_count = len(values)
    block_mean = values.mean(axis=0, dtype=np.float64)
    centered = values.astype(np.float64) - block_mean
    block_m2 = np.square(centered).sum(axis=0, dtype=np.float64)
    merged = count + block_count
    if count == 0:
        return block_mean, block_m2, block_count
    delta = block_mean - mean
    mean += delta * (block_count / merged)
    m2 += block_m2 + np.square(delta) * (count * block_count / merged)
    return mean, m2, merged


def _update_reservoir(
    keys: np.ndarray,
    positions: np.ndarray,
    rows: np.ndarray,
    new_keys: np.ndarray,
    new_positions: np.ndarray,
    new_rows: np.ndarray,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the globally smallest random priorities in bounded row memory."""
    combined_keys = np.concatenate((keys, new_keys))
    combined_positions = np.concatenate((positions, new_positions))
    combined_rows = np.concatenate((rows, new_rows), axis=0)
    if len(combined_keys) > capacity:
        retained = np.argpartition(combined_keys, capacity - 1)[:capacity]
    else:
        retained = np.arange(len(combined_keys))
    return (
        combined_keys[retained].copy(),
        combined_positions[retained].copy(),
        combined_rows[retained].copy(),
    )


def prepare_numeric_artifact(
    root: Path,
    split_manifest: Path,
    output: Path,
    reservoir_rows: int = DEFAULT_RESERVOIR_ROWS,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> dict[str, Any]:
    """Fit numeric moments and knots from every whole fitting sequence.

    Quantile knots use a deterministic priority reservoir bounded by
    ``reservoir_rows``. Only train and fitting-valid features are read; held-out
    groups, targets, and masks never enter preprocessing.
    """
    if type(reservoir_rows) is not int or reservoir_rows < 1:
        raise ValueError("reservoir_rows must be a positive integer")
    if type(seed) is not int or seed < 0 or seed >= 2**64:
        raise ValueError("calibration seed must be an unsigned 64-bit integer")
    root = root.resolve()
    split = load_campaign_split(split_manifest, root)
    raw_manifest = split.path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != split.sha256:
        raise ValueError("campaign split changed after membership validation")
    manifest = json.loads(raw_manifest)
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not all(isinstance(key, str) for key in sources):
        raise ValueError("campaign split source identities are malformed")
    paths = _dataset_paths(root, sources)
    members = _fit_members(split)
    member_lists = [[source, group] for source, group in members]

    mean = np.zeros(NUMERIC_INPUT_WIDTH, dtype=np.float64)
    m2 = np.zeros(NUMERIC_INPUT_WIDTH, dtype=np.float64)
    raw_min = np.full(N_FEATURES, np.inf, dtype=np.float64)
    raw_max = np.full(N_FEATURES, -np.inf, dtype=np.float64)
    generator = np.random.default_rng(seed)
    keys = np.empty(0, dtype=np.float64)
    positions = np.empty(0, dtype=np.int64)
    reservoir = np.empty((0, N_FEATURES), dtype=np.float32)
    count = 0
    files = {name: SequenceFile(path) for name, path in paths.items()}
    for member_index, (source, group) in enumerate(members):
        sequence = files[source].read(group, targets=False, mask=False)
        values = numeric_inputs(sequence.features)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite numeric preprocessing row: {source}/{group}")
        mean, m2, count = _update_moments(values, mean, m2, count)
        raw = values[:, :N_FEATURES]
        raw_min = np.minimum(raw_min, raw.min(axis=0).astype(np.float64))
        raw_max = np.maximum(raw_max, raw.max(axis=0).astype(np.float64))
        row_count = len(raw)
        row_positions = member_index * SEQUENCE_LENGTH + np.arange(row_count, dtype=np.int64)
        row_keys = generator.random(row_count, dtype=np.float64)
        keys, positions, reservoir = _update_reservoir(
            keys,
            positions,
            reservoir,
            row_keys,
            row_positions,
            raw,
            reservoir_rows,
        )
    if (
        count <= 0
        or not members
        or not np.isfinite(raw_min).all()
        or not np.isfinite(raw_max).all()
    ):
        raise ValueError("campaign split has no finite fitting rows")

    std = np.sqrt(np.maximum(m2 / count, 0.0)).clip(0.001)
    mean[NUMERIC_MISSING_OFFSET:] = 0.0
    std[NUMERIC_MISSING_OFFSET:] = 1.0
    mean = mean.astype(np.float32).astype(np.float64)
    std = std.astype(np.float32).astype(np.float64)
    normalized_min = (raw_min - mean[:N_FEATURES]) / std[:N_FEATURES]
    normalized_max = (raw_max - mean[:N_FEATURES]) / std[:N_FEATURES]
    uniform = (
        normalized_min[:, None] + (normalized_max - normalized_min)[:, None] * KNOT_PROBABILITIES
    )
    sampled_raw = reservoir.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        sampled_normalized = (sampled_raw - mean[:N_FEATURES]) / std[:N_FEATURES]
    quantile = np.quantile(sampled_normalized, KNOT_PROBABILITIES, axis=0, method="linear").T
    quantile[:, 0], quantile[:, -1] = normalized_min, normalized_max
    artifact: dict[str, Any] = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "split_sha256": split.sha256,
        "fit_members": member_lists,
        "sources": sources,
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "knots": {
            "uniform16": uniform.astype(np.float32).tolist(),
            "quantile16": quantile.astype(np.float32).tolist(),
        },
        "preparation_identity": {
            "schema": PREPARATION_SCHEMA,
            "algorithm": "float64-moments-priority-reservoir-v1",
            "calibration_seed": seed,
            "reservoir_rows_limit": reservoir_rows,
            "reservoir_rows_sampled": len(reservoir),
            "full_scanned_rows": count,
            "fit_members_sha256": _canonical_sha256(member_lists),
            "source_code_sha256": _module_hashes(),
        },
    }
    order = np.argsort(positions, kind="stable")
    sample_digest = hashlib.sha256()
    sample_digest.update(positions[order].astype("<i8", copy=False).tobytes())
    sample_digest.update(keys[order].astype("<f8", copy=False).tobytes())
    sample_digest.update(reservoir[order].astype("<f4", copy=False).tobytes())
    artifact["preparation_identity"]["reservoir_sample_sha256"] = sample_digest.hexdigest()
    _validate_fit_arrays(artifact)
    artifact["preparation_hash"] = _canonical_sha256(artifact)
    _atomic_create(output, _canonical_bytes(artifact) + b"\n")
    return artifact
