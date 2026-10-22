"""Memory-bounded alignment, provenance, metric, and bootstrap helpers."""

from __future__ import annotations

# Any is reserved for schema-checked JSON receipts and split objects.
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from receiver_assets import (
    Cell,
    ReceiverAssets,
    file_sha256,
    load_json,
)


def selected_targets(
    root: Path, split: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Align cached valid labels, scored rows, and component IDs to the split order."""
    cache_path = root / "cache" / "valid_targets.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        row_groups = np.asarray(cache["row_group"], dtype=np.int64)
        positions = {int(group): i for i, group in enumerate(row_groups)}
        evaluation = np.asarray(split.evaluation_groups, dtype=np.int64)
        if len(positions) != len(row_groups) or any(
            int(group) not in positions for group in evaluation
        ):
            raise ValueError("valid target cache has duplicate or missing evaluation row groups")
        indices = np.asarray([positions[int(group)] for group in evaluation], dtype=np.int64)
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        scored = np.asarray(cache["mask"][indices], dtype=bool)
        sequence_ids = np.asarray(cache["seq_ix"][indices], dtype=np.int64)
    if targets.ndim != 3 or targets.shape[-1] != 2 or scored.shape != targets.shape[:2]:
        raise ValueError("valid target cache must align as (N,T,2) labels and (N,T) scoring mask")
    import pyarrow.parquet as pq

    source = pq.ParquetFile(root / "datasets" / "valid.parquet")
    for index, group in enumerate(evaluation):
        table = source.read_row_group(
            int(group), columns=["seq_ix", "t0", "t1", "is_scored"], use_threads=False
        )
        if table.num_rows != targets.shape[1]:
            raise ValueError(f"valid parquet row group {group} has an unexpected sequence length")
        source_targets = np.column_stack((table["t0"].to_numpy(), table["t1"].to_numpy())).astype(
            np.float32, copy=False
        )
        source_mask = table["is_scored"].to_numpy().astype(bool, copy=False)
        source_sequence_id = int(table["seq_ix"][0].as_py())
        if (
            source_sequence_id != int(sequence_ids[index])
            or not np.array_equal(source_targets, targets[index])
            or not np.array_equal(source_mask, scored[index])
        ):
            raise ValueError(f"valid target cache differs from source parquet at row group {group}")
    from wnn.data import WARMUP

    need_prediction = np.arange(targets.shape[1]) >= int(WARMUP)
    selection = scored & need_prediction[None, :]
    if not selection.any():
        raise ValueError("held-out campaign fold has no need_prediction & is_scored rows")
    component_map = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    try:
        components = np.asarray([component_map[int(group)] for group in evaluation], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"evaluation row group lacks a component binding: {exc}") from exc
    row_digest = hashlib.sha256(np.asarray(row_groups, dtype="<i8").tobytes()).hexdigest()
    split_manifest = load_json(Path(split.path))
    valid_source_identity = split_manifest.get("sources", {}).get("datasets/valid.parquet")
    if not isinstance(valid_source_identity, dict):
        raise ValueError("campaign split lacks its valid parquet source identity")
    return (
        targets,
        selection,
        components,
        {
            "path": str(cache_path),
            "sha256": file_sha256(cache_path),
            "valid_parquet_source_identity": valid_source_identity,
            "row_groups_sha256": row_digest,
            "row_group_count": len(row_groups),
            "target_shape": list(targets.shape),
            "scored_mask_shape": list(scored.shape),
            "verified_against_valid_parquet": True,
            "verified_evaluation_groups": len(evaluation),
        },
    )


def validated_prediction(
    assets: ReceiverAssets, cell: Cell, role: str, run: str, expected_split_sha: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify sidecar, held-out row order, checkpoint binding, and output checksum."""
    from receiver_assets import prediction_name

    name = prediction_name(cell, role)
    prediction_path = assets.root / "cache" / f"pred_{name}_valid.npy"
    metadata_path = prediction_path.with_suffix(".json")
    if not prediction_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"missing held-out {role} prediction for {cell.fold}: {prediction_path}"
        )
    metadata = load_json(metadata_path)
    checkpoint_path = (
        assets.root
        / "artifacts"
        / "runs"
        / run
        / ("model_ep3.pt" if role == "parent" else "model_ep0.pt")
    )
    checkpoint_hash = file_sha256(checkpoint_path)
    provenance_path = checkpoint_path.with_name("provenance.json")
    provenance = load_json(provenance_path)
    checkpoint_name = checkpoint_path.name
    source_identity = provenance.get("trainer_implementation")
    if role == "parent":
        historical_source_sha = provenance.get("run_identity_source_sha256")
        if not isinstance(historical_source_sha, str) or len(historical_source_sha) != 64:
            raise ValueError(f"{cell.fold} parent lacks its recorded historical run-identity hash")
        source_binding = {
            "evidence": "historical_run_identity_source_sha256",
            "sha256": historical_source_sha,
            "trainer_implementation": "not_recorded_by_parent_provenance",
        }
    else:
        if (
            not isinstance(source_identity, dict)
            or not source_identity.get("sha256")
            or not source_identity.get("files")
        ):
            raise ValueError(f"{cell.fold}/{role} child lacks exact trainer source identity")
        source_binding = source_identity
    expected_epoch = 3 if role == "parent" else 0
    expected_membership_sha = hashlib.sha256(
        np.asarray(assets.splits[cell.fold].evaluation_groups, dtype="<i8").tobytes()
    ).hexdigest()
    if (
        metadata.get("run") != run
        or metadata.get("split") != "valid"
        or metadata.get("epoch") != expected_epoch
        or metadata.get("checkpoint_sha256") != checkpoint_hash
        or provenance.get("checkpoint_sha256", {}).get(checkpoint_name) != checkpoint_hash
        or metadata.get("campaign_export_scope") != "evaluation"
        or metadata.get("feature_only") is not True
        or metadata.get("evaluation_membership_sha256") != expected_membership_sha
        or metadata.get("campaign_split_manifest_sha256") != expected_split_sha
        or metadata.get("split_manifest_sha256") != expected_split_sha
        or metadata.get("inference_precision") != "fp32"
        or metadata.get("prediction_sha256") != file_sha256(prediction_path)
    ):
        raise ValueError(f"{cell.fold}/{role} prediction metadata is not bound to its inputs")
    split = assets.splits[cell.fold]
    if metadata.get("row_groups") != list(split.evaluation_groups):
        raise ValueError(f"{cell.fold}/{role} prediction row order differs from held-out groups")
    if metadata.get("campaign_split_manifest_sha256") != file_sha256(assets.split_paths[cell.fold]):
        raise ValueError(f"{cell.fold}/{role} prediction used a different split manifest file")
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(split.evaluation_groups), 20_000, 2)
    if prediction.shape != expected_shape:
        raise ValueError(
            f"{cell.fold}/{role} prediction shape {prediction.shape} != {expected_shape}"
        )
    return prediction, {
        "run": run,
        "output_name": name,
        "checkpoint_sha256": checkpoint_hash,
        "provenance_sha256": file_sha256(provenance_path),
        "source_binding": source_binding,
        "prediction_sha256": metadata["prediction_sha256"],
        "prediction_path": str(prediction_path),
    }


def score_array(
    targets: np.ndarray, prediction: np.ndarray, selection: np.ndarray
) -> dict[str, Any]:
    """Stream official global weighted Pearson over a prediction memmap."""
    from wnn.gmetric import GlobalAccumulator

    accumulator = GlobalAccumulator()
    for start in range(0, len(targets), 8):
        stop = min(start + 8, len(targets))
        accumulator.add(
            targets[start:stop], np.asarray(prediction[start:stop]), selection[start:stop]
        )
    return accumulator.result()


def component_moments(
    targets: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    selection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-sequence centered moments in small slices to bound prediction RAM."""
    from wnn.evaluation import paired_component_moments

    candidate_parts, reference_parts = [], []
    for start in range(0, len(targets), 8):
        stop = min(start + 8, len(targets))
        candidate_part, reference_part = paired_component_moments(
            targets[start:stop],
            np.asarray(candidate[start:stop]),
            np.asarray(reference[start:stop]),
            selection[start:stop],
        )
        candidate_parts.append(candidate_part)
        reference_parts.append(reference_part)
    return np.concatenate(candidate_parts), np.concatenate(reference_parts)


def blend_array(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    """Build one prespecified 75/25 blend in a float32 array."""
    return (
        0.75 * np.asarray(parent, dtype=np.float32) + 0.25 * np.asarray(child, dtype=np.float32)
    ).astype(np.float32)


def fold_seed_streams(seed: int, count: int) -> list[int]:
    """Derive stable, independent RNG streams for the fixed fold order."""
    streams = np.random.SeedSequence(seed).spawn(count)
    return [int(stream.generate_state(1, dtype=np.uint32)[0]) for stream in streams]
