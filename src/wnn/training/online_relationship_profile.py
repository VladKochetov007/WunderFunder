"""Bound an online-relationship panel with complete-sequence feature timings."""

from __future__ import annotations

import hashlib
import json
import os
import gc
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from wnn.data import SequenceFile
from wnn.online_relationship import (
    OnlineRelationshipState,
    fit_online_relationship_normalizer,
)
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.membership import load_component_fit_subset

SCRATCH_ROOT = Path("/home/vlad/development/wunder/research_tmp").resolve()
PROFILE_SCHEMA = "wnn.online-relationships.feature-profile.v1"
ARMS = ("A", "B", "C")
LAGS = (1, 16)


def profile_feature_floor(
    recipe_path: Path,
    output_path: Path,
    *,
    root: Path,
    samples_per_fold: int = 8,
) -> dict[str, Any]:
    """Measure the actual six-stream CPU transform on eight complete fit sequences.

    The empirical floor uses each fold's fastest sampled six-stream sequence
    once for feature-cache generation. A separate estimate shows recomputing
    features for every epoch. This is a cost screen, not a mathematical bound
    or admission: it omits checkpoint inference, fitting, cache I/O and grading.
    """
    if samples_per_fold < 2 or samples_per_fold % 2:
        raise ValueError("samples_per_fold must be an even integer of at least two")
    root = root.resolve()
    recipe_path = _resolve(root, recipe_path)
    output_path = output_path.resolve()
    if not output_path.is_relative_to(SCRATCH_ROOT):
        raise ValueError(f"profile output must be under physical scratch {SCRATCH_ROOT}")
    recipe = _read_json(recipe_path)
    if recipe.get("schema_version") != 1 or recipe.get("resource_contract", {}).get(
        "total_new_fits"
    ) != 12:
        raise ValueError("recipe is not the frozen 12-fit online-relationships panel")
    report_folds = []
    projected_generation_seconds = 0.0
    projected_streaming_seconds = 0.0
    for fold in recipe["folds"]:
        report, generation, streaming = _profile_fold(root, recipe, fold, samples_per_fold)
        report_folds.append(report)
        projected_generation_seconds += generation
        projected_streaming_seconds += streaming
    ceiling = float(recipe["resource_contract"]["inclusive_local_worker_ceiling_seconds"])
    exceeds = projected_generation_seconds > ceiling
    result = {
        "schema": PROFILE_SCHEMA,
        "recipe_id": recipe["recipe_id"],
        "recipe_path": str(recipe_path),
        "recipe_sha256": file_sha256(recipe_path),
        "source_sha256": {
            "profile": file_sha256(Path(__file__).resolve()),
            "feature_module": file_sha256(
                Path(__file__).resolve().parents[1] / "online_relationship.py"
            ),
            "sequence_reader": file_sha256(Path(__file__).resolve().parents[1] / "data.py"),
        },
        "measurement": "CPU-only, complete 20,000-row sequences, one worker thread",
        "samples_per_fold": samples_per_fold,
        "folds": report_folds,
        "feature_cache_generation_empirical_floor_seconds": projected_generation_seconds,
        "three_epoch_streaming_recompute_estimate_seconds": projected_streaming_seconds,
        "inclusive_ceiling_seconds": ceiling,
        "decision": (
            "defer_entire_panel_feature_generation_exceeds_ceiling"
            if exceeds
            else "feature_generation_fits_ceiling_full_profile_required"
        ),
        "admission": "not_admitted_by_feature_only_profile",
        "excluded_costs": [
            "fit-only normalizer scan over every selected fitting sequence",
            "parent prediction cache creation for selected-fit and evaluation groups",
            "GPU readout forward/backward/AdamW updates",
            "full-fold candidate prediction export and grading",
            "checkpoint, sidecar, provenance and cache I/O",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_create_only(output_path, result)
    return result


def _profile_fold(
    root: Path, recipe: dict[str, Any], fold: dict[str, Any], samples_per_fold: int
) -> tuple[dict[str, Any], float, float]:
    split_path = _resolve(root, Path(fold["split_manifest"]))
    subset_path = _resolve(root, Path(fold["subset"]["path"]))
    if file_sha256(split_path) != fold["split_manifest_sha256"]:
        raise ValueError(f"fold {fold['fold']} split manifest hash changed")
    if file_sha256(subset_path) != fold["subset"]["sha256"]:
        raise ValueError(f"fold {fold['fold']} subset manifest hash changed")
    campaign = load_campaign_split(split_path, root)
    if campaign.fold != int(fold["fold"]):
        raise ValueError("recipe fold disagrees with frozen campaign split")
    subset = load_component_fit_subset(
        subset_path, campaign, expected_experiment_id=recipe["experiment_id"]
    )
    if list(campaign.evaluation_groups) != list(fold["evaluation_groups"]):
        raise ValueError("recipe evaluation group order differs from campaign split")
    train_groups = sorted(map(int, subset["train_groups"]))
    valid_groups = sorted(map(int, subset["valid_groups"]))
    refs = _sample_refs(train_groups, valid_groups, samples_per_fold)
    files = {
        split: SequenceFile(root / "datasets" / f"{split}.parquet")
        for split in ("train", "valid")
    }
    normalizers = {}
    norm_total_seconds = {}
    norm_read_seconds = {}
    for lag in LAGS:
        started = time.perf_counter()
        reads = {"seconds": 0.0}
        normalizers[lag] = fit_online_relationship_normalizer(
            _raw_sample_stream(files, refs, reads), lag
        )
        norm_total_seconds[str(lag)] = time.perf_counter() - started
        norm_read_seconds[str(lag)] = reads["seconds"]
    row_times: list[dict[str, Any]] = []
    state_seconds = 0.0
    sample_read_seconds = 0.0
    for split, group in refs:
        read_started = time.perf_counter()
        raw = files[split].read(group, targets=False, mask=False).features
        sample_read_seconds += time.perf_counter() - read_started
        started = time.perf_counter()
        for lag in LAGS:
            for arm in ARMS:
                features = OnlineRelationshipState(arm, lag, normalizers[lag]).transform(raw)
                if features.shape != (len(raw), 208) or not np.isfinite(features).all():
                    raise ValueError("online relationship transform returned malformed features")
                del features
        elapsed = time.perf_counter() - started
        state_seconds += elapsed
        row_times.append({"split": split, "row_group": group, "six_stream_seconds": elapsed})
        del raw
    del files
    gc.collect()
    sample_fastest = min(row["six_stream_seconds"] for row in row_times)
    total_fit_sequences = len(train_groups) + len(valid_groups)
    evaluation_sequences = len(campaign.evaluation_groups)
    generation_passes = total_fit_sequences + evaluation_sequences
    streaming_passes = 3 * total_fit_sequences + evaluation_sequences
    generation_floor = sample_fastest * generation_passes
    streaming_estimate = sample_fastest * streaming_passes
    return (
        {
            "fold": int(fold["fold"]),
            "seed": int(fold["seed"]),
            "split_manifest_sha256": campaign.sha256,
            "subset_sha256": file_sha256(subset_path),
            "fit_sequences": total_fit_sequences,
            "fit_train_sequences": len(train_groups),
            "fit_valid_sequences": len(valid_groups),
            "evaluation_sequences": len(campaign.evaluation_groups),
            "sample_group_refs": [
                {"split": split, "row_group": group} for split, group in refs
            ],
            "complete_sequence_rows": 20_000,
            "sample_read_seconds_eight_sequences": sample_read_seconds,
            "fit_normalizer_sample_read_seconds_by_lag": norm_read_seconds,
            "fit_normalizer_sample_total_seconds_by_lag": norm_total_seconds,
            "state_transform_seconds_eight_sequences_six_streams": state_seconds,
            "state_seconds_per_sample_sequence_fastest": sample_fastest,
            "projected_feature_generation_sequence_passes": generation_passes,
            "feature_cache_generation_empirical_floor_seconds": generation_floor,
            "three_epoch_streaming_recompute_sequence_passes": streaming_passes,
            "three_epoch_streaming_recompute_estimate_seconds": streaming_estimate,
            "sample_sequence_timings": row_times,
        },
        generation_floor,
        streaming_estimate,
    )


def _sample_refs(
    train_groups: list[int], valid_groups: list[int], sample_count: int
) -> list[tuple[str, int]]:
    per_split = sample_count // 2
    if len(train_groups) < per_split or len(valid_groups) < per_split:
        raise ValueError("each fit split must contain half of the complete-sequence profile sample")
    train_ix = np.linspace(0, len(train_groups) - 1, per_split, dtype=np.int64)
    valid_ix = np.linspace(0, len(valid_groups) - 1, per_split, dtype=np.int64)
    return [("train", train_groups[int(i)]) for i in train_ix] + [
        ("valid", valid_groups[int(i)]) for i in valid_ix
    ]


def _raw_sample_stream(
    files: dict[str, SequenceFile], refs: list[tuple[str, int]], read_timing: dict[str, float]
) -> Iterator[np.ndarray]:
    """Yield one raw sequence at a time and account for parquet read time."""
    for split, group in refs:
        started = time.perf_counter()
        sequence = files[split].read(group, targets=False, mask=False)
        read_timing["seconds"] += time.perf_counter() - started
        yield sequence.features
        del sequence


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > 5_000_000:
        raise ValueError(f"JSON metadata has invalid size: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_create_only(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{hashlib.sha256(payload).hexdigest()[:12]}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"profile output is immutable and already exists: {path}") from error
    temporary.unlink()
