#!/usr/bin/env python3
"""Hash, validate, and grade fixed transforms of existing validation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gmetric_standalone import GlobalAccumulator, _centered_moments, _wp_from_moments, wp_sums  # noqa: E402
from prediction_audit_core import (  # noqa: E402
    AGE_LABELS,
    SCENARIO_NAMES,
    age_buckets,
    build_scenarios,
    clipped_label_run_oracle,
    event_flags,
)

EXPECTED_SHAPE = (1873, 20_000, 2)
WARMUP = 99
BOOTSTRAPS = 512
BOOTSTRAP_SEED = 20261017
REGION_NAMES = ("all", "i0_and_event", "quiet_not_and", "target_change_exception")
POLICIES = (
    ("pred_t384_s3_full_valid.npy", "t384_s3", "model_ep2.pt"),
    ("pred_t384_s4_full_valid.npy", "t384_s4", "model_ep2.pt"),
    ("pred_dftt_tlead_s3_valid.npy", "dftt_tlead_s3", "model_ep3.pt"),
)


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_record(path: Path) -> dict[str, int | str]:
    """Return stable file metadata used to detect mutation between phases."""
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def file_record(path: Path) -> dict[str, int | str]:
    """Return file metadata and a SHA-256 digest."""
    return {**stat_record(path), "sha256": sha256_file(path)}


def command_output(args: list[str]) -> str | None:
    """Return a read-only command result, or null when a command is unavailable."""
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def checkpoint_settings(path: Path) -> dict[str, object]:
    """Read selected non-weight checkpoint arguments without running the model."""
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    args = checkpoint.get("args", {})
    names = (
        "name",
        "seed",
        "arch",
        "hidden",
        "layers",
        "enc",
        "train_source",
        "train_seqs",
        "valid_seqs",
        "eval_source",
        "valid_fold",
        "n_valid_folds",
        "full_valid",
        "include_heldout_valid",
        "lookahead",
        "zero_groups",
        "a_mode",
        "domain_flag",
        "online_norm",
        "distill",
        "distill_boot_h",
        "epoch",
    )
    return {name: args.get(name) for name in names if name in args} | {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "n_in": checkpoint.get("n_in"),
    }


def load_fold_membership(path: Path, row_groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align validation chain/fold ids to target-cache row-group order."""
    with np.load(path, allow_pickle=False) as archive:
        split = archive["split"]
        cache_groups = archive["row_group"]
        chains = archive["chain"]
        folds = archive["fold"]
    valid = split == "valid"
    positions = {int(group): i for i, group in enumerate(cache_groups[valid])}
    if set(positions) != set(int(group) for group in row_groups):
        raise ValueError("chain-fold membership does not cover valid-target row groups")
    order = np.asarray([positions[int(group)] for group in row_groups], dtype=np.int64)
    return chains[valid][order].astype(np.int64), folds[valid][order].astype(np.int64)


def load_target_cache(path: Path) -> dict[str, np.ndarray]:
    """Load the fixed target, official mask, row order, and retrospective strata."""
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if tuple(data["y"].shape) != EXPECTED_SHAPE:
        raise ValueError(f"unexpected target cache shape: {data['y'].shape}")
    if data["mask"].shape != EXPECTED_SHAPE[:2]:
        raise ValueError("target mask shape differs from y")
    if data["row_group"].shape != (EXPECTED_SHAPE[0],):
        raise ValueError("target row-group membership has an unexpected shape")
    return data


def inventory(args: argparse.Namespace) -> dict[str, object]:
    """Validate shapes, finite-row-group coverage, ancestry clues, and file hashes."""
    output_path = args.output_dir / "INVENTORY.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    cache_dir, data_dir, source_root = args.cache, args.data, args.source_root
    target_path = cache_dir / "valid_targets.npz"
    fold_path = cache_dir / "chain_folds.npz"
    parquet_path = data_dir / "valid.parquet"
    data = load_target_cache(target_path)
    row_groups = data["row_group"].astype(np.int64)
    seq_ids = data["seq_ix"].astype(np.int64)
    target = data["y"]
    cache_mask = data["mask"].astype(bool, copy=False)
    score_mask = cache_mask & (np.arange(target.shape[1]) >= WARMUP)[None, :]
    if np.any(cache_mask[:, :WARMUP]):
        raise ValueError("cached valid scoring mask unexpectedly selects warmup rows")
    if not np.isfinite(target[score_mask]).all():
        raise ValueError("the official selected target rows contain nonfinite values")
    if row_groups.tolist() != list(range(len(row_groups))) or len(np.unique(seq_ids)) != len(
        seq_ids
    ):
        raise ValueError("valid target cache is not unique row-group order")
    chain_ids, fold_ids = load_fold_membership(fold_path, row_groups)
    parquet = pq.ParquetFile(parquet_path)
    if parquet.metadata.num_row_groups != len(row_groups):
        raise ValueError("valid parquet row-group count differs from target cache")
    if any(
        parquet.metadata.row_group(i).num_rows != EXPECTED_SHAPE[1] for i in range(len(row_groups))
    ):
        raise ValueError("valid parquet has a row group with a nonstandard sequence length")

    candidates: list[dict[str, object]] = []
    complete_sets: list[set[int]] = []
    runs_dir = source_root / "artifacts" / "runs"
    for cache_name, run_name, checkpoint_name in POLICIES:
        prediction_path = cache_dir / cache_name
        prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if prediction.shape != EXPECTED_SHAPE or prediction.dtype != np.float32:
            raise ValueError(f"unexpected prediction schema for {cache_name}")
        finite_groups = np.empty(len(row_groups), dtype=bool)
        for index in range(len(row_groups)):
            finite_groups[index] = bool(np.isfinite(prediction[index]).all())
        complete_group_ids = row_groups[finite_groups].astype(int).tolist()
        complete_sets.append(set(complete_group_ids))
        run_dir = runs_dir / run_name
        checkpoint_path = run_dir / checkpoint_name
        history_path = run_dir / "history.json"
        receipt_path = prediction_path.with_suffix(prediction_path.suffix + ".receipt.json")
        checkpoint_stat = file_record(checkpoint_path)
        history_stat = file_record(history_path)
        settings = checkpoint_settings(checkpoint_path)
        if settings.get("lookahead") not in ("", None):
            raise ValueError(
                f"{run_name} checkpoint has nonempty lookahead: {settings.get('lookahead')!r}"
            )
        candidates.append(
            {
                "cache_name": cache_name,
                "prediction": file_record(prediction_path),
                "shape": list(prediction.shape),
                "dtype": str(prediction.dtype),
                "finite_row_groups": len(complete_group_ids),
                "complete_row_group_ids": complete_group_ids,
                "finite_coverage_sha256": hashlib.sha256(
                    np.packbits(finite_groups, bitorder="little").tobytes()
                ).hexdigest(),
                "checkpoint_file_inspected_not_bound_to_prediction": checkpoint_stat,
                "checkpoint_settings": settings,
                "checkpoint_causality_check": "lookahead empty in inspected checkpoint args; cache producer linkage remains unverified",
                "history_file": history_stat,
                "prediction_receipt_present": receipt_path.exists(),
                "producer_binding": "unverified: matching prediction receipt is absent",
            }
        )
        del prediction
    common_groups = sorted(set.intersection(*complete_sets))
    common_positions = np.asarray([int(group) for group in common_groups], dtype=np.int64)
    common_fold_counts = Counter(int(fold_ids[position]) for position in common_positions)
    common_chain_ids = chain_ids[common_positions]
    common_selected = score_mask[common_positions]
    coverage_bytes = np.asarray(common_groups, dtype="<i8").tobytes()
    membership = hashlib.sha256(
        coverage_bytes + np.packbits(common_selected, bitorder="little").tobytes()
    ).hexdigest()

    prediction_source = source_root / "scripts" / "predict_run.py"
    trainer_source = source_root / "scripts" / "train_rnn.py"
    model_source = source_root / "src" / "wnn" / "nn.py"
    source_files = [prediction_source, trainer_source, model_source]
    git_head = command_output(["git", "-C", str(source_root), "rev-parse", "HEAD"])
    git_status = command_output(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--short",
            "--",
            "scripts/predict_run.py",
            "scripts/train_rnn.py",
            "src/wnn/nn.py",
        ]
    )
    record: dict[str, object] = {
        "schema": "wunder.prediction-audit.inventory.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection": "two t384 seeds plus one distinct tiedlead encoder; fixed by lineage before scoring",
        "metric_id": "global_wp_clip2_v1",
        "warmup_rows": WARMUP,
        "scoring_mask_source": "valid_targets.npz mask & row_index >= 99 (cache mask contains no pre-99 selected rows)",
        "cache_order_contract_source_current_not_producer_bound": str(prediction_source),
        "source_git_head": git_head,
        "source_dirty_status_for_relevant_files": git_status,
        "source_files_current_hashes_not_producer_bound": [
            file_record(path) for path in source_files
        ],
        "audit_code": {
            path.name: file_record(path)
            for path in (
                HERE / "run_prediction_audit.py",
                HERE / "prediction_audit_core.py",
                HERE / "gmetric_standalone.py",
            )
        },
        "data": file_record(parquet_path),
        "target_cache": file_record(target_path),
        "fold_cache": file_record(fold_path),
        "target_schema": {
            "shape": list(target.shape),
            "dtype": str(target.dtype),
            "valid_row_groups": len(row_groups),
            "row_group_order_sha256": hashlib.sha256(
                row_groups.astype("<i8").tobytes()
            ).hexdigest(),
            "sequence_order_sha256": hashlib.sha256(seq_ids.astype("<i8").tobytes()).hexdigest(),
            "official_mask_sha256": hashlib.sha256(
                np.packbits(score_mask, bitorder="little").tobytes()
            ).hexdigest(),
            "selected_rows_all_valid": int(score_mask.sum()),
            "selected_rows_common_complete": int(common_selected.sum()),
            "target_finite_all_selected": bool(np.isfinite(target[score_mask]).all()),
            "market_class_sequence_counts": {
                str(int(value)): int((data["market_class"] == value).sum())
                for value in np.unique(data["market_class"])
            },
        },
        "candidates": candidates,
        "shared_complete_coverage": {
            "row_groups": common_groups,
            "count": len(common_groups),
            "membership_sha256": membership,
            "fold_counts": {str(key): value for key, value in sorted(common_fold_counts.items())},
            "unique_chain_count": int(np.unique(common_chain_ids).size),
            "selected_rows": int(common_selected.sum()),
            "rule": "intersection of whole-row-group finite coverage, determined before grading; no individual rows removed",
        },
        "limits": [
            "candidate prediction files have no receipt sidecars, so their producer checkpoint/source linkage cannot be proven",
            "the full-valid t384 caches and fold-limited tiedlead cache are evaluated only on their fixed shared complete-group intersection",
            "these exposed validation predictions are not claimed as pristine OOF or hidden-test predictions",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "inventory": str(output_path),
                "candidate_groups": {c["cache_name"]: c["finite_row_groups"] for c in candidates},
                "common_groups": len(common_groups),
                "common_fold_counts": dict(common_fold_counts),
                "selected_rows": int(common_selected.sum()),
                "data_sha256": record["data"]["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return record


def _prediction_metrics(
    accumulator: GlobalAccumulator, squared_error: np.ndarray
) -> dict[str, object]:
    """Extract canonical scores and per-target mass, covariance, and weighted MSE."""
    if accumulator.selected_rows == 0:
        return {"selected_rows": 0, "t0": None, "t1": None, "wp": None, "targets": []}
    result = accumulator.result()
    components = []
    for target_index, name in enumerate(("t0", "t1")):
        moment = accumulator._moments[target_index]
        mass = float(moment[0])
        components.append(
            {
                "target": name,
                "wp": float(_wp_from_moments(moment)),
                "weight_mass": mass,
                "weighted_covariance_numerator": float(moment[5]),
                "weighted_target_variance_sum": float(moment[3]),
                "weighted_prediction_variance_sum": float(moment[4]),
                "weighted_mse_sum": float(squared_error[target_index]),
                "weighted_mse": float(squared_error[target_index] / mass) if mass > 0 else None,
            }
        )
    return {**result, "targets": components}


def _score_from_sequence_moments(
    moments: np.ndarray, multiplicity: np.ndarray
) -> tuple[float, float, float]:
    """Merge sequence-level centered moments with bootstrap multiplicities."""
    correlations: list[float] = []
    for component in range(2):
        values = moments[:, component]
        count = multiplicity.astype(np.float64)
        weights = count * values[:, 0]
        total = float(weights.sum())
        if total < 1e-8:
            correlations.append(0.0)
            continue
        mean_y = float(np.sum(weights * values[:, 1]) / total)
        mean_p = float(np.sum(weights * values[:, 2]) / total)
        var_y = float(np.sum(count * (values[:, 3] + values[:, 0] * (values[:, 1] - mean_y) ** 2)))
        var_p = float(np.sum(count * (values[:, 4] + values[:, 0] * (values[:, 2] - mean_p) ** 2)))
        covariance = float(
            np.sum(
                count
                * (values[:, 5] + values[:, 0] * (values[:, 1] - mean_y) * (values[:, 2] - mean_p))
            )
        )
        merged = np.asarray([total, mean_y, mean_p, var_y, var_p, covariance], dtype=np.float64)
        correlations.append(float(_wp_from_moments(merged)))
    return correlations[0], correlations[1], 0.5 * (correlations[0] + correlations[1])


def paired_chain_bootstrap(
    sequence_moments: np.ndarray, chain_ids: np.ndarray
) -> dict[str, object]:
    """Estimate paired policy differences conditional on fixed predictions."""
    unique_chains, chain_index = np.unique(chain_ids, return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(unique_chains), size=(BOOTSTRAPS, len(unique_chains)))
    chain_counts = np.zeros((BOOTSTRAPS, len(unique_chains)), dtype=np.int16)
    row = np.arange(BOOTSTRAPS)[:, None]
    np.add.at(chain_counts, (row, draws), 1)
    sequence_counts = chain_counts[:, chain_index]
    n_models, n_scenarios, n_sequences = sequence_moments.shape[:3]
    scores = np.empty((BOOTSTRAPS, n_models, n_scenarios, 3), dtype=np.float64)
    for replicate in range(BOOTSTRAPS):
        multiplicity = sequence_counts[replicate]
        for model in range(n_models):
            for scenario in range(n_scenarios):
                scores[replicate, model, scenario] = _score_from_sequence_moments(
                    sequence_moments[model, scenario], multiplicity
                )
    intervals: dict[str, dict[str, object]] = {}
    for model in range(n_models):
        model_results = {}
        for scenario, name in enumerate(SCENARIO_NAMES[1:], start=1):
            delta = scores[:, model, scenario] - scores[:, model, 0]
            model_results[name] = {
                target_name: {
                    "mean_delta": float(np.mean(delta[:, target_index])),
                    "q025": float(np.quantile(delta[:, target_index], 0.025)),
                    "q975": float(np.quantile(delta[:, target_index], 0.975)),
                }
                for target_index, target_name in enumerate(("t0", "t1", "wp"))
            }
        intervals[str(model)] = model_results
    return {
        "method": "paired resampling of validation chain clusters, conditional on fixed cached outputs",
        "replicates": BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
        "cluster_count": int(len(unique_chains)),
        "interval_columns": ["mean_delta", "q025", "q975"],
        "models_by_index": list(range(n_models)),
        "intervals_by_model_index": intervals,
        "limits": "not fit variability, independent seeds, or a pristine holdout interval",
    }


def grade(args: argparse.Namespace) -> dict[str, object]:
    """Run the predeclared policy matrix on shared, whole-group-complete rows."""
    inventory_path = args.output_dir / "INVENTORY.json"
    result_path = args.output_dir / "RESULTS.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    inventory_data = json.loads(inventory_path.read_text(encoding="utf-8"))
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    target_path = args.cache / "valid_targets.npz"
    fold_path = args.cache / "chain_folds.npz"
    parquet_path = args.data / "valid.parquet"
    target_data = load_target_cache(target_path)
    for label, path, expected in (
        ("target cache", target_path, inventory_data["target_cache"]),
        ("fold cache", fold_path, inventory_data["fold_cache"]),
        ("validation parquet", parquet_path, inventory_data["data"]),
    ):
        actual_stat = stat_record(path)
        expected_stat = {key: expected[key] for key in actual_stat}
        if actual_stat != expected_stat:
            raise ValueError(f"{label} changed after inventory")
    row_groups = target_data["row_group"].astype(np.int64)
    positions_by_group = {int(group): position for position, group in enumerate(row_groups)}
    group_ids = np.asarray(inventory_data["shared_complete_coverage"]["row_groups"], dtype=np.int64)
    positions = np.asarray([positions_by_group[int(group)] for group in group_ids], dtype=np.int64)
    if not len(positions):
        raise ValueError("no common finite row groups remain")
    chain_ids, fold_ids = load_fold_membership(fold_path, row_groups)
    common_folds = np.unique(fold_ids[positions])
    if len(common_folds) != 1:
        raise ValueError(
            "shared complete groups span multiple folds; freeze grouping before grading"
        )

    target = target_data["y"]
    official_mask = target_data["mask"].astype(bool, copy=False)
    official_mask &= (np.arange(target.shape[1]) >= WARMUP)[None, :]
    _, n_time, _ = EXPECTED_SHAPE
    selected_target = target[positions]
    selected_mask = official_mask[positions]
    and_events = np.empty((len(positions), n_time), dtype=bool)
    or_events = np.empty_like(and_events)
    label_changes = np.zeros_like(and_events)
    ages = np.empty((len(positions), n_time), dtype=np.uint8)
    parquet = pq.ParquetFile(parquet_path)
    for local_index, (group, position) in enumerate(zip(group_ids, positions, strict=True)):
        table = parquet.read_row_group(
            int(group), columns=["i0_p0", "i0_p11", "seq_ix"], use_threads=False
        )
        if table.num_rows != n_time:
            raise ValueError(f"row group {group} has {table.num_rows} rows")
        seq_values = table["seq_ix"].to_numpy(zero_copy_only=False)
        if not np.all(seq_values == target_data["seq_ix"][position]):
            raise ValueError(f"cached and parquet sequence identities differ for row group {group}")
        price_p0 = table["i0_p0"].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        price_p11 = table["i0_p11"].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        and_flag, or_flag = event_flags(price_p0[None, :], price_p11[None, :])
        and_events[local_index] = and_flag[0]
        or_events[local_index] = or_flag[0]
        ages[local_index] = age_buckets(and_flag)[0]
        raw_target = target[position]
        label_changes[local_index, 1:] = np.any(raw_target[1:] != raw_target[:-1], axis=1)

    exceptions = label_changes & ~and_events
    input_prediction_hashes_before = {
        candidate["cache_name"]: sha256_file(args.cache / candidate["cache_name"])
        for candidate in inventory_data["candidates"]
    }
    if any(
        input_prediction_hashes_before[candidate["cache_name"]] != candidate["prediction"]["sha256"]
        for candidate in inventory_data["candidates"]
    ):
        raise ValueError("prediction input hash differs from pre-score inventory")
    input_file_stats = [
        stat_record(args.cache / c["cache_name"]) for c in inventory_data["candidates"]
    ]
    candidate_arrays = [
        np.load(args.cache / candidate["cache_name"], mmap_mode="r", allow_pickle=False)
        for candidate in inventory_data["candidates"]
    ]
    if [
        stat_record(args.cache / c["cache_name"]) for c in inventory_data["candidates"]
    ] != input_file_stats:
        raise ValueError("prediction input changed during opening")

    n_models, n_scenarios = len(candidate_arrays), len(SCENARIO_NAMES)
    accumulators = [[GlobalAccumulator() for _ in SCENARIO_NAMES] for _ in candidate_arrays]
    region_accumulators = [
        [[GlobalAccumulator() for _ in REGION_NAMES] for _ in SCENARIO_NAMES]
        for _ in candidate_arrays
    ]
    sequence_moments = np.zeros((n_models, n_scenarios, len(positions), 2, 6), dtype=np.float64)
    sequence_sums = np.zeros_like(sequence_moments)
    region_counts = np.zeros((n_models, n_scenarios, len(REGION_NAMES)), dtype=np.int64)
    region_mass = np.zeros((n_models, n_scenarios, len(REGION_NAMES), 2), dtype=np.float64)
    region_squared_error = np.zeros_like(region_mass)
    age_counts = np.zeros((n_models, n_scenarios, 2, len(AGE_LABELS)), dtype=np.int64)
    age_mass = np.zeros((n_models, n_scenarios, 2, len(AGE_LABELS)), dtype=np.float64)
    age_squared_error = np.zeros_like(age_mass)
    exposure = np.zeros((n_models, n_scenarios, 3), dtype=np.int64)
    exposure_names = ("rows", "selected_rows", "different_from_baseline_rows")
    within_variance = np.zeros((n_models, 2), dtype=np.float64)
    start = time.time()
    batch_size = 8
    for start_index in range(0, len(positions), batch_size):
        stop = min(start_index + batch_size, len(positions))
        local = slice(start_index, stop)
        positions_chunk = positions[local]
        target_chunk = selected_target[local]
        mask_chunk = selected_mask[local]
        and_chunk, or_chunk = and_events[local], or_events[local]
        age_chunk = ages[local]
        region_chunk = [None, and_chunk, ~and_chunk, exceptions[local]]
        for model_index, prediction_map in enumerate(candidate_arrays):
            raw_prediction = np.asarray(prediction_map[positions_chunk], dtype=np.float32)
            outputs = build_scenarios(raw_prediction, and_chunk, or_chunk, mask_chunk, target_chunk)
            clipped_baseline = np.clip(raw_prediction, -2.0, 2.0).astype(np.float32)
            oracle_prediction, within = clipped_label_run_oracle(
                target_chunk, clipped_baseline, mask_chunk
            )
            if not np.array_equal(oracle_prediction, outputs["oracle_clipped_label_run"]):
                raise AssertionError("scenario builder and oracle partition differ")
            within_variance[model_index] += within
            baseline = raw_prediction
            changed_counts = np.zeros((n_scenarios, 3), dtype=np.int64)
            for scenario_index, scenario_name in enumerate(SCENARIO_NAMES):
                policy_prediction = outputs[scenario_name]
                policy_clipped = np.clip(policy_prediction, -2.0, 2.0).astype(np.float64)
                target_clipped = np.clip(target_chunk.astype(np.float32), -2.0, 2.0).astype(
                    np.float64
                )
                target_weight = np.abs(target_clipped)
                accumulators[model_index][scenario_index].add(
                    target_chunk, policy_prediction, mask_chunk
                )
                for region_index, region_mask in enumerate(region_chunk):
                    selected = mask_chunk if region_mask is None else mask_chunk & region_mask
                    region_accumulators[model_index][scenario_index][region_index].add(
                        target_chunk, policy_prediction, selected
                    )
                    region_counts[model_index, scenario_index, region_index] += int(selected.sum())
                    for component in range(2):
                        weights = target_weight[..., component] * selected
                        region_mass[model_index, scenario_index, region_index, component] += (
                            weights.sum()
                        )
                        error = target_clipped[..., component] - policy_clipped[..., component]
                        region_squared_error[
                            model_index, scenario_index, region_index, component
                        ] += (weights * error * error).sum()
                for component in range(2):
                    selected = mask_chunk
                    buckets = age_chunk[selected]
                    weights = target_weight[..., component][selected]
                    errors = (
                        target_clipped[..., component][selected]
                        - policy_clipped[..., component][selected]
                    )
                    age_counts[model_index, scenario_index, component] += np.bincount(
                        buckets, minlength=len(AGE_LABELS)
                    )
                    age_mass[model_index, scenario_index, component] += np.bincount(
                        buckets, weights=weights, minlength=len(AGE_LABELS)
                    )
                    age_squared_error[model_index, scenario_index, component] += np.bincount(
                        buckets, weights=weights * errors * errors, minlength=len(AGE_LABELS)
                    )
                changed = np.any(policy_prediction != baseline, axis=2)
                changed_counts[scenario_index] = [
                    int(changed.size),
                    int((changed & mask_chunk).sum()),
                    int(changed.sum()),
                ]
                if scenario_index == 0:
                    changed_counts[scenario_index] = [int(changed.size), int(mask_chunk.sum()), 0]
                for batch_row, absolute_index in enumerate(range(start_index, stop)):
                    for component in range(2):
                        y_values = target_clipped[batch_row, :, component][mask_chunk[batch_row]]
                        p_values = policy_clipped[batch_row, :, component][mask_chunk[batch_row]]
                        sequence_moments[model_index, scenario_index, absolute_index, component] = (
                            _centered_moments(y_values, p_values)
                        )
                        sequence_sums[model_index, scenario_index, absolute_index, component] = (
                            wp_sums(y_values, p_values)
                        )
            exposure[model_index] += changed_counts
        if stop == len(positions) or stop % 64 == 0:
            print(f"graded {stop}/{len(positions)} shared row groups", flush=True)

    results_by_model: list[dict[str, object]] = []
    for model_index, candidate in enumerate(inventory_data["candidates"]):
        scenario_results: dict[str, object] = {}
        for scenario_index, name in enumerate(SCENARIO_NAMES):
            acc = accumulators[model_index][scenario_index]
            metric = _prediction_metrics(acc, region_squared_error[model_index, scenario_index, 0])
            if metric["selected_rows"] != int(selected_mask.sum()):
                raise AssertionError("official selected-row count changed across a policy")
            region_metrics = {}
            for region_index, region_name in enumerate(REGION_NAMES):
                region_metrics[region_name] = _prediction_metrics(
                    region_accumulators[model_index][scenario_index][region_index],
                    region_squared_error[model_index, scenario_index, region_index],
                )
            age_metrics = {}
            for component, component_name in enumerate(("t0", "t1")):
                age_metrics[component_name] = [
                    {
                        "age_bin": AGE_LABELS[age_index],
                        "selected_rows": int(
                            age_counts[model_index, scenario_index, component, age_index]
                        ),
                        "weight_mass": float(
                            age_mass[model_index, scenario_index, component, age_index]
                        ),
                        "weighted_mse": (
                            float(
                                age_squared_error[model_index, scenario_index, component, age_index]
                                / age_mass[model_index, scenario_index, component, age_index]
                            )
                            if age_mass[model_index, scenario_index, component, age_index] > 0
                            else None
                        ),
                    }
                    for age_index in range(len(AGE_LABELS))
                ]
            scenario_results[name] = {
                "global": metric,
                "regions": region_metrics,
                "age_weighted_mse": age_metrics,
                "exposure": dict(
                    zip(
                        exposure_names,
                        [int(v) for v in exposure[model_index, scenario_index]],
                        strict=True,
                    )
                ),
            }
        results_by_model.append(
            {
                "cache_name": candidate["cache_name"],
                "metric_scope": "shared complete row groups only; all are fold0",
                "scenarios": scenario_results,
            }
        )

    # The oracle's covariance should be unchanged because each clipped label is constant per run.
    oracle_checks = []
    for model_index, result in enumerate(results_by_model):
        scenarios = result["scenarios"]
        base_acc = accumulators[model_index][SCENARIO_NAMES.index("baseline")]
        oracle_acc = accumulators[model_index][SCENARIO_NAMES.index("oracle_clipped_label_run")]
        target_checks = []
        for component, component_name in enumerate(("t0", "t1")):
            base_moment = base_acc._moments[component]
            oracle_moment = oracle_acc._moments[component]
            total_variance = float(base_moment[4])
            within = float(within_variance[model_index, component])
            between = total_variance - within
            covariance_delta = float(oracle_moment[5] - base_moment[5])
            fraction = within / total_variance if total_variance > 0 else None
            base_wp = float(scenarios["baseline"]["global"][component_name])
            oracle_wp = float(scenarios["oracle_clipped_label_run"]["global"][component_name])
            expected_wp = (
                base_wp / np.sqrt(1.0 - fraction)
                if fraction is not None and fraction < 1.0 and total_variance > 0
                else None
            )
            target_checks.append(
                {
                    "target": component_name,
                    "within_run_prediction_variance_sum": within,
                    "between_run_mean_prediction_variance_sum": between,
                    "total_prediction_variance_sum": total_variance,
                    "variance_partition_relative_error": abs(within + between - total_variance)
                    / max(total_variance, 1e-12),
                    "fraction_prediction_variance_within_runs": fraction,
                    "weighted_covariance_numerator_delta_oracle_minus_base": covariance_delta,
                    "baseline_wp": base_wp,
                    "oracle_wp": oracle_wp,
                    "identity_expected_wp": float(expected_wp) if expected_wp is not None else None,
                    "identity_wp_abs_error": (
                        abs(oracle_wp - float(expected_wp)) if expected_wp is not None else None
                    ),
                }
            )
        oracle_checks.append({"cache_name": result["cache_name"], "targets": target_checks})

    bootstrap = paired_chain_bootstrap(sequence_moments, chain_ids[positions])
    for model_index, _model_results in enumerate(results_by_model):
        for scenario_index, _name in enumerate(SCENARIO_NAMES):
            observed = accumulators[model_index][scenario_index].result()["wp"]
            boot_observed = _score_from_sequence_moments(
                sequence_moments[model_index, scenario_index], np.ones(len(positions))
            )[2]
            if abs(float(observed) - boot_observed) > 5e-10:
                raise AssertionError(
                    "sequence-moment bootstrap baseline differs from canonical scorer"
                )

    post_prediction_hashes = {
        candidate["cache_name"]: sha256_file(args.cache / candidate["cache_name"])
        for candidate in inventory_data["candidates"]
    }
    if any(
        post_prediction_hashes[candidate["cache_name"]] != candidate["prediction"]["sha256"]
        for candidate in inventory_data["candidates"]
    ):
        raise AssertionError("a read-only prediction input changed during the audit")
    sequence_sums_path = args.output_dir / "SEQUENCE_SUMS.npz"
    if sequence_sums_path.exists():
        with np.load(sequence_sums_path, allow_pickle=False) as existing:
            expected_metadata = {
                "row_group_ids": group_ids.astype(np.int64),
                "chain_ids": chain_ids[positions].astype(np.int64),
                "scenario_names": np.asarray(SCENARIO_NAMES),
                "model_names": np.asarray(
                    [candidate["cache_name"] for candidate in inventory_data["candidates"]]
                ),
                "statistic_names": np.asarray(["w", "wy", "wp", "wyy", "wpp", "wyp"]),
            }
            if not np.array_equal(existing["sums"], sequence_sums) or any(
                not np.array_equal(existing[name], expected)
                for name, expected in expected_metadata.items()
            ):
                raise ValueError(
                    "preserved sufficient-statistics artifact differs from deterministic rerun"
                )
    else:
        np.savez_compressed(
            sequence_sums_path,
            sums=sequence_sums,
            row_group_ids=group_ids.astype(np.int64),
            chain_ids=chain_ids[positions].astype(np.int64),
            scenario_names=np.asarray(SCENARIO_NAMES),
            model_names=np.asarray(
                [candidate["cache_name"] for candidate in inventory_data["candidates"]]
            ),
            statistic_names=np.asarray(["w", "wy", "wp", "wyy", "wpp", "wyp"]),
        )
    report: dict[str, object] = {
        "schema": "wunder.prediction-audit.results.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_sha256": sha256_file(Path(__file__)),
        "core_sha256": sha256_file(HERE / "prediction_audit_core.py"),
        "gmetric_sha256": sha256_file(HERE / "gmetric_standalone.py"),
        "inventory_sha256": sha256_file(inventory_path),
        "metric_id": "global_wp_clip2_v1",
        "scenarios_in_fixed_order": list(SCENARIO_NAMES),
        "scope": {
            "common_group_count": int(len(group_ids)),
            "row_groups": group_ids.astype(int).tolist(),
            "folds": common_folds.astype(int).tolist(),
            "selected_rows": int(selected_mask.sum()),
            "selected_mask_sha256": inventory_data["shared_complete_coverage"]["membership_sha256"],
            "weight_mass_by_target": np.abs(
                np.clip(selected_target[selected_mask], -2.0, 2.0).astype(np.float64)
            )
            .sum(axis=0)
            .tolist(),
            "price_trigger_counts": {
                "and_all_rows": int(and_events.sum()),
                "or_all_rows": int(or_events.sum()),
                "and_selected_rows": int((and_events & selected_mask).sum()),
                "or_selected_rows": int((or_events & selected_mask).sum()),
            },
            "selected_target_change_exceptions": int((exceptions & selected_mask).sum()),
            "exception_rule": "any raw t0/t1 change from the preceding row, outside the i0 p0 AND p11 change clock; row zero is false",
        },
        "models": results_by_model,
        "oracle_variance_checks": oracle_checks,
        "paired_chain_bootstrap": bootstrap,
        "per_sequence_sufficient_statistics": {
            "path": str(sequence_sums_path),
            "sha256": sha256_file(sequence_sums_path),
            "shape": list(sequence_sums.shape),
            "statistic_order": ["w", "wy", "wp", "wyy", "wpp", "wyp"],
            "note": "raw additive moments per validation row group, target, and prespecified scenario; supports between/within-sequence pooling decompositions",
        },
        "input_prediction_hashes_before_read": input_prediction_hashes_before,
        "input_prediction_hashes_after_read": post_prediction_hashes,
        "read_only_check": "prediction files retain their inventory SHA-256 after scoring",
        "resource": {
            "elapsed_seconds": time.time() - start,
            "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu_threads": 1,
            "batch_sequences": batch_size,
        },
        "limits": [
            "cached predictions are compared on a single shared complete fold0 slice; rows in incomplete groups are not used",
            "full-valid cache names and checkpoint settings do not prove the historical producer checkpoint/source binding",
            "validation predictions and R1 exposed folds are not claimed as pristine OOF or hidden-test estimates",
            "output holds transform emitted predictions only; the original cached backbone outputs already reflect hidden-state updates on every row",
            "label-run oracle uses future labels and within-run future predictions; it is a noncausal diagnostic, not a quality candidate or deployable model",
            "chain bootstrap intervals are conditional on fixed arrays and do not include fitting or selection uncertainty",
        ],
    }
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "results": str(result_path),
                "common_groups": len(group_ids),
                "selected_rows": int(selected_mask.sum()),
                "wp_by_model_baseline": [
                    model["scenarios"]["baseline"]["global"]["wp"] for model in results_by_model
                ],
                "elapsed_seconds": report["resource"]["elapsed_seconds"],
                "maxrss_kib": report["resource"]["maxrss_kib"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    """Parse the two append-only phases and their fixed data paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("inventory", "grade"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run integrity inventory or fixed prediction-policy grading."""
    args = parse_args()
    if args.phase == "inventory":
        inventory(args)
    else:
        grade(args)


if __name__ == "__main__":
    main()
