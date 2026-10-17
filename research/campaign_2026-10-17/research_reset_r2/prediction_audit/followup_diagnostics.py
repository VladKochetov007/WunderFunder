#!/usr/bin/env python3
"""Fixed-slice cross-model and age-by-market-class diagnostics; no fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from gmetric_standalone import _merge_moments
from prediction_audit_core import AGE_LABELS, age_buckets, event_flags, hold_until_event

POLICY_NAMES = (
    ("pred_t384_s3_full_valid.npy", "t384_s3"),
    ("pred_t384_s4_full_valid.npy", "t384_s4"),
    ("pred_dftt_tlead_s3_valid.npy", "dftt_tlead_s3"),
)
PAIRS = ((0, 1), (0, 2), (1, 2))
PAIR_LABELS = (("t384_s3", "t384_s4"), ("t384_s3", "dftt_tlead_s3"), ("t384_s4", "dftt_tlead_s3"))
AGE_EDGES = np.asarray([0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 20_001], dtype=np.int32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_moments(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> np.ndarray:
    positive = weight > 0
    if not positive.any():
        return np.zeros(6, dtype=np.float64)
    x = left[positive].astype(np.float64, copy=False)
    y = right[positive].astype(np.float64, copy=False)
    w = weight[positive].astype(np.float64, copy=False)
    total = float(w.sum())
    x_shift, y_shift = x - x[0], y - y[0]
    mean_x_shift = float(np.sum(w * x_shift) / total)
    mean_y_shift = float(np.sum(w * y_shift) / total)
    dx, dy = x_shift - mean_x_shift, y_shift - mean_y_shift
    return np.asarray(
        [
            total,
            float(x[0] + mean_x_shift),
            float(y[0] + mean_y_shift),
            float(np.sum(w * dx * dx)),
            float(np.sum(w * dy * dy)),
            float(np.sum(w * dx * dy)),
        ],
        dtype=np.float64,
    )


def correlation(moments: np.ndarray) -> float:
    if moments[0] <= 1e-8 or moments[3] <= 0 or moments[4] <= 0:
        return 0.0
    return float(np.clip(moments[5] / np.sqrt(moments[3] * moments[4]), -1.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory_path = args.output_dir / "INVENTORY.json"
    alignment_path = args.output_dir / "ALIGNMENT.json"
    results_path = args.output_dir / "RESULTS.json"
    output_path = args.output_dir / "FOLLOWUP_DIAGNOSTICS.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    results_bytes = results_path.read_bytes()
    if alignment["checks"] and any(alignment["checks"].values()):
        raise ValueError("cache/parquet alignment artifact has mismatches")
    if (
        alignment["scope"]["membership_sha256"]
        != inventory["shared_complete_coverage"]["membership_sha256"]
    ):
        raise ValueError("alignment and inventory row membership differ")

    candidates = [args.cache / name for name, _ in POLICY_NAMES]
    initial_hashes = {path.name: sha256_file(path) for path in candidates}
    expected_hashes = {
        item["cache_name"]: item["prediction"]["sha256"] for item in inventory["candidates"]
    }
    if any(initial_hashes[path.name] != expected_hashes[path.name] for path in candidates):
        raise ValueError("candidate prediction changed since frozen inventory")
    predictions = [np.load(path, mmap_mode="r", allow_pickle=False) for path in candidates]
    with np.load(args.cache / "valid_targets.npz", allow_pickle=False) as archive:
        target_cache = archive["y"]
        mask_cache = archive["mask"].astype(bool, copy=False)
        row_groups = archive["row_group"].astype(np.int64, copy=False)
        classes = archive["market_class"].astype(np.int64, copy=False)
        seq_ids = archive["seq_ix"].astype(np.int64, copy=False)
    position_by_group = {int(group): position for position, group in enumerate(row_groups)}
    groups = [int(group) for group in inventory["shared_complete_coverage"]["row_groups"]]
    sequence_ids = np.asarray([seq_ids[position_by_group[group]] for group in groups])
    group_positions = np.asarray([position_by_group[group] for group in groups], dtype=np.int64)
    parquet = pq.ParquetFile(args.data / "valid.parquet")
    if parquet.metadata.num_row_groups != len(row_groups):
        raise ValueError("parquet row-group count changed")
    n_time = target_cache.shape[1]
    row = np.arange(n_time, dtype=np.int32)
    expected_need = row >= 99

    # scope, pair, target, six centered weighted moments
    moments = np.zeros((3, len(PAIRS), 2, 6), dtype=np.float64)
    scope_names = ("clipped_predictions", "clipped_residuals", "within_label_run_fluctuations")
    class_values = sorted(set(int(value) for value in classes[group_positions]))
    class_names = {
        -1: "unclassified",
        0: "quantum_0",
        1: "quantum_1",
        2: "quantum_2",
        3: "quantum_3",
    }
    # model x scenario(baseline/AND hold) x class x age x target x (count,mass,MSE numerator,target variance)
    strata = np.zeros((3, 2, len(class_values), len(AGE_LABELS), 2, 4), dtype=np.float64)
    start = time.time()
    for group_index, (group, position) in enumerate(zip(groups, group_positions, strict=True)):
        table = parquet.read_row_group(
            group,
            columns=["seq_ix", "step_in_seq", "need_prediction", "i0_p0", "i0_p11"],
            use_threads=False,
        )
        if table.num_rows != n_time:
            raise ValueError(f"row group {group} has an unexpected row count")
        seq = table["seq_ix"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        step = table["step_in_seq"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        need = table["need_prediction"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
        if (
            not np.all(seq == sequence_ids[group_index])
            or not np.array_equal(step, row)
            or not np.array_equal(need, expected_need)
        ):
            raise ValueError(f"row identity or need_prediction mismatch in group {group}")
        p0 = table["i0_p0"].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        p11 = table["i0_p11"].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        and_flag, _ = event_flags(p0[None, :], p11[None, :])
        age = age_buckets(and_flag)[0]
        raw_y = target_cache[position].astype(np.float32, copy=False)
        y = np.clip(raw_y, -2.0, 2.0).astype(np.float32)
        selected = mask_cache[position] & expected_need
        weight = np.abs(y).astype(np.float64)
        raw_outputs = [
            np.asarray(prediction[position], dtype=np.float32) for prediction in predictions
        ]
        held_outputs = [
            hold_until_event(output[None, :, :], and_flag)[0][0] for output in raw_outputs
        ]
        clipped_predictions = [
            np.clip(output, -2.0, 2.0).astype(np.float32) for output in raw_outputs
        ]
        clipped_held = [np.clip(output, -2.0, 2.0).astype(np.float32) for output in held_outputs]
        errors = [y - output for output in clipped_predictions]

        for target_index in range(2):
            target_selected = selected
            w = weight[:, target_index]
            label = y[:, target_index]
            selected_run = target_selected
            run_starts = np.empty(n_time, dtype=bool)
            run_starts[0] = True
            run_starts[1:] = label[1:] != label[:-1]
            run_ids = np.cumsum(run_starts, dtype=np.int32) - 1
            selected_run_ids = run_ids[selected_run]
            run_count = int(run_starts.sum())
            run_sizes = np.bincount(selected_run_ids, minlength=run_count)
            run_means = []
            for model_index in range(3):
                pred_selected = clipped_predictions[model_index][selected_run, target_index].astype(
                    np.float64
                )
                run_sum = np.bincount(selected_run_ids, weights=pred_selected, minlength=run_count)
                means = np.divide(
                    run_sum, run_sizes, out=np.zeros_like(run_sum), where=run_sizes > 0
                )
                run_means.append(pred_selected - means[selected_run_ids])
            for pair_index, (left, right) in enumerate(PAIRS):
                left_pred = clipped_predictions[left][selected, target_index]
                right_pred = clipped_predictions[right][selected, target_index]
                left_resid = errors[left][selected, target_index]
                right_resid = errors[right][selected, target_index]
                w_selected = w[selected]
                for scope_index, (a, b) in enumerate(
                    (
                        (left_pred, right_pred),
                        (left_resid, right_resid),
                        (run_means[left], run_means[right]),
                    )
                ):
                    added = weighted_moments(a, b, w_selected)
                    moments[scope_index, pair_index, target_index] = _merge_moments(
                        moments[scope_index, pair_index, target_index], added
                    )

        class_value = int(classes[position])
        class_index = class_values.index(class_value)
        for target_index in range(2):
            target_selected = selected
            label = y[:, target_index]
            w = weight[:, target_index]
            for age_index in range(len(AGE_LABELS)):
                idx = target_selected & (age == age_index)
                if not idx.any():
                    continue
                w_selected = w[idx]
                y_selected = label[idx]
                target_var = weighted_moments(y_selected, y_selected, w_selected)[3]
                for model_index in range(3):
                    for scenario_index, prediction in enumerate(
                        (clipped_predictions[model_index], clipped_held[model_index])
                    ):
                        error = y_selected - prediction[idx, target_index]
                        cell = strata[
                            model_index, scenario_index, class_index, age_index, target_index
                        ]
                        cell[0] += int(idx.sum())
                        cell[1] += float(w_selected.sum())
                        cell[2] += float(np.sum(w_selected * error.astype(np.float64) ** 2))
                        cell[3] += target_var
        if (group_index + 1) % 64 == 0 or group_index + 1 == len(groups):
            print(f"diagnosed {group_index + 1}/{len(groups)} shared row groups", flush=True)

    pair_results: dict[str, object] = {}
    for scope_index, scope_name in enumerate(scope_names):
        pair_results[scope_name] = {}
        for pair_index, pair in enumerate(PAIR_LABELS):
            pair_results[scope_name]["/".join(pair)] = {
                target_name: {
                    "weighted_correlation": correlation(
                        moments[scope_index, pair_index, target_index]
                    ),
                    "weighted_covariance_sum": float(
                        moments[scope_index, pair_index, target_index, 5]
                    ),
                    "weighted_variance_sums": [
                        float(moments[scope_index, pair_index, target_index, 3]),
                        float(moments[scope_index, pair_index, target_index, 4]),
                    ],
                    "weight_mass": float(moments[scope_index, pair_index, target_index, 0]),
                }
                for target_index, target_name in enumerate(("t0", "t1"))
            }

    strata_records = []
    for model_index, (_, model_name) in enumerate(POLICY_NAMES):
        for class_index, class_value in enumerate(class_values):
            for age_index, age_label in enumerate(AGE_LABELS):
                for target_index, target_name in enumerate(("t0", "t1")):
                    baseline_cell = strata[model_index, 0, class_index, age_index, target_index]
                    hold_cell = strata[model_index, 1, class_index, age_index, target_index]
                    mass = float(baseline_cell[1])
                    if baseline_cell[0] == 0:
                        continue
                    strata_records.append(
                        {
                            "model": model_name,
                            "market_class": class_names.get(class_value, f"class_{class_value}"),
                            "market_class_id": class_value,
                            "age_bin_since_last_i0_and_event": age_label,
                            "target": target_name,
                            "selected_rows": int(baseline_cell[0]),
                            "weight_mass": mass,
                            "weighted_target_variance_sum": float(baseline_cell[3]),
                            "baseline_weighted_mse": float(baseline_cell[2] / mass)
                            if mass
                            else None,
                            "and_hold_weighted_mse": float(hold_cell[2] / hold_cell[1])
                            if hold_cell[1]
                            else None,
                        }
                    )

    final_hashes = {path.name: sha256_file(path) for path in candidates}
    if final_hashes != initial_hashes:
        raise AssertionError("a prediction cache changed during follow-up diagnostics")
    report = {
        "schema": "wunder.prediction-audit.followup-diagnostics.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_sha256": sha256_file(Path(__file__)),
        "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "alignment_sha256": hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
        "scope": {
            "row_group_count": len(groups),
            "row_groups": groups,
            "folds": inventory["shared_complete_coverage"]["fold_counts"],
            "selected_rows": int(alignment["scope"]["official_selected_rows"]),
            "membership_sha256": inventory["shared_complete_coverage"]["membership_sha256"],
            "prediction_hashes_before_and_after": initial_hashes,
        },
        "cross_model_weighted_correlations": pair_results,
        "age_by_market_class_weighted_mse": {
            "selection": "canonical need_prediction AND is_scored",
            "weights": "abs(clipped target)",
            "target_variance": "weighted centered variance sum within each age-by-class stratum",
            "records": strata_records,
        },
        "limits": [
            "all correlations assume the historical cached arrays share the stated row-group order; prediction receipts are absent and array order is unverified",
            "market_class is a retrospective target-derived label, used only as a descriptive stratum and never for prediction",
            "ages reset at sequence start and are left-censored at that boundary",
            "these are conditional fixed-array diagnostics; no fit, calibration, coefficient choice, or model inference was performed",
        ],
        "resource": {
            "elapsed_seconds": time.time() - start,
            "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu_threads": 1,
        },
    }
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "elapsed_seconds": report["resource"]["elapsed_seconds"],
                "maxrss_kib": report["resource"]["maxrss_kib"],
                "strata_records": len(strata_records),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
