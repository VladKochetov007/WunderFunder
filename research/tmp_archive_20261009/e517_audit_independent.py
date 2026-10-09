from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
LINEAGE = ROOT / "research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204"
E514 = ROOT / "research/campaign_2026-10-06/bandit_continuation/conditional_weak_portfolio_E514"
for path in (ROOT / "src", LINEAGE):
    sys.path.insert(0, str(path))

from disagreement_screen import load_e514_inputs  # noqa: E402
from wnn.gmetric import GlobalAccumulator, _merge_moments, _wp_from_moments, global_wp  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    if not array.flags.c_contiguous:
        raise ValueError(f"array is not C-contiguous: {array.shape}")
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def beta_fit(y: np.ndarray, anchor: np.ndarray, delta: np.ndarray, selected: np.ndarray) -> float:
    weights = np.abs(y[selected])
    d = delta[selected]
    denom = float(np.sum(weights * d * d, dtype=np.float64))
    if denom <= np.finfo(np.float64).eps:
        return 0.0
    numer = float(np.sum(weights * d * (y[selected] - anchor[selected]), dtype=np.float64))
    return float(np.clip(numer / denom, 0.0, 1.0))


def official_chain_score(targets, predictions, mask, chain_rows):
    accumulator = GlobalAccumulator()
    for rows in chain_rows:
        accumulator.add(targets[rows], predictions[rows], mask[rows])
    return accumulator.result()


def main() -> None:
    result_path = LINEAGE / "E517_RESULT.json"
    attempt_path = LINEAGE / "E517_ATTEMPT_STARTED.json"
    result = json.loads(result_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    prereg_path = LINEAGE / "GATED_BLEND_PREREGISTRATION_E517.md"
    runner_path = LINEAGE / "disagreement_gate.py"
    loader_path = LINEAGE / "disagreement_screen.py"
    notebook_path = LINEAGE / "E517_single_action_eda.py"
    partition_path = E514 / "E514_CHAIN_PARTITION.json"
    manifest_path = E514 / "SOURCE_MANIFEST_E514_V3.json"
    gated_path = LINEAGE / "E517_GATED_OOF_PREDICTIONS.npy"
    global_path = LINEAGE / "E517_GLOBAL_OOF_PREDICTIONS.npy"
    plot_path = LINEAGE / "plots/E517_FOLD_GATE_RESULTS.png"
    chain_csv_path = LINEAGE / "plots/E517_CHAIN_SUMMARY.csv"

    inputs = load_e514_inputs(ROOT)
    source_manifest_digest = inputs.source_manifest_sha256
    partition_digest = inputs.chain_partition_sha256
    row_groups_digest = hashlib.sha256(
        inputs.row_groups.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    targets = inputs.targets
    is_scored = inputs.is_scored
    anchor = inputs.anchor
    ab12 = inputs.candidate_predictions["ab12"]
    chain_by_sequence = inputs.chain_by_sequence
    chain_to_fold = dict(inputs.chain_to_outer_fold)
    row_groups = inputs.row_groups
    del inputs

    gated = np.load(gated_path, mmap_mode="r", allow_pickle=False)
    global_blend = np.load(global_path, mmap_mode="r", allow_pickle=False)
    expected_shape = targets.shape
    print("INPUT", json.dumps({
        "shape": list(expected_shape),
        "dtype_targets": str(targets.dtype),
        "dtype_mask": str(is_scored.dtype),
        "groups": int(expected_shape[0]),
        "chains": int(np.unique(chain_by_sequence).size),
        "source_manifest_sha256": source_manifest_digest,
        "partition_sha256": partition_digest,
        "row_groups_sha256": row_groups_digest,
        "fold_chain_counts": {
            str(fold): int(sum(chain_to_fold[int(c)] == fold for c in np.unique(chain_by_sequence)))
            for fold in (0, 1, 2)
        },
        "fold_group_counts": {
            str(fold): int(sum(chain_to_fold[int(chain)] == fold for chain in chain_by_sequence))
            for fold in (0, 1, 2)
        },
    }, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))

    digest_checks = {
        "manifest_file": (source_manifest_digest, sha256_file(manifest_path)),
        "partition_file": (partition_digest, sha256_file(partition_path)),
        "row_groups": (result["row_groups_sha256"], row_groups_digest),
        "result_manifest": (result["source_manifest_sha256"], source_manifest_digest),
        "result_partition": (result["chain_partition_sha256"], partition_digest),
        "result_runner": (result["implementation_sha256"], sha256_file(runner_path)),
        "result_loader": (result["input_loader_sha256"], sha256_file(loader_path)),
        "result_prereg": (result["preregistration_sha256"], sha256_file(prereg_path)),
        "result_notebook": (result["notebook_sha256"], sha256_file(notebook_path)),
        "result_plot": (result["plot_sha256"], sha256_file(plot_path)),
        "chain_summary_file": (result.get("chain_summary_sha256", "NOT_REGISTERED"), sha256_file(chain_csv_path)),
        "current_full_notebook": (result.get("notebook_sha256", ""), sha256_file(LINEAGE / "H204_complementarity_eda.py")),
        "result_gated_raw": (result["oof_gated_sha256"], sha256_array(gated)),
        "result_global_raw": (result["oof_global_sha256"], sha256_array(global_blend)),
        "result_gated_file": (result["oof_gated_file_sha256"], sha256_file(gated_path)),
        "result_global_file": (result["oof_global_file_sha256"], sha256_file(global_path)),
    }
    print("ATTEMPT", json.dumps({
        "experiment_id": attempt.get("experiment_id"),
        "registered_seed": attempt.get("registered_bootstrap_seed"),
        "result_seed": result.get("bootstrap_seed"),
        "input_load_mode": attempt.get("input_load_mode"),
        "result_load_mode": result.get("input_load_mode"),
    }, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
    print("HASH_CHECKS", json.dumps({
        name: {"match": left == right, "expected": left, "observed": right}
        for name, (left, right) in digest_checks.items()
    }, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))

    if gated.shape != expected_shape or global_blend.shape != expected_shape:
        raise ValueError("saved OOF prediction shape mismatch")
    if gated.dtype != np.float32 or global_blend.dtype != np.float32:
        raise ValueError("saved OOF prediction dtype mismatch")
    if result["bootstrap_seed"] != 3_772_181_504 or attempt["registered_bootstrap_seed"] != 3_772_181_504:
        raise ValueError("registered bootstrap seed mismatch")

    warmup = np.arange(targets.shape[1]) >= 99
    metric_mask = is_scored & warmup[None, :]
    fold_by_sequence = np.asarray([chain_to_fold[int(c)] for c in chain_by_sequence], dtype=np.int8)
    chain_ids = np.unique(chain_by_sequence).astype(np.int64)
    fold_chain_ids = {
        fold: np.asarray([c for c in chain_ids if chain_to_fold[int(c)] == fold], dtype=np.int64)
        for fold in (0, 1, 2)
    }
    computed_gated = anchor.copy()
    computed_global = anchor.copy()
    gate_mass_records = []
    fold_score_records = []

    for fold in (0, 1, 2):
        eval_rows = np.flatnonzero(fold_by_sequence == fold)
        cal_rows = np.flatnonzero(fold_by_sequence != fold)
        cal_mask = metric_mask[cal_rows]
        y_cal = np.clip(targets[cal_rows, :, 0].astype(np.float64), -2.0, 2.0)
        a_cal = np.clip(anchor[cal_rows, :, 0].astype(np.float64), -2.0, 2.0)
        c_cal = np.clip(ab12[cal_rows, :, 0].astype(np.float64), -2.0, 2.0)
        delta_cal = c_cal - a_cal
        threshold = float(np.quantile(np.abs(delta_cal[cal_mask]), 0.75, method="linear"))
        calibration_gate = np.abs(delta_cal) >= threshold
        selected_cal_gate = cal_mask & calibration_gate
        beta_gate = beta_fit(y_cal, a_cal, delta_cal, selected_cal_gate)
        beta_global = beta_fit(y_cal, a_cal, delta_cal, cal_mask)
        cal_weight = np.abs(y_cal)
        cal_rows_n = int(cal_mask.sum())
        cal_gated_n = int(selected_cal_gate.sum())
        cal_total_weight = float(cal_weight[cal_mask].sum(dtype=np.float64))
        cal_gated_weight = float(cal_weight[selected_cal_gate].sum(dtype=np.float64))

        y_eval = np.clip(targets[eval_rows, :, 0].astype(np.float64), -2.0, 2.0)
        a_eval = np.clip(anchor[eval_rows, :, 0].astype(np.float64), -2.0, 2.0)
        c_eval = np.clip(ab12[eval_rows, :, 0].astype(np.float64), -2.0, 2.0)
        delta_eval = c_eval - a_eval
        eval_gate = np.abs(delta_eval) >= threshold
        eval_mask = metric_mask[eval_rows]
        selected_eval_gate = eval_mask & eval_gate
        computed_gated[eval_rows, :, 0] = np.clip(
            a_eval + beta_gate * eval_gate * delta_eval, -2.0, 2.0
        ).astype(np.float32)
        computed_global[eval_rows, :, 0] = np.clip(
            a_eval + beta_global * delta_eval, -2.0, 2.0
        ).astype(np.float32)
        eval_rows_n = int(eval_mask.sum())
        eval_weight = np.abs(y_eval)
        eval_total_weight = float(eval_weight[eval_mask].sum(dtype=np.float64))
        eval_gated_weight = float(eval_weight[selected_eval_gate].sum(dtype=np.float64))

        stored_fold = result["folds"][fold]
        fit = stored_fold["fit"]
        mass = stored_fold["evaluation_gate"]
        checks = {
            "threshold": np.isclose(threshold, fit["threshold"], rtol=0, atol=1e-12),
            "beta_gate": np.isclose(beta_gate, fit["gated_beta"], rtol=0, atol=1e-12),
            "beta_global": np.isclose(beta_global, fit["global_beta"], rtol=0, atol=1e-12),
            "calibration_rows": cal_rows_n == fit["calibration_rows"],
            "calibration_gated_rows": cal_gated_n == fit["calibration_gated_rows"],
            "calibration_row_fraction": np.isclose(cal_gated_n / cal_rows_n, fit["calibration_gate_fraction"], rtol=0, atol=1e-12),
            "calibration_weight_fraction": np.isclose(cal_gated_weight / cal_total_weight, fit["calibration_weighted_gate_fraction"], rtol=0, atol=1e-12),
            "evaluation_scored_rows": eval_rows_n == mass["scored_row_count"],
            "evaluation_gated_rows": int(selected_eval_gate.sum()) == mass["gated_scored_row_count"],
            "evaluation_row_fraction": np.isclose(selected_eval_gate.sum() / eval_rows_n, mass["gated_scored_row_fraction"], rtol=0, atol=1e-12),
            "evaluation_weight_fraction": np.isclose(eval_gated_weight / eval_total_weight, mass["gated_official_weight_fraction"], rtol=0, atol=1e-12),
            "evaluation_chains": int(fold_chain_ids[fold].size) == stored_fold["evaluation_chain_count"] == 41,
            "calibration_chains": 123 - int(fold_chain_ids[fold].size) == stored_fold["calibration_chain_count"] == 82,
        }
        gate_mass_records.append({
            "fold": fold,
            "threshold": threshold,
            "beta_gate": beta_gate,
            "beta_global": beta_global,
            "calibration_rows": cal_rows_n,
            "calibration_gated_rows": cal_gated_n,
            "calibration_row_fraction": cal_gated_n / cal_rows_n,
            "calibration_weight_fraction": cal_gated_weight / cal_total_weight,
            "evaluation_scored_rows": eval_rows_n,
            "evaluation_gated_rows": int(selected_eval_gate.sum()),
            "evaluation_row_fraction": selected_eval_gate.sum() / eval_rows_n,
            "evaluation_weight_fraction": eval_gated_weight / eval_total_weight,
            "checks": checks,
        })
        for model_name, pred in (("anchor", anchor), ("gated_ab12", gated), ("global_ab12", global_blend)):
            actual = global_wp(targets[eval_rows], pred[eval_rows], metric_mask[eval_rows])
            stored = stored_fold[model_name]
            fold_score_records.append({
                "fold": fold,
                "model": model_name,
                "official": actual,
                "stored_wp": stored["wp"],
                "wp_match": np.isclose(actual["wp"], stored["wp"], rtol=0, atol=1e-12),
                "selected_rows_match": actual["selected_rows"] == stored["selected_rows"],
            })

    exact_gated_equal = bool(np.array_equal(computed_gated, gated))
    exact_global_equal = bool(np.array_equal(computed_global, global_blend))
    target1_anchor = bool(np.array_equal(gated[:, :, 1], anchor[:, :, 1]) and np.array_equal(global_blend[:, :, 1], anchor[:, :, 1]))
    print("REBUILT_OOF", json.dumps({
        "gated_exact_match": exact_gated_equal,
        "global_exact_match": exact_global_equal,
        "target1_unchanged_anchor": target1_anchor,
        "gated_equals_global": bool(np.array_equal(gated, global_blend)),
        "gated_changed_values": int(np.count_nonzero(gated != anchor)),
        "global_changed_values": int(np.count_nonzero(global_blend != anchor)),
        "positive_gate_folds": int(sum(record["beta_gate"] > 0 for record in gate_mass_records)),
    }, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
    print("FOLD_GATE_MASS", json.dumps(gate_mass_records, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
    print("FOLD_ENDPOINTS", json.dumps(fold_score_records, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))

    endpoint_records = {}
    for model_name, pred in (("anchor", anchor), ("gated_ab12", gated), ("global_ab12", global_blend)):
        actual = global_wp(targets, pred, metric_mask)
        stored = result[model_name]
        endpoint_records[model_name] = {
            "official": actual,
            "stored": stored,
            "wp_match": np.isclose(actual["wp"], stored["wp"], rtol=0, atol=1e-12),
            "selected_rows_match": actual["selected_rows"] == stored["selected_rows"],
        }
    print("POOLED_ENDPOINTS", json.dumps(endpoint_records, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))

    model_arrays = (anchor, gated, global_blend)
    chain_moments: list[dict[int, np.ndarray]] = [dict() for _ in model_arrays]
    rows_by_chain = {int(chain): np.flatnonzero(chain_by_sequence == chain) for chain in chain_ids}
    for chain in chain_ids:
        rows = rows_by_chain[int(chain)]
        for model_index, prediction in enumerate(model_arrays):
            accumulator = GlobalAccumulator()
            accumulator.add(targets[rows], prediction[rows], metric_mask[rows])
            chain_moments[model_index][int(chain)] = accumulator._moments.copy()

    rng = np.random.default_rng(3_772_181_504)
    boot_deltas = np.empty((2_000, 2), dtype=np.float64)
    for replicate in range(2_000):
        totals = [np.zeros((2, 6), dtype=np.float64) for _ in model_arrays]
        for fold in (0, 1, 2):
            fold_members = fold_chain_ids[fold]
            sampled = rng.choice(fold_members, size=fold_members.size, replace=True)
            for chain in sampled:
                for model_index in range(len(model_arrays)):
                    totals[model_index] = np.stack([
                        _merge_moments(totals[model_index][target_index], chain_moments[model_index][int(chain)][target_index])
                        for target_index in (0, 1)
                    ])
        scores = [0.5 * sum(_wp_from_moments(m[target_index]) for target_index in (0, 1)) for m in totals]
        boot_deltas[replicate, 0] = scores[1] - scores[0]
        boot_deltas[replicate, 1] = scores[2] - scores[0]
    ci_gated_anchor = np.quantile(boot_deltas[:, 0], [0.025, 0.975]).tolist()
    ci_global_anchor = np.quantile(boot_deltas[:, 1], [0.025, 0.975]).tolist()
    ci_gated_global = np.quantile(boot_deltas[:, 0] - boot_deltas[:, 1], [0.025, 0.975]).tolist()
    print("BOOTSTRAP", json.dumps({
        "seed": 3_772_181_504,
        "replicates": 2_000,
        "conditional_on_fitted_fold_rules": True,
        "gated_minus_anchor_ci95": ci_gated_anchor,
        "global_minus_anchor_ci95": ci_global_anchor,
        "gated_minus_global_ci95": ci_gated_global,
        "stored_gated_minus_anchor_ci95": result["paired_chain_bootstrap_delta_gated_vs_anchor_ci95"],
        "stored_global_minus_anchor_ci95": result["paired_chain_bootstrap_delta_global_vs_anchor_ci95"],
        "stored_gated_minus_global_ci95": result["paired_chain_bootstrap_delta_gated_vs_global_ci95"],
    }, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))

    positive_gate_folds = int(sum(record["beta_gate"] > 0 for record in gate_mass_records))
    independently_passed = bool(
        endpoint_records["gated_ab12"]["official"]["wp"] > endpoint_records["anchor"]["official"]["wp"]
        and endpoint_records["gated_ab12"]["official"]["wp"] > endpoint_records["global_ab12"]["official"]["wp"]
        and ci_gated_anchor[0] > 0.0
        and ci_gated_global[0] > 0.0
        and positive_gate_folds >= 2
    )
    print("INDEPENDENT_DECISION", json.dumps({
        "pass": independently_passed,
        "stored_pass_flag": result["development_gate_passed"],
        "positive_gate_folds": positive_gate_folds,
        "condition_gated_above_anchor": endpoint_records["gated_ab12"]["official"]["wp"] > endpoint_records["anchor"]["official"]["wp"],
        "condition_gated_above_global": endpoint_records["gated_ab12"]["official"]["wp"] > endpoint_records["global_ab12"]["official"]["wp"],
        "condition_gated_anchor_lower_ci_positive": ci_gated_anchor[0] > 0.0,
        "condition_gated_global_lower_ci_positive": ci_gated_global[0] > 0.0,
        "all_chains_groups_exposed": True,
    }, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))


if __name__ == "__main__":
    main()
