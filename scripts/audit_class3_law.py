"""Measure the scorer weight and oracle headroom of class-3 label-law failures.

Future book events are used only to audit known validation labels. They are never
features of a submission prediction.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import decide
import numpy as np
from online_c3 import rebuilt_labels
from sklearn.metrics import roc_auc_score

from wnn.data import SequenceFile
from wnn.gmetric import wp_from_sums, wp_sums
from wnn.splits import valid_row_groups
from wnn.ticks import CLASS3_QUANTUM, reconstruct


def score(sums: np.ndarray) -> float:
    """Return the official two-target pooled score from accumulated moments."""
    return float(np.mean(wp_from_sums(sums)))


def main() -> None:
    """Audit class-3 failures on a fixed random sample of one validation partition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--partition", choices=("dev", "holdout"), default="dev")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.max_sequences < 1:
        parser.error("--max-sequences must be positive")

    with np.load(decide.ROOT / "cache" / "valid_targets.npz") as valid:
        row_groups, classes = valid["row_group"], valid["market_class"]
    path = decide.ROOT / "datasets" / "valid.parquet"
    dev_groups, held_groups = valid_row_groups(path)
    partition_groups = dev_groups if args.partition == "dev" else held_groups
    candidates = np.flatnonzero(np.isin(row_groups, partition_groups) & (classes == 3))
    rng = np.random.default_rng(41)
    selected = np.sort(rng.choice(candidates, min(args.max_sequences, len(candidates)), False))

    p0 = np.load(
        decide.ROOT / "cache" / "pred_ab67_final_since_s4_gcorr_valid.npy",
        mmap_mode="r",
    )
    p1 = np.load(
        decide.ROOT / "cache" / "pred_ab67_final_since_s3_gcorr_valid.npy",
        mmap_mode="r",
    )
    with (decide.ROOT / "cache" / "class3_moves_full.pkl").open("rb") as source:
        moves, spreads = pickle.load(source)
    sequences = SequenceFile(path)
    baseline_sums = np.zeros((2, 6), dtype=np.float64)
    oracle_sums = np.zeros((2, 6), dtype=np.float64)
    total_mass = failure_mass = total_error_mass = failure_error_mass = 0.0
    correct_side_mass = correct_side_error_mass = 0.0
    failure_correct_side_error_mass = 0.0
    known_rows = failure_rows = scored_rows = unknown_transitions = 0
    age_mass = np.zeros((4, 2), dtype=np.float64)
    magnitude_mass = np.zeros((4, 2), dtype=np.float64)
    risk_labels: list[np.ndarray] = []
    risk_weights: list[np.ndarray] = []
    risk_features: dict[str, list[np.ndarray]] = {
        "seed_disagreement": [],
        "negative_prediction_magnitude": [],
        "book_event_age": [],
    }
    failure_patterns: Counter[tuple[int, int, int, int]] = Counter()
    pattern_weight: Counter[tuple[int, int, int, int]] = Counter()

    for index in selected:
        sequence = sequences.read(int(row_groups[index]))
        y = sequence.targets
        mask = sequence.scored
        if y is None or mask is None:
            raise ValueError("Validation sequence must include labels and score mask")
        prediction = 0.5 * (
            np.asarray(p0[index], dtype=np.float32) + np.asarray(p1[index], dtype=np.float32)
        )
        ask, bid, unknown = reconstruct(sequence.features, moves, spreads)
        rebuilt, known_at = rebuilt_labels(ask, bid)
        is_scored = mask & (np.arange(len(mask)) >= decide.WARMUP)
        is_known = known_at < len(mask)
        is_failure = is_known & np.any(np.abs(rebuilt - y) > 1e-3, axis=1)
        scored_failure = is_scored & is_failure
        clipped_y = np.clip(y, -2, 2)
        weight = np.abs(clipped_y).sum(axis=1)
        error = (np.abs(clipped_y - np.clip(prediction, -2, 2)) * np.abs(clipped_y)).sum(axis=1)
        total_mass += float(weight[is_scored].sum())
        failure_mass += float(weight[scored_failure].sum())
        total_error_mass += float(error[is_scored].sum())
        failure_error_mass += float(error[scored_failure].sum())
        has_side = np.abs(y[:, 0] - y[:, 1]) > 1e-3
        correct_side = has_side & (np.argmax(y, axis=1) == np.argmax(prediction, axis=1))
        correct_side_mass += float(weight[is_scored & correct_side].sum())
        correct_side_error_mass += float(error[is_scored & correct_side].sum())
        failure_correct_side_error_mass += float(error[scored_failure & correct_side].sum())
        if scored_failure.any():
            pattern = np.rint(
                np.column_stack((y[scored_failure], rebuilt[scored_failure])) / CLASS3_QUANTUM
            ).astype(np.int16)
            unique, inverse, counts = np.unique(
                pattern, axis=0, return_inverse=True, return_counts=True
            )
            masses = np.bincount(inverse, weights=weight[scored_failure])
            for ticks, count, mass in zip(unique, counts, masses, strict=True):
                key = tuple(int(value) for value in ticks)
                failure_patterns[key] += int(count)
                pattern_weight[key] += float(mass)
        known_rows += int(np.sum(is_scored & is_known))
        failure_rows += int(scored_failure.sum())
        scored_rows += int(is_scored.sum())
        unknown_transitions += unknown

        event = np.r_[True, (np.diff(ask) != 0) | (np.diff(bid) != 0)]
        last_event = np.maximum.accumulate(np.where(event, np.arange(len(event)), 0))
        age_bin = np.searchsorted((4, 16, 64), np.arange(len(event)) - last_event)
        risk_rows = is_scored & is_known
        risk_labels.append(is_failure[risk_rows])
        risk_weights.append(weight[risk_rows])
        risk_features["seed_disagreement"].append(
            np.mean(np.abs(p0[index, risk_rows] - p1[index, risk_rows]), axis=1)
        )
        risk_features["negative_prediction_magnitude"].append(
            -np.mean(np.abs(prediction[risk_rows]), axis=1)
        )
        risk_features["book_event_age"].append(
            np.arange(len(event))[risk_rows] - last_event[risk_rows]
        )
        for age in range(4):
            age_rows = is_scored & (age_bin == age)
            age_mass[age, 0] += weight[age_rows].sum()
            age_mass[age, 1] += weight[age_rows & is_failure].sum()
        magnitude_bin = np.searchsorted((0.5, 1.0, 1.5), np.abs(clipped_y))
        for magnitude in range(4):
            magnitude_rows = is_scored[:, None] & (magnitude_bin == magnitude)
            magnitude_mass[magnitude, 0] += np.abs(clipped_y)[magnitude_rows].sum()
            failure_rows_for_magnitude = scored_failure[:, None] & (magnitude_bin == magnitude)
            magnitude_mass[magnitude, 1] += np.abs(clipped_y)[failure_rows_for_magnitude].sum()

        for target in range(2):
            original = prediction[:, target]
            oracle = original.copy()
            oracle[scored_failure] = y[scored_failure, target]
            baseline_sums[target] += wp_sums(y[:, target], original, is_scored)
            oracle_sums[target] += wp_sums(y[:, target], oracle, is_scored)

    labels = np.concatenate(risk_labels)
    risk_weight = np.concatenate(risk_weights)
    risk_auc = {
        name: float(roc_auc_score(labels, np.concatenate(values), sample_weight=risk_weight))
        for name, values in risk_features.items()
    }
    result = {
        "partition": args.partition,
        "sampled_sequences": int(len(selected)),
        "row_groups": row_groups[selected].tolist(),
        "scored_rows": scored_rows,
        "known_scored_rows": known_rows,
        "failure_scored_rows": failure_rows,
        "failure_row_fraction_of_known": failure_rows / known_rows,
        "failure_fraction_of_target_weight": failure_mass / total_mass,
        "failure_fraction_of_weighted_absolute_error": failure_error_mass / total_error_mass,
        "correct_side_fraction_of_target_weight": correct_side_mass / total_mass,
        "correct_side_fraction_of_weighted_absolute_error": correct_side_error_mass
        / total_error_mass,
        "failure_correct_side_fraction_of_weighted_absolute_error": (
            failure_correct_side_error_mass / total_error_mass
        ),
        "class3_sample_score": score(baseline_sums),
        "oracle_failure_rows_score": score(oracle_sums),
        "oracle_delta": score(oracle_sums) - score(baseline_sums),
        "age_bins": [
            {"target_weight": float(total), "failure_weight": float(failed)}
            for total, failed in age_mass
        ],
        "target_magnitude_bins": [
            {"target_weight": float(total), "failure_weight": float(failed)}
            for total, failed in magnitude_mass
        ],
        "causal_feature_auc_for_law_failure": risk_auc,
        "largest_failure_patterns_by_target_weight": [
            {"true_then_rebuilt_ticks": key, "rows": failure_patterns[key], "weight": mass}
            for key, mass in pattern_weight.most_common(15)
        ],
        "unknown_book_transitions": unknown_transitions,
        "oracle_is_diagnostic_only": True,
    }
    output = args.output or (
        decide.ROOT / "research" / "artifacts" / f"class3_law_audit_{args.partition}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
