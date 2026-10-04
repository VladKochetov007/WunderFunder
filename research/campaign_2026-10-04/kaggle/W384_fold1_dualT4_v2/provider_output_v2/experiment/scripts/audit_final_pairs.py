"""Compare final two-GRU packages on the reserved validation sequences.

The primary score uses the published mask and global weighted Pearson. Mask
softening and class-mix changes are sensitivity checks, not score forecasts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import decide
import numpy as np
from probe_pure_pair import pair_sums

from wnn.splits import valid_row_groups

PAIRS = {
    "submitted_gcorr": (
        "ab67_final_since_s4_gcorr",
        "ab67_final_since_s3_gcorr",
    ),
    "mixed_submitted": ("ab72_final_ema6_s3_ep5", "ab67_final_since_s4"),
    "six_plus_gcorr": (
        "ab72_final_ema6_s3_ep5",
        "ab67_final_since_s4_gcorr",
    ),
    "both_six": ("ab72_final_ema6_s3_ep5", "ab72_final_ema6_s4_ep5"),
    "both_four_noft": ("ab67_final_since_s4", "ab67_final_since_s3"),
}
LAMBDAS = (0.0, 0.02, 0.05)


def score(sums: np.ndarray, lam: float, row_weights: np.ndarray) -> float:
    """Calculate pooled score after multiplying each sequence's sufficient sums."""
    return decide.wp(sums * row_weights[:, None, None, None], lam)


def class_weights(classes: np.ndarray, desired_mix: np.ndarray) -> np.ndarray:
    """Reweight the observed class counts to a fixed four-class mixture."""
    counts = np.bincount(classes, minlength=4)
    if np.any(counts == 0):
        raise ValueError("Every market class must appear in the selected rows")
    return desired_mix[classes] * len(classes) / counts[classes]


def chain_bootstraps(chains: np.ndarray, count: int) -> list[np.ndarray]:
    """Draw paired sequence indices with their recovered chains kept together."""
    rng = np.random.default_rng(41)
    rows = np.arange(len(chains))
    return [decide._resample(rows, rng, chains) for _ in range(count)]


def main() -> None:
    """Write scores, paired uncertainty, and mix sensitivity for final pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=decide.ROOT / "research" / "artifacts" / "final_pair_audit.json",
    )
    args = parser.parse_args()
    if args.boot < 0:
        parser.error("--boot must be nonnegative")

    with np.load(decide.ROOT / "cache" / "valid_targets.npz") as valid:
        y, mask = valid["y"], valid["mask"]
        classes, row_groups = valid["market_class"], valid["row_group"]
    with np.load(decide.ROOT / "cache" / "train_targets.npz") as train:
        train_classes = train["market_class"]
    _, held_groups = valid_row_groups(decide.ROOT / "datasets" / "valid.parquet")
    idx = np.flatnonzero(np.isin(row_groups, held_groups))
    held_classes = classes[idx]
    held_mix = np.bincount(held_classes, minlength=4) / len(held_classes)
    all_valid_mix = np.bincount(classes, minlength=4) / len(classes)
    train_mix = np.bincount(train_classes, minlength=4) / len(train_classes)
    train_unlikeness = np.load(decide.ROOT / "cache" / "adv_valid_score.npy")
    shift_bin = np.zeros(len(classes), dtype=np.int8)
    for market_class in range(4):
        class_rows = classes == market_class
        cuts = np.quantile(train_unlikeness[class_rows], (1 / 3, 2 / 3))
        shift_bin[class_rows] = np.searchsorted(cuts, train_unlikeness[class_rows], side="right")
    mixes = {
        "observed_holdout": held_mix,
        "valid": all_valid_mix,
        "midpoint": (all_valid_mix + train_mix) / 2,
        "train": train_mix,
    }
    weights = {name: class_weights(held_classes, mix) for name, mix in mixes.items()}
    prediction_names = {name for pair in PAIRS.values() for name in pair}
    predictions = {
        name: np.load(decide.ROOT / "cache" / f"pred_{name}_valid.npy", mmap_mode="r")
        for name in prediction_names
    }
    sums = {
        name: pair_sums(predictions[a], predictions[b], y, mask, idx)
        for name, (a, b) in PAIRS.items()
    }
    reference = sums["submitted_gcorr"]
    chain_ids = decide._chains()[0][idx]
    boots = chain_bootstraps(chain_ids, args.boot)
    result: dict[str, object] = {
        "reference": "submitted_gcorr",
        "reserved_sequences": int(len(idx)),
        "reserved_class_counts": np.bincount(held_classes, minlength=4).tolist(),
        "mixes": {name: mix.tolist() for name, mix in mixes.items()},
        "pairs": {},
    }
    pair_results: dict[str, object] = {}
    for name, candidate in sums.items():
        cells: dict[str, object] = {}
        for mix_name, row_weights in weights.items():
            for lam in LAMBDAS:
                key = f"{mix_name}_lambda_{lam:g}"
                value = score(candidate, lam, row_weights)
                delta = value - score(reference, lam, row_weights)
                boot_delta = [
                    score(candidate[b], lam, class_weights(held_classes[b], mixes[mix_name]))
                    - score(reference[b], lam, class_weights(held_classes[b], mixes[mix_name]))
                    for b in boots
                ]
                cells[key] = {
                    "score": value,
                    "delta": delta,
                    "chain_boot_sd": float(np.std(boot_delta)) if boots else None,
                }
        groups: dict[str, object] = {}
        for group_name, group_ids in (
            ("market_class", held_classes),
            ("train_unlikeness_tercile", shift_bin[idx]),
        ):
            groups[group_name] = [
                {
                    "sequences": int(np.sum(group_ids == group_id)),
                    "delta_lambda_0": decide.wp(candidate[group_ids == group_id])
                    - decide.wp(reference[group_ids == group_id]),
                }
                for group_id in range(4 if group_name == "market_class" else 3)
            ]
        pair_results[name] = {"members": PAIRS[name], "cells": cells, "groups": groups}
    result["pairs"] = pair_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
