"""Compare all six fixed-final endpoints on the prepared B300 receiver only."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from packet_acceptance import require_acceptance
from packet_contract import CELLS, REPO, binding, file_hash, read, require, write_new
from packet_lineage import validate_cache, validate_evaluation, validate_terminal

# Score receipts are heterogeneous JSON; boundary validation below precedes use.
USEFUL_WP = 0.001


def read_score(root: Path, cell: str) -> dict[str, Any]:
    """Verify immutable lineage and return one canonical full-valid score receipt."""
    directory = root / cell
    actual = validate_evaluation(root, cell)
    require(read(directory / "evaluation_receipt.json") == actual, "Evaluation receipt changed")
    score = read(directory / "score.json")
    metrics = score["score"]
    require(metrics.get("metric_id") == "global_wp_clip2_v1", "Official metric required")
    for name in ("wp", "t0", "t1"):
        value = metrics.get(name)
        require(type(value) in (float, int) and math.isfinite(value), "Invalid score")
        require(-1 <= value <= 1, "Correlation outside [-1,1]")
    require(abs(metrics["wp"] - (metrics["t0"] + metrics["t1"]) / 2) < 1e-12, "WP differs")
    require(type(metrics.get("selected_rows")) is int and metrics["selected_rows"] > 0, "No rows")
    require(bool(score.get("component_centered_moments")), "Missing component statistics")
    if cell.startswith("kd_"):
        teacher = validate_cache(root, cell)
        identity = validate_terminal(root, cell)["identity"]
        require(identity["data"].get("teacher") == teacher, "Student teacher lineage changed")
    return score


def paired_contrast(
    candidate: dict[str, Any], reference: dict[str, Any], bootstrap_seed: int
) -> dict[str, Any]:
    """Resample the same graph components for two already fitted endpoints."""
    import numpy as np

    sys.path.insert(0, str(REPO / "src"))
    from wnn.evaluation import paired_component_bootstrap

    a, b = candidate["component_centered_moments"], reference["component_centered_moments"]
    require(set(a) == set(b), "Component coverage differs")
    keys = sorted(a, key=int)
    candidate_moments = np.asarray([a[key] for key in keys], dtype=np.float64)
    reference_moments = np.asarray([b[key] for key in keys], dtype=np.float64)
    require(candidate_moments.shape == (len(keys), 2, 6), "Malformed component moments")
    require(reference_moments.shape == candidate_moments.shape, "Moment shape differs")
    require(
        np.isfinite(candidate_moments).all() and np.isfinite(reference_moments).all(), "NaN moments"
    )
    # Weight, target mean and target centered variance must identify the same labels.
    require(
        np.allclose(
            candidate_moments[..., [0, 1, 3]],
            reference_moments[..., [0, 1, 3]],
            rtol=1e-10,
            atol=1e-10,
        ),
        "Target statistics differ between paired endpoints",
    )
    interval = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        np.asarray([int(key) for key in keys]),
        n_boot=2000,
        seed=bootstrap_seed,
        centered=True,
    )
    for side, score in (("candidate_wp", candidate), ("reference_wp", reference)):
        require(abs(interval[side] - score["score"]["wp"]) < 1e-9, "Moments do not reproduce score")
    return {
        "delta": {
            name: candidate["score"][name] - reference["score"][name] for name in ("wp", "t0", "t1")
        },
        "conditional_component_bootstrap": interval,
        "bootstrap_input_grain": "component; library n_sequences counts aggregated component records",
        "interval_scope": "Fixed predictions, graph-component resampling only; not training variance, pristine holdout or independent seed/fold evidence.",
    }


def compare(root: Path, data_root: Path, commit: str, output: Path) -> None:
    """Require all six cells, preserve every result and emit a bounded follow-up signal."""
    packet = binding(root, data_root, commit)
    require_acceptance(root)
    scores = {cell: read_score(root, cell) for cell in CELLS}
    row_counts = {score["score"]["selected_rows"] for score in scores.values()}
    require(len(row_counts) == 1, "Selected-row coverage differs")
    component_sets = [set(score["component_centered_moments"]) for score in scores.values()]
    require(all(keys == component_sets[0] for keys in component_sets), "Panel components differ")
    pairs = {}
    for seed in (1001, 1002):
        labels = scores[f"labels_s{seed}"]
        pairs[str(seed)] = {
            "teacher_minus_labels": paired_contrast(scores[f"teacher_s{seed}"], labels, seed),
            "kd_minus_labels": paired_contrast(scores[f"kd_s{seed}"], labels, seed),
        }
    deltas = [pairs[str(seed)]["kd_minus_labels"]["delta"]["wp"] for seed in (1001, 1002)]
    mean_delta = sum(deltas) / 2
    if mean_delta >= USEFUL_WP and min(deltas) >= 0:
        decision = "FOLLOW_UP_SIGNAL"
    elif max(deltas) < 0:
        decision = "NEGATIVE_FOR_THIS_RECIPE"
    else:
        decision = "UNCERTAIN"
    write_new(
        output,
        {
            "schema": "wnn.source-kd-comparison.v1",
            "git_commit": commit,
            "split_sha256": packet["split_sha256"],
            "scientific_fits": 6,
            "independent_splits": 1,
            "seeds": [1001, 1002],
            "fixed_endpoint": "FINAL5",
            "selected_rows_per_endpoint": next(iter(row_counts)),
            "components": len(component_sets[0]),
            "endpoints": {cell: score["score"] for cell, score in scores.items()},
            "score_sha256": {cell: file_hash(root / cell / "score.json") for cell in CELLS},
            "paired_contrasts": pairs,
            "mean_by_role": {
                role: {
                    name: sum(scores[f"{role}_s{seed}"]["score"][name] for seed in (1001, 1002)) / 2
                    for name in ("wp", "t0", "t1")
                }
                for role in ("teacher", "labels", "kd")
            },
            "mean_kd_minus_labels_wp": mean_delta,
            "useful_effect_wp": USEFUL_WP,
            "decision": decision,
            "interpretation": "Development evidence on exposed full valid. Two seeds are not significance. Decisions allocate follow-up only; no automatic fit, submission or direction closure.",
        },
    )


def main() -> None:
    """Expose only an explicit receiver-side comparison action, never fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(
        args.root.resolve(), args.data_root.resolve(), args.expected_commit, args.output.resolve()
    )


if __name__ == "__main__":
    main()
