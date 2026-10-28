"""Compare all six fixed-final endpoints on the prepared B300 receiver only."""

from __future__ import annotations

import argparse
from pathlib import Path

from packet_acceptance import require_acceptance
from packet_comparison import paired_contrast, read_score
from packet_contract import CELLS, binding, file_hash, require, write_new
from packet_study import DEFAULT_STUDY, STUDY

# Score receipts are heterogeneous JSON; boundary validation below precedes use.
USEFUL_WP = 0.001


def compare(root: Path, data_root: Path, commit: str, output: Path) -> None:
    """Require all six cells, preserve every result and emit a bounded follow-up signal."""
    require(STUDY.slug == DEFAULT_STUDY, "Use the selected study's own comparison command")
    require(
        tuple(f"{role}_s{seed}" for seed in (1001, 1002) for role in ("teacher", "labels", "kd"))
        == CELLS,
        "Use the matching study comparison command",
    )
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
