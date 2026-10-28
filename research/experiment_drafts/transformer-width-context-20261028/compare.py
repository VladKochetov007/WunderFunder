"""Compare six fixed-final Transformer endpoints without fitting or selecting epochs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SLUG = "transformer-width-context-20261028"
ROLES = ("baseline", "wide", "context")
SEEDS = (1001, 1002)


def compare(root: Path, data_root: Path, commit: str, output: Path) -> None:
    """Bind the teacher-only study and report both paired architecture contrasts."""
    shared = Path(__file__).resolve().parent.parent / "source-holdout-transformer-kd-20261028"
    os.environ["WNN_SEQUENCE_STUDY"] = SLUG
    sys.path.insert(0, str(shared))
    from packet_acceptance import require_acceptance
    from packet_comparison import paired_contrast, read_score
    from packet_contract import CELLS, binding, file_hash, read, require, write_new

    expected = tuple(f"{role}_s{seed}" for seed in SEEDS for role in ROLES)
    require(expected == CELLS, "Wrong study loaded; use a fresh comparison process")
    packet = binding(root, data_root, commit)
    require_acceptance(root)
    scores = {cell: read_score(root, cell) for cell in CELLS}
    counts = {value["score"]["selected_rows"] for value in scores.values()}
    require(len(counts) == 1, "All endpoints must cover the same scored rows")
    components = [set(value["component_centered_moments"]) for value in scores.values()]
    require(all(keys == components[0] for keys in components), "Component coverage differs")
    contrasts = {
        role: {
            str(seed): paired_contrast(scores[f"{role}_s{seed}"], scores[f"baseline_s{seed}"], seed)
            for seed in SEEDS
        }
        for role in ("wide", "context")
    }
    decisions = {}
    for role, cells in contrasts.items():
        deltas = [cells[str(seed)]["delta"]["wp"] for seed in SEEDS]
        mean = sum(deltas) / len(deltas)
        if mean >= 0.001 and min(deltas) >= 0:
            decision = "FOLLOW_UP_SIGNAL"
        elif max(deltas) < 0:
            decision = "NEGATIVE_AT_FIXED_FIVE_EPOCHS"
        else:
            decision = "UNCERTAIN"
        decisions[role] = {"mean_delta_wp": mean, "decision": decision}
    resources = {}
    for cell in CELLS:
        profile = read(root / cell / "profile/profile_receipt.json")
        approved = read(root / cell / "approved_profile.json")
        require(
            approved["admission"]["raw_profile_sha256"]
            == file_hash(root / cell / "profile/profile_receipt.json"),
            "Resource profile changed after approval",
        )
        require(profile["identity"] == approved["identity"], "Resource profile identity changed")
        resources[cell] = {
            "model_parameters": profile["model_parameters"],
            "memory_by_rank": profile["memory_by_rank"],
            "profile_elapsed_seconds": profile["elapsed_seconds"],
            "profile_updates": profile["updates"],
            "final_checkpoint_bytes": (root / cell / "fit/model.pt").stat().st_size,
            "limits": "Disposable profile includes startup; not measured full-fit time or CPU serving cost.",
        }
    write_new(
        output,
        {
            "schema": "wnn.transformer-width-context-comparison.v1",
            "study": SLUG,
            "git_commit": commit,
            "split_sha256": packet["split_sha256"],
            "scientific_fits": 6,
            "student_fits": 0,
            "reused_controls": 0,
            "independent_splits": 1,
            "seeds": list(SEEDS),
            "fixed_endpoint": "FINAL5",
            "selected_rows_per_endpoint": next(iter(counts)),
            "components": len(components[0]),
            "endpoints": {cell: score["score"] for cell, score in scores.items()},
            "score_sha256": {cell: file_hash(root / cell / "score.json") for cell in CELLS},
            "paired_contrasts": contrasts,
            "mean_by_role": {
                role: {
                    name: sum(scores[f"{role}_s{seed}"]["score"][name] for seed in SEEDS)
                    / len(SEEDS)
                    for name in ("wp", "t0", "t1")
                }
                for role in ROLES
            },
            "useful_effect_wp": 0.001,
            "decisions": decisions,
            "resources": resources,
            "interpretation": "Two exposed-data architecture contrasts, one split, two seeds. Conditional intervals do not include training uncertainty or multiple-search selection. Follow-up signals are not significance, public forecasts, optimal horizons or automatic launches.",
        },
    )


def main() -> None:
    """Expose an explicit receiver-only comparison after all six endpoints finish."""
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
