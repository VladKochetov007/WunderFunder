"""Grade the fixed future-input dynamics panel on full F0/F2 evaluation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from dynamics_panel_artifacts import (
    ROOT,
    ROWS,
    SEEDS,
    SPLITS,
    SUBSETS,
    _verify_run,
)
from grade_prediction_bank import score, sha256

from wnn.evaluation import paired_component_bootstrap, paired_component_moments
from wnn.splits import file_sha256

ARMS = ("control", "aux005")
SCORED_FROM_ROW = 99
POSITION_BINS = ((99, 500), (500, 2_000), (2_000, 10_000), (10_000, 20_000))
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 271_828


def _expected_membership(fold: int, split: dict[str, Any]) -> tuple[list[int], np.ndarray]:
    """Return exact evaluation order and source-component IDs from the frozen split."""
    groups = [int(group) for group in split["evaluation"]["valid"]]
    component_of = {
        int(row["row_group"]): int(row["component"]) for row in split["sequences"]["valid"]
    }
    if len(groups) != len(set(groups)):
        raise ValueError(f"F{fold} evaluation groups contain duplicates")
    components = np.asarray([component_of[group] for group in groups], dtype=np.int64)
    if (len(groups), len(np.unique(components))) != {0: (468, 141), 2: (443, 144)}[fold]:
        raise ValueError(f"F{fold} evaluation membership differs from the source split")
    return groups, components


def _mixture(control: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Apply the fixed 90/10 control/candidate mixture in float64 then store float32."""
    return (
        0.9 * np.asarray(control, dtype=np.float64) + 0.1 * np.asarray(candidate, dtype=np.float64)
    ).astype(np.float32)


def _delta_report(
    targets: np.ndarray,
    candidate: np.ndarray,
    control: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
) -> dict[str, Any]:
    """Report all-target, per-target, position-bin deltas and paired component uncertainty."""
    candidate_score = score(targets, mask, candidate)
    control_score = score(targets, mask, control)
    paired_a, paired_b = paired_component_moments(targets, candidate, control, mask)
    bootstrap = paired_component_bootstrap(
        paired_a,
        paired_b,
        components,
        n_boot=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    deltas = {
        metric: candidate_score[metric] - control_score[metric] for metric in ("wp", "t0", "t1")
    }
    target_bootstraps = {}
    for index, name in enumerate(("t0", "t1")):
        y = np.repeat(targets[..., index : index + 1], 2, axis=-1)
        a = np.repeat(candidate[..., index : index + 1], 2, axis=-1)
        b = np.repeat(control[..., index : index + 1], 2, axis=-1)
        moments_a, moments_b = paired_component_moments(y, a, b, mask)
        target_bootstraps[name] = paired_component_bootstrap(
            moments_a,
            moments_b,
            components,
            n_boot=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
    bins = {}
    for start, stop in POSITION_BINS:
        selected = mask.copy()
        selected[:, :start] = False
        selected[:, stop:] = False
        name = f"rows_{start}_{stop}"
        a, b = score(targets, selected, candidate), score(targets, selected, control)
        bins[name] = {metric: a[metric] - b[metric] for metric in ("wp", "t0", "t1")}
    return {
        "candidate_scores": candidate_score,
        "control_scores": control_score,
        "deltas": deltas,
        "paired_component_bootstrap": bootstrap,
        "per_target_component_bootstrap": target_bootstraps,
        "position_bin_deltas": bins,
    }


def _grade_fold(fold: int, run_names: dict[str, str]) -> dict[str, Any]:
    """Score the two fit arms and fixed mixture on the exact full evaluation fold."""
    split_path, subset_path = SPLITS[fold], SUBSETS[fold]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    split_sha = file_sha256(split_path)
    if subset.get("split_manifest_sha256") != split_sha:
        raise ValueError(f"F{fold}: fit subset and evaluation split SHA differ")
    groups, components = _expected_membership(fold, split)
    predictions, identities = {}, {}
    for arm in ARMS:
        predictions[arm], identities[arm] = _verify_run(
            fold, arm, run_names[arm], groups, split_sha, subset
        )
    for identity in (
        "initial_parameter_sha256",
        "normalization_identity",
        "target_statistics_sha256",
        "fit_members_sha256",
    ):
        if identities["control"][identity] != identities["aux005"][identity]:
            raise ValueError(f"F{fold}: paired arms differ in {identity}")

    target_path = ROOT / "cache/valid_targets.npz"
    with np.load(target_path, allow_pickle=False) as cache:
        positions = {int(group): index for index, group in enumerate(cache["row_group"])}
        if len(positions) != len(cache["row_group"]) or not set(groups).issubset(positions):
            raise ValueError(f"F{fold}: target cache has duplicate or missing evaluation groups")
        index = [positions[group] for group in groups]
        targets = np.asarray(cache["y"][index], dtype=np.float32)
        mask = np.asarray(cache["mask"][index], dtype=bool)
    mask &= np.arange(ROWS)[None, :] >= SCORED_FROM_ROW
    if targets.shape != (len(groups), ROWS, 2) or mask.shape != (len(groups), ROWS):
        raise ValueError(f"F{fold}: official target/mask cache shape differs")
    for arm in ARMS:
        for start in range(0, len(groups), 8):
            selection = slice(start, start + 8)
            if not np.isfinite(predictions[arm][selection][mask[selection]]).all():
                raise ValueError(f"F{fold} {arm}: nonfinite prediction on a scored row")
    predictions["mix_90control_10aux005"] = _mixture(predictions["control"], predictions["aux005"])
    scores = {name: score(targets, mask, value) for name, value in predictions.items()}
    pairs = {
        "aux005_minus_control": (predictions["aux005"], predictions["control"]),
        "mix_90control_10aux005_minus_control": (
            predictions["mix_90control_10aux005"],
            predictions["control"],
        ),
    }
    comparisons = {
        name: _delta_report(targets, candidate, control, mask, components)
        for name, (candidate, control) in pairs.items()
    }
    return {
        "fold": fold,
        "seed": SEEDS[fold],
        "split_manifest_sha256": split_sha,
        "fit_subset_sha256": file_sha256(subset_path),
        "evaluation_groups": len(groups),
        "evaluation_components": int(len(np.unique(components))),
        "scored_rows": int(mask.sum()),
        "runs": run_names,
        "prediction_and_model_identity": identities,
        "scores": scores,
        "paired_comparisons": comparisons,
        "exposure": "historically inspected development fold; component intervals are conditional on the paired fitted cell, not seed/fold/public uncertainty",
    }


def main() -> int:
    """Write the frozen two-cell grade without fitting a mixture coefficient."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    names = {
        0: {
            "control": "dynamics_f0_s1001_control_r1",
            "aux005": "dynamics_f0_s1001_aux005_r1",
        },
        2: {
            "control": "dynamics_f2_s1002_control_r1",
            "aux005": "dynamics_f2_s1002_aux005_r1",
        },
    }
    report = {
        "schema": "wnn.future-input-dynamics-panel-grade.v1",
        "experiment_id": "E-FUTURE-INPUT-DYNAMICS-20261021-R1",
        "metric_id": "global_wp_clip2_v1",
        "fit_counts": {"fits_per_arm": 2, "total_new_fits": 4, "teacher_fits": 0},
        "mixture": {"control": 0.9, "aux005": 0.1},
        "bootstrap": {
            "unit": "source-qualified connected evaluation component",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "source_sha256": {
            "grader": sha256(Path(__file__)),
            "scorer": sha256(ROOT / "scripts/grade_prediction_bank.py"),
            "metric": sha256(ROOT / "src/wnn/gmetric.py"),
            "evaluation": sha256(ROOT / "src/wnn/evaluation.py"),
            "dynamics_auxiliary": sha256(ROOT / "src/wnn/training/dynamics_auxiliary.py"),
            "group_registry": sha256(ROOT / "src/wnn/kalman_group_features.py"),
        },
        "folds": {str(fold): _grade_fold(fold, names[fold]) for fold in (0, 2)},
        "claim_limit": "two exposed fold/seed cells estimate direction only and cannot separate fold from seed variation or establish significance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {
                fold: {name: values["scores"][name]["wp"] for name in values["scores"]}
                for fold, values in report["folds"].items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
