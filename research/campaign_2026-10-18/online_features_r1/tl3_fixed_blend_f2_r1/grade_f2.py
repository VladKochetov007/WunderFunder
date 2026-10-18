"""Grade frozen TL3/R3 endpoints on the previously inspected F2 fold."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
from f2_grade_preflight import (
    ARTIFACTS_DEFAULT,
    BOOT_SEED,
    CANDIDATES,
    N_BOOT,
    PACKET,
    ROOT,
    SCORED_FROM_ROW,
    TARGET_REL,
    load_campaign_scorer,
    sha256_file,
)
from f2_grade_preflight import (
    grade_preflight as preflight,
)


def grade(
    root: Path, artifacts: Path, recipe_path: Path, preflight_path: Path
) -> dict[str, object]:
    """Score frozen endpoints and paired component bootstrap intervals."""
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments
    from wnn.gmetric import GlobalAccumulator

    frozen = cast(dict[str, object], json.loads(preflight_path.read_text(encoding="utf-8")))
    if frozen.get("recipe_sha256") != sha256_file(recipe_path):
        raise ValueError("preflight does not bind the frozen recipe")
    if frozen.get("grader_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("grader source changed after preflight")
    observed = preflight(root, artifacts, recipe_path)
    if observed.get("input_fingerprint") != frozen.get("input_fingerprint"):
        raise ValueError("candidate, split, prediction, or source bytes changed after preflight")

    mmap_helper = load_campaign_scorer(root)
    target_arrays = cast(dict[str, np.memmap], mmap_helper.mmap_stored_npz(root / TARGET_REL))
    target_groups = target_arrays["row_group"]
    target_position = {int(group): index for index, group in enumerate(target_groups)}
    identity = cast(dict[str, object], frozen["identity"])
    groups = cast(list[int], identity["evaluation_group_ids"])
    components = np.asarray(identity["component_ids"], dtype=np.int64)
    if len(target_position) != len(target_groups) or any(
        group not in target_position for group in groups
    ):
        raise ValueError("target cache does not uniquely cover all F2 evaluation groups")
    target_indices = [target_position[group] for group in groups]

    predictions = {
        name: np.load(
            artifacts / str(specification["prediction"]), mmap_mode="r", allow_pickle=False
        )
        for name, specification in CANDIDATES.items()
    }
    names = ("tl3", "r3", "blend_75_25", "blend_50_50")
    accumulators = {name: GlobalAccumulator() for name in names}
    comparisons = {
        "r3_vs_tl3": ("r3", "tl3"),
        "blend_75_25_vs_tl3": ("blend_75_25", "tl3"),
        "blend_50_50_vs_tl3": ("blend_50_50", "tl3"),
    }
    candidate_stats = {
        name: np.zeros((len(groups), 2, 6), dtype=np.float64) for name in comparisons
    }
    reference_stats = {
        name: np.zeros((len(groups), 2, 6), dtype=np.float64) for name in comparisons
    }
    selected_rows = 0
    for index, target_index in enumerate(target_indices):
        target = np.asarray(target_arrays["y"][target_index], dtype=np.float32)
        mask = np.asarray(target_arrays["mask"][target_index], dtype=bool).copy()
        mask[:SCORED_FROM_ROW] = False
        selected_rows += int(mask.sum())
        tl3 = np.asarray(predictions["tl3"][index], dtype=np.float32)
        r3 = np.asarray(predictions["r3"][index], dtype=np.float32)
        endpoint_predictions = {
            "tl3": tl3,
            "r3": r3,
            "blend_75_25": (0.75 * tl3 + 0.25 * r3).astype(np.float32),
            "blend_50_50": (0.5 * tl3 + 0.5 * r3).astype(np.float32),
        }
        for name, prediction in endpoint_predictions.items():
            accumulators[name].add(target[None], prediction[None], mask[None])
        for name, (candidate_name, reference_name) in comparisons.items():
            candidate_moments, reference_moments = paired_component_moments(
                target[None],
                endpoint_predictions[candidate_name][None],
                endpoint_predictions[reference_name][None],
                mask[None],
            )
            candidate_stats[name][index] = candidate_moments[0]
            reference_stats[name][index] = reference_moments[0]
        del target, mask, tl3, r3, endpoint_predictions

    scores = {name: accumulator.result() for name, accumulator in accumulators.items()}
    intervals = {
        name: paired_component_bootstrap(
            candidate_stats[name],
            reference_stats[name],
            components,
            n_boot=N_BOOT,
            seed=BOOT_SEED,
            centered=True,
        )
        for name in comparisons
    }
    if selected_rows <= 0:
        raise ValueError("official F2 score mask selected no rows")
    score_values = {name: float(value["wp"]) for name, value in scores.items()}
    return {
        "schema": "wnn.tl3-fixed-blend-f2-grade.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "recipe_sha256": sha256_file(recipe_path),
        "preflight_sha256": sha256_file(preflight_path),
        "input_fingerprint": frozen["input_fingerprint"],
        "metric_id": "global_wp_clip2_v1",
        "clip": [-2.0, 2.0],
        "mask": "official cached valid target mask AND sequence row >=99",
        "selected_rows": selected_rows,
        "evaluation_groups": len(groups),
        "evaluation_components": len(np.unique(components)),
        "scores": scores,
        "paired_component_bootstrap": {
            "scope": "conditional on fixed package predictions and previously inspected F2 components; not seed, ancestry, or transfer uncertainty",
            "replicates": N_BOOT,
            "seed": BOOT_SEED,
            "comparisons": intervals,
        },
        "fixed_blends": {
            "blend_75_25": {"tl3": 0.75, "r3": 0.25},
            "blend_50_50": {"tl3": 0.5, "r3": 0.5},
        },
        "descriptive_ranking": sorted(score_values, key=lambda name: -score_values[name]),
        "ancestry": {
            "tl3": "unknown; package metadata calls its parents all-data",
            "r3": "inference-only derivative of R2, adaptively selected on exposed F0",
        },
        "exposure": "F2 was inspected before; TL3 ancestry is unknown and may include F2",
        "claim_limit": "cross-fold robustness diagnostic only; no clean-CV, causal-complementarity, seed, or public-transfer claim",
    }


def main() -> None:
    """Run metadata/prediction preflight or the separate label-reading grade."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "score"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DEFAULT)
    parser.add_argument("--recipe", type=Path, default=PACKET / "recipe.json")
    parser.add_argument("--preflight", type=Path, default=PACKET / "preflight.json")
    parser.add_argument("--output", type=Path, default=PACKET / "result.json")
    args = parser.parse_args()
    root, artifacts, recipe_path = (
        args.root.resolve(strict=True),
        args.artifacts.resolve(strict=True),
        args.recipe.resolve(strict=True),
    )
    if args.mode == "preflight":
        if args.preflight.exists():
            raise FileExistsError(args.preflight)
        report = preflight(root, artifacts, recipe_path)
        args.preflight.parent.mkdir(parents=True, exist_ok=True)
        with args.preflight.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "fingerprint": report["input_fingerprint"]}))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    result = grade(root, artifacts, recipe_path, args.preflight.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "scored", "scores": result["scores"]}, sort_keys=True))


if __name__ == "__main__":
    main()
