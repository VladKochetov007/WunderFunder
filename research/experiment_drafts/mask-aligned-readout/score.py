"""Held-out fold scoring and paired component-bootstrap report for PR36."""

from __future__ import annotations

# Any is used only for schema-checked JSON receipts and dynamic metric records.
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PACKET = Path(__file__).resolve().parent
REPO_ROOT = PACKET.parents[2]
sys.path.insert(0, str(PACKET))
sys.path.insert(0, str(REPO_ROOT / "src"))
from receiver_assets import (  # noqa: E402
    file_sha256,
    load_json,
    output_root,
    recipe_cells,
    run_preflight,
)
from score_audit import check_audit_receipts  # noqa: E402
from score_helpers import (  # noqa: E402
    blend_array,
    component_moments,
    fold_seed_streams,
    score_array,
    selected_targets,
    validated_prediction,
)


def grade(assets_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Score both held-out folds and bootstrap the equal-cell treatment effect."""
    assets = run_preflight(assets_path)
    recipe = load_json(PACKET / "recipe.json")
    cells = recipe_cells(recipe)
    audit_bindings = {
        cell.fold: check_audit_receipts(assets, cell, ("all_valid", "is_scored_weighted"))
        for cell in cells
    }
    target_cache_identity: dict[str, Any] | None = None
    uncertainty = recipe["metrics"]["uncertainty"]
    master_seed = int(uncertainty["seed"])
    seeds = fold_seed_streams(master_seed, len(cells))
    fold_results: dict[str, Any] = {}
    fold_draws: list[np.ndarray] = []
    point_deltas: list[float] = []
    n_boot = int(uncertainty["replicates"])
    from wnn.evaluation import paired_component_bootstrap

    for cell, fold_seed in zip(cells, seeds, strict=True):
        split = assets.splits[cell.fold]
        targets, selection, components, cache_identity = selected_targets(assets.root, split)
        shared_cache_identity = {
            key: cache_identity[key]
            for key in (
                "path",
                "sha256",
                "valid_parquet_source_identity",
                "row_groups_sha256",
                "row_group_count",
            )
        }
        if target_cache_identity is None:
            target_cache_identity = shared_cache_identity
        elif target_cache_identity != shared_cache_identity:
            raise ValueError("fold scoring did not use one immutable valid target cache")
        cell_recipe = next(item for item in recipe["cells"] if item["fold"] == cell.fold)
        split_sha = str(cell_recipe["split_manifest_sha256"])
        parent, parent_meta = validated_prediction(
            assets, cell, "parent", cell.parent_run, split_sha
        )
        control, control_meta = validated_prediction(
            assets, cell, "control", cell.run_name("all_valid"), split_sha
        )
        treatment, treatment_meta = validated_prediction(
            assets, cell, "treatment", cell.run_name("is_scored_weighted"), split_sha
        )
        parent_score = score_array(targets, parent, selection)
        control_score = score_array(targets, control, selection)
        treatment_score = score_array(targets, treatment, selection)
        treatment_moments, control_moments = component_moments(
            targets, treatment, control, selection
        )
        bootstrap = paired_component_bootstrap(
            treatment_moments,
            control_moments,
            components,
            n_boot=n_boot,
            seed=fold_seed,
            centered=True,
            return_draws=True,
        )
        fold_draws.append(np.asarray(bootstrap.pop("bootstrap_deltas"), dtype=np.float64))
        point_deltas.append(float(bootstrap["mean_diff"]))
        control_blend = score_array(targets, blend_array(parent, control), selection)
        treatment_blend = score_array(targets, blend_array(parent, treatment), selection)
        fold_results[cell.fold] = {
            "seed": cell.seed,
            "fold_bootstrap_seed": fold_seed,
            "n_evaluation_groups": len(split.evaluation_groups),
            "n_components": int(len(np.unique(components))),
            "n_selected_rows": int(selection.sum()),
            "parent": parent_score,
            "control": control_score,
            "treatment": treatment_score,
            "treatment_minus_control_wp": float(treatment_score["wp"] - control_score["wp"]),
            "paired_component_bootstrap": bootstrap,
            "secondary_blends": {
                "control": control_blend,
                "control_delta_vs_parent": float(control_blend["wp"] - parent_score["wp"]),
                "treatment": treatment_blend,
                "treatment_delta_vs_parent": float(treatment_blend["wp"] - parent_score["wp"]),
                "treatment_minus_control_wp": float(treatment_blend["wp"] - control_blend["wp"]),
                "parent_weight": 0.75,
                "child_weight": 0.25,
            },
            "predictions": {
                "parent": parent_meta,
                "control": control_meta,
                "treatment": treatment_meta,
            },
            "source_binding": {
                "recipe_sha256": file_sha256(PACKET / "recipe.json"),
                "split_manifest_sha256": file_sha256(assets.split_paths[cell.fold]),
                "target_cache": cache_identity,
                "audit_receipts": audit_bindings[cell.fold],
            },
        }
    equal_cell_draws = np.mean(np.stack(fold_draws), axis=0)
    score_sources = sorted((REPO_ROOT / "src").rglob("*.py"))
    score_sources.extend(sorted(PACKET.glob("*.py")))
    score_sources.append(REPO_ROOT / "scripts" / "predict_run.py")
    score_source_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): file_sha256(path) for path in score_sources
    }
    score_source_identity = hashlib.sha256(
        json.dumps(score_source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": 1,
        "experiment_id": recipe["id"],
        "metric_id": "global_wp_clip2_v1",
        "recipe_sha256": file_sha256(PACKET / "recipe.json"),
        "scoring_implementation": {
            "sha256": score_source_identity,
            "files": score_source_hashes,
        },
        "target_cache": target_cache_identity,
        "selection": "need_prediction AND is_scored",
        "clip": [-2, 2],
        "uncertainty": {
            "method": "paired_component_bootstrap_conditional_on_fixed_predictions",
            "replicates_per_fold": n_boot,
            "master_seed": master_seed,
            "fold_streams": {cell.fold: seed for cell, seed in zip(cells, seeds, strict=True)},
            "equal_cell_mean_delta": float(np.mean(point_deltas)),
            "equal_cell_mean_delta_ci95": np.percentile(equal_cell_draws, [2.5, 97.5]).tolist(),
            "folds_weighted_equally": True,
        },
        "folds": fold_results,
        "interpretation": "Conditional uncertainty over fixed fold predictions; two folds confound parent, fold, and seed.",
    }
    destination = output_path or (output_root(assets.root) / "grade.json")
    if destination.exists():
        raise FileExistsError(f"grade output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return result


def parse_args() -> argparse.Namespace:
    """Parse receiver paths without loading model code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=os.environ.get("WNN_MASK_READOUT_ASSETS"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Write and print the immutable held-out score report."""
    args = parse_args()
    if args.assets is None:
        raise SystemExit("pass --assets or set WNN_MASK_READOUT_ASSETS")
    result = grade(args.assets, args.output)
    print(json.dumps(result["uncertainty"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
