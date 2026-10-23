"""Independently grade the fixed TL3 clean-fold label/KD screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any  # Any covers schema-checked artifacts and canonical runtime adapters.

import numpy as np
from common import (
    PACKET,
    ROOT,
    digest,
    load_assets,
    output_root,
    preflight,
    read_json,
    write_new,
)
from grade_artifacts import verify_teacher_caches
from grade_audit import build_run_plan, verify_run
from grade_math import (
    component_bootstrap,
    equal_cell_target_means,
    fit_pair_gate,
    score_cell,
)


def grade(assets_path: Path, destination: Path | None = None) -> dict[str, Any]:
    """Verify every required run and score fixed standalone and mixture endpoints."""
    assets_path = assets_path.expanduser().resolve()
    sys.path.insert(0, str(PACKET))
    sys.path.insert(0, str(ROOT / "src"))
    assets = load_assets(assets_path)
    assets_sha = digest(assets_path)
    if assets_sha != assets.get("_sha256"):
        raise ValueError("assets JSON changed while being loaded")
    pins = preflight(assets)
    recipe_path = PACKET / "recipe.json"
    recipe = read_json(recipe_path)
    plan, mixture_weights = build_run_plan(recipe)
    project_root = Path(str(assets["project_root"])).expanduser().resolve()
    split_records = assets.get("splits")
    if not isinstance(split_records, dict):
        raise ValueError("validated assets must provide F0/F2 split paths and hashes")

    source_sha = pins.get("source_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("preflight did not bind a valid source identity")
    recipe_sha = digest(recipe_path)
    if pins.get("recipe_sha256") != recipe_sha:
        raise ValueError("preflight recipe identity differs from recipe.json")
    evidence_root = output_root(assets)
    target = destination.expanduser().resolve() if destination else evidence_root / "grade.json"
    if target.exists():
        raise FileExistsError(f"refusing to replace existing grade: {target}")

    import pyarrow.parquet as pq
    from audit import require_fit
    from audit import teacher_targets as audited_teacher_targets
    from wnn.data import WARMUP
    from wnn.evaluation import (
        _merge_moments,
        _score_component_moments,
        paired_component_moments,
    )
    from wnn.gmetric import METRIC_ID, GlobalAccumulator
    from wnn.splits import load_campaign_split

    if METRIC_ID != "global_wp_clip2_v1":
        raise ValueError(f"canonical metric changed unexpectedly: {METRIC_ID}")
    parquet_path = project_root / "datasets" / "valid.parquet"
    parquet = pq.ParquetFile(parquet_path)
    records: dict[str, dict[str, Any]] = {}
    splits: dict[str, Any] = {}
    split_paths: dict[str, Path] = {}
    teacher_cache_bindings: dict[str, list[dict[str, Any]]] = {}
    for cell_id, fold in (("F0", 0), ("F2", 2)):
        item = split_records.get(cell_id)
        if not isinstance(item, dict):
            raise ValueError(f"validated split {cell_id} is missing")
        split_path = Path(str(item["path"])).expanduser().resolve()
        split_sha = str(item["sha256"])
        if digest(split_path) != split_sha:
            raise ValueError(f"{cell_id} split manifest hash changed")
        split = load_campaign_split(split_path, project_root)
        if split.fold != fold or split.sha256 != split_sha:
            raise ValueError(f"{cell_id} split loader returned a different fold or identity")
        splits[cell_id], split_paths[cell_id] = split, split_path
        cell_records = [record for record in plan if record["cell"] == cell_id]
        for record in (item for item in cell_records if item["kind"] == "teacher"):
            verified = verify_run(
                record,
                project_root,
                evidence_root,
                split,
                split_sha,
                recipe_sha,
                source_sha,
                require_fit(assets, cell_id, record["role"]),
                digest,
                read_json,
            )
            records[verified["run"]] = verified
        teacher_targets, cache_files = verify_teacher_caches(
            cell_id,
            project_root,
            split,
            audited_teacher_targets(assets, cell_id),
            digest,
            read_json,
        )
        teacher_cache_bindings[cell_id] = cache_files
        for record in (item for item in cell_records if item["kind"] == "student"):
            verified = verify_run(
                record,
                project_root,
                evidence_root,
                split,
                split_sha,
                recipe_sha,
                source_sha,
                require_fit(assets, cell_id, record["role"]),
                digest,
                read_json,
                teacher_targets,
                cache_files,
            )
            records[verified["run"]] = verified
        fit_pair_gate(records, cell_id)

    if len({record["runtime_sha256"] for record in records.values()}) != 1:
        raise ValueError("screen runs do not share one verified runtime identity")
    if len({record["source_sha256"] for record in records.values()}) != 1:
        raise ValueError("screen runs do not share one verified source identity")

    fold_results: dict[str, Any] = {}
    moment_rows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    primary_deltas: list[float] = []
    for cell_id in ("F0", "F2"):
        result, candidate, reference, components = score_cell(
            cell_id,
            splits[cell_id],
            records,
            mixture_weights,
            parquet,
            int(WARMUP),
            GlobalAccumulator,
            paired_component_moments,
        )
        fold_results[cell_id] = result
        moment_rows[cell_id] = (candidate, reference, components)
        primary_deltas.append(result["full_mixture_kd_minus_labels"]["wp"])

    metrics = recipe.get("metrics", {})
    replicates = int(metrics.get("bootstrap_replicates", 0))
    master_seed = int(metrics.get("bootstrap_seed", 0))
    if replicates != 10_000 or master_seed != 231023:
        raise ValueError("recipe bootstrap must remain 10,000 replicates with seed 231023")
    streams = np.random.SeedSequence(master_seed).spawn(2)
    fold_draws: dict[str, np.ndarray] = {}
    fold_bootstrap: dict[str, Any] = {}
    for cell_id, stream in zip(("F0", "F2"), streams, strict=True):
        seed = int(stream.generate_state(1, dtype=np.uint32)[0])
        candidate, reference, components = moment_rows[cell_id]
        draws, summary = component_bootstrap(
            candidate,
            reference,
            components,
            seed,
            replicates,
            _merge_moments,
            _score_component_moments,
        )
        fold_draws[cell_id] = draws
        summary["point_delta"] = fold_results[cell_id]["full_mixture_kd_minus_labels"]["wp"]
        fold_bootstrap[cell_id] = summary
    equal_cell_draws = np.mean(np.stack([fold_draws["F0"], fold_draws["F2"]]), axis=0)
    mean_delta = float(np.mean(primary_deltas))
    ci95 = np.percentile(equal_cell_draws, [2.5, 97.5]).tolist()
    useful = float(metrics.get("useful_delta", 0.0))
    if useful != 0.0003:
        raise ValueError("recipe useful-delta threshold must stay fixed at 0.0003")
    both_nonnegative = all(value >= 0.0 for value in primary_deltas)
    threshold_passed = mean_delta >= useful and both_nonnegative
    target_means = equal_cell_target_means(fold_results, ("F0", "F2"))
    if threshold_passed:
        interpretation = (
            "screen_threshold_met; separate confirmation is warranted; interval remains conditional"
        )
    elif ci95[0] <= 0.0 <= ci95[1] or not both_nonnegative:
        interpretation = (
            "uncertain_screen; mixed signs or a zero-crossing interval do not reject the direction"
        )
    else:
        interpretation = "useful_delta_threshold_not_met; retain as exploratory evidence without automatic rejection"

    result = {
        "schema_version": 1,
        "experiment_id": recipe.get("id", "TL3-CLEAN-FOLD-DISTILLATION-R1"),
        "status": "complete_two_cell_screen",
        "metric_id": METRIC_ID,
        "selection": "need_prediction AND is_scored",
        "clip": [-2, 2],
        "mixture_weights": mixture_weights,
        "coefficient_tuning": False,
        "primary": {
            "endpoint": "equal_cell_mean(full_mixture_kd_wp - full_mixture_labels_wp)",
            "cell_deltas_wp": {
                cell_id: fold_results[cell_id]["full_mixture_kd_minus_labels"]["wp"]
                for cell_id in ("F0", "F2")
            },
            "cell_deltas_t0": {
                cell_id: fold_results[cell_id]["full_mixture_kd_minus_labels"]["t0"]
                for cell_id in ("F0", "F2")
            },
            "cell_deltas_t1": {
                cell_id: fold_results[cell_id]["full_mixture_kd_minus_labels"]["t1"]
                for cell_id in ("F0", "F2")
            },
            "equal_cell_mean_delta_t0": target_means["t0"],
            "equal_cell_mean_delta_t1": target_means["t1"],
            "equal_cell_mean_delta_wp": mean_delta,
            "useful_delta": useful,
            "both_cells_nonnegative": both_nonnegative,
            "threshold_passed": threshold_passed,
            "interpretation": interpretation,
        },
        "uncertainty": {
            "method": "paired_whole_component_bootstrap_conditional_on_fixed_predictions",
            "replicates_per_cell": replicates,
            "master_seed": master_seed,
            "fold_streams": {cell_id: fold_bootstrap[cell_id]["seed"] for cell_id in ("F0", "F2")},
            "equal_cell_mean_delta_ci95": ci95,
            "folds_weighted_equally": True,
            "folds": fold_bootstrap,
            "training_seed_uncertainty_estimated": False,
        },
        "folds": fold_results,
        "inputs": {
            "assets_path": str(assets_path.resolve()),
            "assets_sha256": assets_sha,
            "assets_identity": pins.get("assets_sha256"),
            "source_sha256": source_sha,
            "recipe_path": str(recipe_path),
            "recipe_sha256": recipe_sha,
            "valid_parquet": str(parquet_path),
            "valid_parquet_source_identity": {
                cell_id: read_json(split_paths[cell_id])
                .get("sources", {})
                .get("datasets/valid.parquet")
                for cell_id in ("F0", "F2")
            },
            "splits": {
                cell_id: {
                    "path": str(split_paths[cell_id]),
                    "sha256": splits[cell_id].sha256,
                }
                for cell_id in ("F0", "F2")
            },
            "teacher_cache_bindings": teacher_cache_bindings,
            "grader_files": {
                name: {"path": str(PACKET / name), "sha256": digest(PACKET / name)}
                for name in ("grade.py", "grade_audit.py", "grade_artifacts.py", "grade_math.py")
            },
            "runs": records,
        },
        "interpretation_limits": [
            "All campaign folds have prior exposure; this is not a fresh-holdout estimate.",
            "The bootstrap conditions on the fixed fits and resamples whole source components; it does not estimate training-seed variation.",
            "A complete F0/F2 panel is required; no partial-cell promotion or coefficient search is permitted.",
        ],
    }
    if preflight(assets) != pins or digest(assets_path) != assets_sha:
        raise ValueError("source, data bindings or assets changed during grading")
    write_new(target, result)
    return result


def main() -> int:
    """Run the read-only grader and print its primary decision summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=os.environ.get("TL3_CLEAN_KD_ASSETS"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.assets is None:
        parser.error("pass --assets or set TL3_CLEAN_KD_ASSETS")
    result = grade(Path(args.assets), args.output)
    print(json.dumps(result["primary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
