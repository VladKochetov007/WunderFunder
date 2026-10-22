"""Grade a root-admitted online relationship panel after validating every artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any  # JSON metadata is validated at runtime.

from online_relationship_grade_contract import (
    ARMS,
    CandidateArtifacts,
    ParentCacheArtifacts,
    _resolve,
    read_json,
    validate_recipe,
)
from online_relationship_grade_core import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    sha256_file,
)
from online_relationship_grade_outputs import verify_candidate
from online_relationship_grade_parent import verify_parent_cache
from online_relationship_grade_scoring import grade_fold
from online_relationship_grade_targets import load_targets

ROOT = Path(__file__).resolve().parents[3]


def _validate_artifacts(
    recipe_path: Path, recipe_sha: str
) -> tuple[
    dict[str, Any],
    dict[int, ParentCacheArtifacts],
    dict[tuple[int, int, str], CandidateArtifacts],
    dict[int, dict[str, Any]],
]:
    recipe, actual_recipe_sha = validate_recipe(recipe_path, ROOT, recipe_sha)
    folds = {int(record["fold"]): record for record in recipe["folds"]}
    parent_caches = {
        fold_id: verify_parent_cache(ROOT, recipe, actual_recipe_sha, fold, recipe_path)
        for fold_id, fold in folds.items()
    }
    subsets = {
        fold_id: read_json(_resolve(ROOT, fold["subset"]["path"]))
        for fold_id, fold in folds.items()
    }
    candidates: dict[tuple[int, int, str], CandidateArtifacts] = {}
    for cell in recipe["cells"]:
        fold_id = int(cell["fold"])
        key = (fold_id, int(cell["lag"]), str(cell["arm"]))
        if key in candidates:
            raise ValueError(f"duplicate candidate cell {key}")
        candidates[key] = verify_candidate(
            ROOT,
            recipe,
            recipe_path,
            actual_recipe_sha,
            folds[fold_id],
            cell,
            subsets[fold_id],
            parent_caches[fold_id],
        )
    if len(parent_caches) != 2 or len(candidates) != 12:
        raise ValueError("all two parent endpoints and twelve candidates are required")
    _validate_panel_consistency(parent_caches, candidates)
    if sha256_file(recipe_path) != actual_recipe_sha:
        raise ValueError("frozen recipe changed while artifacts were being validated")
    return recipe, parent_caches, candidates, folds


def _validate_panel_consistency(
    parent_caches: dict[int, ParentCacheArtifacts],
    candidates: dict[tuple[int, int, str], CandidateArtifacts],
) -> None:
    """Require each matched fold/lag panel to share all source and membership inputs."""
    all_candidate_sources: set[str] = set()
    for fold_id, parent in parent_caches.items():
        fold_rows = [value.identity for key, value in candidates.items() if key[0] == fold_id]
        if len(fold_rows) != 6:
            raise ValueError(f"F{fold_id} does not have all six lag/arm outputs")
        parent_sha = parent.identity["evaluation_prediction_sidecar_sha256"]
        if any(row["parent_cache_manifest_sha256"] != parent_sha for row in fold_rows):
            raise ValueError(f"F{fold_id} candidates do not share the verified parent cache")
        if len({row["fit_membership_sha256"] for row in fold_rows}) != 1:
            raise ValueError(f"F{fold_id} candidates use different fit memberships")
        source_ids = {json.dumps(row["source_sha256"], sort_keys=True) for row in fold_rows}
        if len(source_ids) != 1:
            raise ValueError(f"F{fold_id} candidates use different source-code identities")
        all_candidate_sources.update(source_ids)
        if len({row["evaluation_membership_sha256"] for row in fold_rows}) != 1:
            raise ValueError(f"F{fold_id} candidates use different evaluation order")
        for lag in (1, 16):
            normalizers = {
                candidates[(fold_id, lag, arm)].identity["normalizer_sha256"] for arm in ARMS
            }
            if len(normalizers) != 1:
                raise ValueError(f"F{fold_id} lag{lag} arms use different stream normalizers")
    if len(all_candidate_sources) != 1:
        raise ValueError("candidate source-code identity differs across folds")
    if (
        len(
            {
                json.dumps(value.identity["cache_source_sha256"], sort_keys=True)
                for value in parent_caches.values()
            }
        )
        != 1
    ):
        raise ValueError("parent-cache builder source identity differs across folds")


def _recheck_output_files(
    parents: dict[int, ParentCacheArtifacts],
    candidates: dict[tuple[int, int, str], CandidateArtifacts],
) -> None:
    """Refuse scoring if a verified model output changed after its first hash check."""
    for fold_id, artifact in parents.items():
        for path_key, sha_key in (
            ("evaluation_prediction_path", "evaluation_prediction_sha256"),
            ("evaluation_prediction_sidecar_path", "evaluation_prediction_sidecar_sha256"),
        ):
            if sha256_file(Path(artifact.identity[path_key])) != artifact.identity[sha_key]:
                raise ValueError(f"F{fold_id} parent endpoint changed after validation")
    for artifact in candidates.values():
        identity = artifact.identity
        for path_key, sha_key in (
            ("prediction_path", "prediction_sha256"),
            ("prediction_sidecar_path", "prediction_sidecar_sha256"),
            ("checkpoint_path", "checkpoint_sha256"),
            ("provenance_path", "provenance_sha256"),
        ):
            if sha256_file(Path(identity[path_key])) != identity[sha_key]:
                raise ValueError(f"{identity['run']}: candidate changed after validation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe_path = _resolve(ROOT, args.recipe)
    output_path = _resolve(ROOT, args.output)

    recipe, parents, candidates, folds = _validate_artifacts(recipe_path, args.recipe_sha256)
    scratch = Path(recipe["resource_contract"]["scratch"]).resolve()
    if not output_path.is_relative_to(scratch):
        raise ValueError("grade report must be written to physical scratch")
    _recheck_output_files(parents, candidates)
    _, target_folds = load_targets(recipe, folds)

    fold_reports = []
    contrast_reports = []
    for fold_id in sorted(folds):
        endpoint_report, contrasts = grade_fold(
            fold_id, parents[fold_id], candidates, *target_folds[fold_id]
        )
        fold_reports.append(endpoint_report)
        contrast_reports.extend(contrasts)
    if len(contrast_reports) != 26:
        raise AssertionError("panel grading did not produce exactly 26 fold-local contrasts")
    _recheck_output_files(parents, candidates)

    grader_sources = (
        Path(__file__),
        Path(__file__).with_name("online_relationship_grade_contract.py"),
        Path(__file__).with_name("online_relationship_grade_parent.py"),
        Path(__file__).with_name("online_relationship_grade_cache.py"),
        Path(__file__).with_name("online_relationship_grade_outputs.py"),
        Path(__file__).with_name("online_relationship_grade_provenance.py"),
        Path(__file__).with_name("online_relationship_grade_fit_contract.py"),
        Path(__file__).with_name("online_relationship_grade_core.py"),
        Path(__file__).with_name("online_relationship_grade_scoring.py"),
        Path(__file__).with_name("online_relationship_grade_targets.py"),
        ROOT / "scripts/grade_prediction_bank.py",
        ROOT / "src/wnn/evaluation.py",
        ROOT / "src/wnn/gmetric.py",
        ROOT / "src/wnn/splits.py",
        ROOT / "src/wnn/training/membership.py",
    )
    report = {
        "schema": "wnn.online-relationships-grade.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "recipe": {"path": str(recipe_path), "sha256": args.recipe_sha256},
        "grader_source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in grader_sources
        },
        "metric": "global_wp_clip2_v1",
        "targets": {
            "path": recipe["target_cache"]["path"],
            "sha256": recipe["target_cache"]["sha256"],
        },
        "endpoint_count": 14,
        "contrast_count": len(contrast_reports),
        "folds": fold_reports,
        "contrasts": contrast_reports,
        "interpretation": {
            "fold_seed_confounding": "F0 and seed1001 versus F1 and seed1002 are separate cells; no pooled seed effect is estimated.",
            "validation_exposure": "Both full validation folds were previously inspected and remain exposed development evidence.",
            "fit_parent_scope": "Parent predictions on selected fitting groups are in-sample and are never graded as endpoints.",
            "interval_scope": "Overall official-WP intervals resample dependency components within each fold, conditional on these fitted endpoints.",
            "multiplicity": "The 26 fold-local contrast intervals are reported without multiplicity adjustment; they are not a confirmatory familywise test.",
            "fixed_points": "Both target correlations and four fixed position-bin scores are point deltas for these fixed models.",
            "lag_scope": "Lag contrasts are fixed-panel point comparisons with conditional component intervals; they do not select a universally best lag.",
            "expected_public_score": None,
        },
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"grade": str(output_path), "endpoints": 14, "contrasts": 26}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
