"""Grade one fixed uniform shrink of saved F0 residual corrections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any  # Frozen recipe and admission receipts use dynamic JSON fields.

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
SCOUT_DIR = HERE.parent
ONLINE_DIR = ROOT / "research/experiment_drafts/online-relationships-20261022"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ONLINE_DIR))
sys.path.insert(0, str(SCOUT_DIR))
sys.path.insert(0, str(HERE))

from correction_shrink_scoring import (  # noqa: E402
    CELL_NAMES,
    PREDICTION_RUNS,
    build_result,
    compare_raw_scores,
    score_endpoints,
    validate_alpha,
    validate_contrasts,
)
from f0_grade_preflight import PreparedF0Grade, prepare_f0_grade, write_json_exclusive  # noqa: E402
from online_relationship_grade_core import sha256_file  # noqa: E402

REQUIRED_SOURCE_FILES = {
    "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/grade_correction_shrink_r1.py",
    "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/correction_shrink_scoring.py",
    "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/check_correction_shrink_synthetic.py",
    "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f0_grade_preflight.py",
    "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f0_grade_scoring.py",
    "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f0_grade_ancestry.py",
    "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/SOURCE_MANIFEST_R2.json",
    "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/F0_EXPLORATORY_RECIPE_R2.json",
    "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/F0_EXPLORATORY_GRADE_R2.json",
    "research/experiment_drafts/online-relationships-20261022/online_relationship_grade_core.py",
    "scripts/limited",
    "pyproject.toml",
    "uv.lock",
    "src/wnn/evaluation.py",
    "src/wnn/gmetric.py",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read one metadata/admission object without touching array payloads."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def resolve_repo_path(value: str) -> Path:
    """Resolve a path while keeping repository inputs and outputs contained."""
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"recipe path escapes the repository: {value}")
    return resolved


def verify_source_manifest(recipe: dict[str, Any]) -> None:
    """Check this adapter and its direct code dependencies after admission."""
    record = recipe["source_manifest"]
    path = resolve_repo_path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError("correction-shrink source-manifest SHA differs from recipe")
    manifest = read_json(path)
    if manifest.get("schema") != "wnn.correction-shrink-source-manifest.v2":
        raise ValueError("correction-shrink source manifest schema is unsupported")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("correction-shrink source manifest has no file pins")
    if not REQUIRED_SOURCE_FILES.issubset(files):
        raise ValueError("correction-shrink source manifest omits a direct code dependency")
    evidence = manifest.get("evidence")
    synthetic_record = evidence.get("synthetic_validation") if isinstance(evidence, dict) else None
    if synthetic_record != recipe.get("synthetic_validation"):
        raise ValueError("synthetic validation receipt differs from the frozen recipe")
    for record in evidence.values():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("correction-shrink source manifest contains malformed evidence")
        if sha256_file(resolve_repo_path(record["path"])) != record.get("sha256"):
            raise ValueError(f"correction-shrink evidence digest differs: {record['path']}")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("correction-shrink source manifest contains malformed pins")
        if sha256_file(resolve_repo_path(relative)) != expected:
            raise ValueError(f"frozen correction-shrink source changed: {relative}")


def validate_admission(
    admission: dict[str, Any],
    recipe: dict[str, Any],
    recipe_sha: str,
    preflight_sha: str,
    baseline_sha: str,
) -> dict[str, Any]:
    """Require root authorization and the prospective resource envelope."""
    expected = {
        "status": "admitted",
        "recipe_sha256": recipe_sha,
        "source_manifest_sha256": recipe["source_manifest"]["sha256"],
        "base_recipe_sha256": recipe["base_recipe"]["sha256"],
        "base_source_manifest_sha256": recipe["base_source_manifest"]["sha256"],
        "baseline_grade_sha256": baseline_sha,
        "preflight_receipt_sha256": preflight_sha,
        "metric": "global_wp_clip2_v1",
        "alpha": 0.1,
        "new_fit_count": 0,
        "target_quality_reads_authorized": True,
        "output_path": recipe["output"]["path"],
    }
    if admission.get("schema") != "wnn.correction-shrink-f0-grade-admission.v1":
        raise ValueError("root admission schema is unsupported")
    if any(admission.get(key) != value for key, value in expected.items()):
        raise ValueError("root admission does not bind the frozen shrink-grade inputs")
    resources = admission.get("resources", {})
    memory = resources.get("memory_max_bytes")
    cpu = resources.get("cpu_cores")
    wall = resources.get("max_wall_seconds")
    if not isinstance(memory, int) or not 0 < memory <= 400_000_000:
        raise ValueError("root admission exceeds the 400 MB worker cap")
    if not isinstance(cpu, (int, float)) or not 0 < cpu <= 1:
        raise ValueError("root admission exceeds one CPU core")
    if not isinstance(wall, int) or not 0 < wall <= 300:
        raise ValueError("root admission exceeds the five-minute wall cap")
    if resources.get("swap_bytes") != 0:
        raise ValueError("root admission must disable swap")
    if resources.get("scratch") != "/home/vlad/development/wunder/research_tmp":
        raise ValueError("root admission must use approved physical scratch")
    return resources


def verify_label_free_inputs(
    prepared: PreparedF0Grade,
    preflight: dict[str, Any],
    baseline: dict[str, Any],
    recipe: dict[str, Any],
) -> None:
    """Match current endpoint metadata to the frozen preflight and baseline."""
    if prepared.report != preflight:
        raise ValueError("current label-free preflight differs from the admitted receipt")
    if baseline.get("schema") != "wnn.online-relationships-f0-grade.v1":
        raise ValueError("baseline scalar grade schema is unsupported")
    if baseline.get("status") != "graded_exploratory_f0_only":
        raise ValueError("baseline scalar grade is not complete")
    if baseline.get("preflight") != preflight:
        raise ValueError("baseline grade and admitted label-free receipt differ")
    if (
        preflight.get("recipe_sha256") != recipe["base_recipe"]["sha256"]
        or preflight.get("source_manifest_sha256") != recipe["base_source_manifest"]["sha256"]
    ):
        raise ValueError("preflight source identity differs from the frozen base recipes")
    target_pin = recipe["target_cache"]
    target_receipt = preflight["target_cache"]
    if (
        target_receipt.get("path") != target_pin["path"]
        or target_receipt.get("expected_sha256") != target_pin["sha256"]
        or target_receipt.get("file_bytes") != target_pin["file_bytes"]
    ):
        raise ValueError("label-free target metadata differs from the frozen target pin")
    artifact_by_run = {artifact.get("run"): artifact for artifact in preflight.get("artifacts", [])}
    parent_artifacts = [
        artifact
        for artifact in preflight.get("artifacts", [])
        if artifact.get("endpoint") == "parent"
    ]
    if len(parent_artifacts) != 1:
        raise ValueError("preflight must contain exactly one parent prediction artifact")
    parent_artifact = parent_artifacts[0]
    parent_pin = recipe["parent_prediction"]
    for recipe_key, report_key in (
        ("path", "path"),
        ("sha256", "sha256"),
        ("file_bytes", "file_bytes"),
        ("shape", "shape"),
        ("dtype", "dtype"),
        ("evaluation_membership_sha256", "evaluation_membership_sha256"),
    ):
        if parent_pin[recipe_key] != parent_artifact.get(report_key):
            raise ValueError(f"parent prediction {recipe_key} changed")
    candidates = recipe["candidate_predictions"]
    if set(candidates) != set(CELL_NAMES) or len(prepared.candidate_prediction_paths) != 6:
        raise ValueError("six artifact runs must map to the five frozen unique cells")
    for cell, pin in candidates.items():
        current = artifact_by_run.get(pin["run"])
        if current is None:
            raise ValueError(f"candidate prediction identity changed: {cell}")
        if pin["run"] != PREDICTION_RUNS[cell]:
            raise ValueError(f"candidate run mapping changed: {cell}")
        for recipe_key, report_key in (
            ("prediction_path", "prediction_path"),
            ("prediction_sha256", "prediction_sha256"),
            ("sidecar_sha256", "prediction_sidecar_sha256"),
            ("provenance_sha256", "provenance_sha256"),
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("prediction_bytes", "file_bytes"),
            ("shape", "shape"),
            ("dtype", "dtype"),
            ("evaluation_membership_sha256", "evaluation_membership_sha256"),
        ):
            if pin[recipe_key] != current.get(report_key):
                raise ValueError(f"candidate {recipe_key} changed: {cell}")
    a1 = artifact_by_run[PREDICTION_RUNS["A_lag1"]]
    a16_run = recipe["omitted_duplicate"]["run"]
    a16 = artifact_by_run[a16_run]
    duplicate_pin = recipe["omitted_duplicate"]
    if a16_run != "online_rel_f0_s1001_lag16_a_r1":
        raise ValueError("omitted A lag16 candidate run identity changed")
    for recipe_key, report_key in (
        ("prediction_path", "prediction_path"),
        ("prediction_sha256", "prediction_sha256"),
        ("sidecar_sha256", "prediction_sidecar_sha256"),
        ("provenance_sha256", "provenance_sha256"),
        ("checkpoint_sha256", "checkpoint_sha256"),
        ("prediction_bytes", "file_bytes"),
        ("shape", "shape"),
        ("dtype", "dtype"),
        ("evaluation_membership_sha256", "evaluation_membership_sha256"),
    ):
        if duplicate_pin[recipe_key] != a16.get(report_key):
            raise ValueError(f"omitted duplicate {recipe_key} changed")
    if a1["prediction_sha256"] != a16["prediction_sha256"]:
        raise ValueError("A lag16 is no longer an exact duplicate of A lag1")
    scope = recipe["scope"]
    if (
        len(prepared.evaluation_groups) != scope["evaluation_groups"]
        or prepared.report.get("evaluation_membership_sha256")
        != scope["evaluation_membership_sha256"]
        or len({int(value) for value in prepared.components}) != scope["evaluation_components"]
    ):
        raise ValueError("source-qualified F0 evaluation membership differs from recipe")
    validate_alpha(recipe["shrink"]["alpha"])
    validate_contrasts(recipe)


def parse_args() -> argparse.Namespace:
    """Parse exact recipe, root admission, preflight and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--baseline-grade", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Verify root admission before hashing candidate arrays or reading labels."""
    args = parse_args()
    recipe_path = resolve_repo_path(str(args.recipe))
    if sha256_file(recipe_path) != args.recipe_sha256:
        raise ValueError("shrink recipe SHA differs from the command pin")
    recipe = read_json(recipe_path)
    if (
        recipe.get("revision") != 2
        or recipe.get("status") != "prepared_for_root_admission"
        or recipe.get("new_fit_count") != 0
    ):
        raise ValueError("recipe is not the frozen no-fit prepared proposal")
    admission_path = resolve_repo_path(str(args.admission))
    if str(admission_path) != recipe["admission"]["path"]:
        raise ValueError("admission path differs from the frozen recipe")
    preflight_path = args.preflight_receipt.resolve()
    if str(preflight_path) != recipe["preflight_receipt"]["path"]:
        raise ValueError("preflight path differs from the frozen receipt location")
    baseline_path = resolve_repo_path(str(args.baseline_grade))
    if str(baseline_path) != recipe["baseline_grade"]["path"]:
        raise ValueError("baseline grade path differs from the frozen location")
    preflight = read_json(preflight_path)
    baseline = read_json(baseline_path)
    output_path = resolve_repo_path(str(args.output))
    if str(output_path) != recipe["output"]["path"]:
        raise ValueError("output path differs from the frozen recipe")
    if output_path.exists():
        raise FileExistsError(f"create-only output already exists: {output_path}")
    preflight_sha = sha256_file(preflight_path)
    baseline_sha = sha256_file(baseline_path)
    if preflight_sha != recipe["preflight_receipt"]["sha256"]:
        raise ValueError("label-free preflight SHA differs from recipe")
    if baseline_sha != recipe["baseline_grade"]["sha256"]:
        raise ValueError("baseline grade SHA differs from recipe")
    target_cache = preflight.get("target_cache", {})
    if (
        preflight.get("status") != "passed_label_free_preflight"
        or target_cache.get("labels_read") is not False
        or target_cache.get("content_hash_checked") is not False
    ):
        raise ValueError("admitted preflight is not label-free")
    admission = read_json(admission_path)
    resources = validate_admission(
        admission, recipe, args.recipe_sha256, preflight_sha, baseline_sha
    )
    verify_source_manifest(recipe)
    base_recipe_path = resolve_repo_path(recipe["base_recipe"]["path"])
    if sha256_file(base_recipe_path) != recipe["base_recipe"]["sha256"]:
        raise ValueError("original F0 recipe SHA differs from shrink recipe")
    base_recipe = read_json(base_recipe_path)
    if base_recipe.get("source_manifest") != recipe["base_source_manifest"]:
        raise ValueError("base F0 recipe source-manifest identity differs from shrink recipe")
    prepared = prepare_f0_grade(base_recipe_path, recipe["base_recipe"]["sha256"])
    verify_label_free_inputs(prepared, preflight, baseline, recipe)
    scores = score_endpoints(prepared)
    compare_raw_scores(scores["endpoint_scores"], baseline)
    result = build_result(
        recipe, args.recipe_sha256, prepared, scores, admission, admission_path, preflight_path
    )
    result["resource_contract"] = resources
    write_json_exclusive(output_path, result)
    print(json.dumps({"status": result["status"], "output": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
