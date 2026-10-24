"""Preflight or grade the frozen F1 mean-only correction shrink diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from f1_grade_preflight_r2 import (  # noqa: E402
    PreparedF1Grade,
    _read_json,
    _resolve,
    prepare_f1_grade,
)
from f1_grade_scoring import score_f1_endpoints  # noqa: E402
from online_relationship_grade_core import sha256_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse the pinned recipe and explicit root admission paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--admission", type=Path)
    return parser.parse_args()


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Create a JSON artifact without replacing an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _require_admission(
    prepared: PreparedF1Grade, admission_path: Path, preflight_path: Path
) -> dict[str, Any]:
    """Require root to bind this exact recipe, source set, preflight, and metric."""
    recipe = prepared.recipe
    admission = _read_json(admission_path)
    expected = {
        "schema": "wnn.correction-shrink-f1-grade-admission.v1",
        "status": "admitted",
        "recipe_sha256": prepared.recipe_sha256,
        "source_manifest_sha256": recipe["source_manifest"]["sha256"],
        "preflight_receipt_sha256": sha256_file(preflight_path),
        "metric": recipe["metric"]["id"],
        "target_quality_reads_authorized": True,
        "output_path": str(_resolve(recipe["output_path"])),
    }
    if any(admission.get(key) != value for key, value in expected.items()):
        raise ValueError("root admission does not bind this F1 recipe, preflight, and metric")
    resources = admission.get("resources", {})
    memory = resources.get("memory_max_bytes")
    cpu = resources.get("cpu_cores")
    runtime = resources.get("max_runtime_seconds")
    if not isinstance(memory, int) or not 0 < memory <= 400_000_000:
        raise ValueError("F1 grade memory limit exceeds 400,000,000 bytes")
    if not isinstance(cpu, (int, float)) or not 0 < cpu <= 1:
        raise ValueError("F1 grade must use at most one aggregate CPU core")
    if not isinstance(runtime, int) or not 0 < runtime <= 300:
        raise ValueError("F1 grade runtime must be bounded to at most 300 seconds")
    if resources.get("swap_bytes") != 0:
        raise ValueError("F1 grade admission must disable swap")
    if resources.get("scratch") != recipe["resource_contract"]["scratch"]:
        raise ValueError("F1 grade admission must use physical-disk scratch")
    if _read_json(preflight_path) != prepared.report:
        raise ValueError("preflight receipt differs from the current source check")
    return admission


def _result(
    prepared: PreparedF1Grade,
    scores: dict[str, Any],
    admission: dict[str, Any],
    admission_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    """Combine score output with the frozen question, exposure, and admission."""
    recipe = prepared.recipe
    return {
        "schema": "wnn.correction-shrink-f1-grade.v1",
        "status": "graded_exploratory_f1_followup",
        "recipe_sha256": prepared.recipe_sha256,
        "source_manifest_sha256": recipe["source_manifest"]["sha256"],
        "base_recipe_sha256": recipe["base_recipe"]["sha256"],
        "preflight_receipt_sha256": sha256_file(preflight_path),
        "admission": {
            "path": str(admission_path),
            "sha256": sha256_file(admission_path),
            "root_authorized": admission["target_quality_reads_authorized"],
        },
        "study": recipe["study"],
        "metric": recipe["metric"],
        "evaluation": prepared.report["evaluation"],
        "fit_membership": prepared.report["fit_membership"],
        "ancestry": prepared.report["ancestry"],
        "endpoints": scores["endpoint_scores"],
        "contrasts": scores["contrasts"],
        "target_cache_identity": scores["target_cache_identity"],
        "prediction_sha256_after_score": scores["prediction_sha256_after_score"],
        "exposure": recipe["exposure"],
        "interpretation_limits": recipe["interpretation_limits"],
        "new_fit_count": 0,
    }


def main() -> int:
    """Run no-label preflight or score only after exact root admission."""
    args = _parse_args()
    recipe_path = _resolve(str(args.recipe))
    recipe = _read_json(recipe_path)
    expected_output = _resolve(
        recipe["preflight_path"] if args.preflight_only else recipe["output_path"]
    )
    output_path = _resolve(str(args.output))
    if output_path != expected_output:
        raise ValueError("requested output path differs from the frozen recipe")
    if output_path.exists():
        raise FileExistsError(f"create-only output already exists: {output_path}")

    prepared = prepare_f1_grade(recipe_path, args.recipe_sha256)
    if args.preflight_only:
        if args.admission is not None or args.preflight_receipt is not None:
            raise ValueError("preflight-only mode cannot accept admission or receipt inputs")
        _write_exclusive(output_path, prepared.report)
        print(json.dumps({"status": prepared.report["status"], "output": str(output_path)}))
        return 0

    if args.admission is None or args.preflight_receipt is None:
        raise ValueError("grade mode requires root admission and matching preflight receipt")
    admission_path = _resolve(str(args.admission))
    preflight_path = _resolve(str(args.preflight_receipt))
    if admission_path != _resolve(recipe["admission_path"]):
        raise ValueError("admission path differs from the frozen recipe")
    if preflight_path != _resolve(recipe["preflight_path"]):
        raise ValueError("preflight receipt path differs from the frozen recipe")
    admission = _require_admission(prepared, admission_path, preflight_path)
    scores = score_f1_endpoints(prepared)
    _write_exclusive(
        output_path, _result(prepared, scores, admission, admission_path, preflight_path)
    )
    print(json.dumps({"status": "graded_exploratory_f1_followup", "output": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
