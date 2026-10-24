"""Preflight or grade the saved online-relationship F0 exploratory panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from f0_grade_preflight import prepare_f0_grade, write_json_exclusive  # noqa: E402
from f0_grade_scoring import score_prepared_f0  # noqa: E402
from online_relationship_grade_contract import _resolve, read_json  # noqa: E402
from online_relationship_grade_core import sha256_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_admission(
    admission_path: Path, preflight_path: Path, prepared_report: dict[str, Any], recipe_sha: str
) -> None:
    admission = read_json(admission_path)
    if admission.get("schema") != "wnn.online-relationships-f0-grade-admission.v1":
        raise ValueError("root admission receipt has an unsupported schema")
    expected = {
        "status": "admitted",
        "recipe_sha256": recipe_sha,
        "source_manifest_sha256": prepared_report["source_manifest_sha256"],
        "preflight_receipt_sha256": sha256_file(preflight_path),
        "metric": "global_wp_clip2_v1",
        "target_quality_reads_authorized": True,
    }
    if any(admission.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "root admission does not bind this recipe, preflight, metric, and target read"
        )
    resources = admission.get("resources", {})
    memory_limit = resources.get("memory_max_bytes", 0)
    if not isinstance(memory_limit, int) or not 0 < memory_limit <= 400_000_000:
        raise ValueError("grade worker memory limit exceeds the 400 MB adapter bound")
    cpu_limit = resources.get("cpu_cores", 0)
    if not isinstance(cpu_limit, (int, float)) or not 0 < cpu_limit <= 1:
        raise ValueError("grade worker CPU allowance must be at most one aggregate core")
    if resources.get("swap_bytes") != 0:
        raise ValueError("grade admission must disable swap")
    if resources.get("scratch") != "/home/vlad/development/wunder/research_tmp":
        raise ValueError("grade admission must use the approved physical scratch")
    if json.loads(preflight_path.read_text(encoding="utf-8")) != prepared_report:
        raise ValueError("preflight receipt differs from the current label-free artifact check")


def main() -> int:
    """Run a label-free artifact preflight or an admitted fixed-endpoint grade."""
    args = _parse_args()
    recipe_path = _resolve(ROOT, args.recipe)
    prepared = prepare_f0_grade(recipe_path, args.recipe_sha256)
    if args.preflight_only:
        if args.admission is not None or args.preflight_receipt is not None:
            raise ValueError("preflight-only mode does not accept a grade admission")
        write_json_exclusive(_resolve(ROOT, args.output), prepared.report)
        print(json.dumps({"status": prepared.report["status"], "output": str(args.output)}))
        return 0

    if args.admission is None or args.preflight_receipt is None:
        raise ValueError("grade mode requires root admission and the matching preflight receipt")
    preflight_path = _resolve(ROOT, args.preflight_receipt)
    admission_path = _resolve(ROOT, args.admission)
    _require_admission(admission_path, preflight_path, prepared.report, args.recipe_sha256)
    report = score_prepared_f0(prepared)
    report["admission"] = {
        "path": str(admission_path),
        "sha256": sha256_file(admission_path),
        "preflight_receipt_sha256": sha256_file(preflight_path),
        "root_authorized": True,
    }
    write_json_exclusive(_resolve(ROOT, args.output), report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
