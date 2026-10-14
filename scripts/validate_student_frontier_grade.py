"""Validate one fixed-epoch F0 score report against its frozen grade inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MAX_JSON_BYTES = 2_000_000


def sha256_file(path: Path) -> str:
    """Hash a source file in bounded streaming blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    """Read a bounded JSON file after checking its actual size."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise ValueError(f"{path} exceeds the {MAX_JSON_BYTES}-byte read bound")
    with path.open("rb") as source:
        raw = source.read(MAX_JSON_BYTES + 1)
    if len(raw) != size:
        raise ValueError(f"{path} changed while being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_grade(
    manifest_path: Path,
    expected_manifest_sha256: str,
    report_path: Path,
    run: str,
    expected_reference: str | None,
    receipt_path: Path,
) -> dict[str, object]:
    """Check grader code pins and the score report's exact fixed comparison."""
    manifest_size = manifest_path.stat().st_size
    if manifest_size <= 0 or manifest_size > MAX_JSON_BYTES:
        raise ValueError("grader-input manifest exceeds its read/hash bound")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("grader-input manifest differs from the sequence pin")
    manifest = read_json(manifest_path)
    grader = manifest["grader"]
    if not isinstance(grader, dict):
        raise ValueError("grader-input manifest lacks its grader section")
    for field in ("score_fold0_panel", "evaluate_bandit_continuation", "metric"):
        entry = grader[field]
        if not isinstance(entry, dict):
            raise ValueError(f"grader-input manifest has an invalid {field} pin")
        path = Path(str(entry["path"]))
        if path.stat().st_size <= 0:
            raise ValueError(f"pinned grader source is empty: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"pinned grader source changed: {path}")
    target = grader["targets"]
    if not isinstance(target, dict):
        raise ValueError("grader-input manifest has an invalid target pin")
    target_path = Path(str(target["path"]))
    if target_path.stat().st_size != target["bytes"]:
        raise ValueError("frozen validation-target byte size changed")

    report = read_json(report_path)
    expected_candidate = f"{run}@ep0"
    if report.get("candidate") != expected_candidate:
        raise ValueError("grade report candidate does not match the fixed epoch-0 run")
    if report.get("split_manifest_sha256") != manifest["split"]["sha256"]:
        raise ValueError("grade report split hash differs from the frozen fold-0 manifest")
    if report.get("valid_targets_sha256") != target["sha256"]:
        raise ValueError("grade report validation-target hash differs from its frozen pin")
    if report.get("evaluation_groups") != 468:
        raise ValueError("grade report does not cover all 468 fold-0 evaluation groups")
    exposure = report.get("exposure")
    if not isinstance(exposure, str) or "not a public forecast" not in exposure:
        raise ValueError("grade report lacks the required exposed-development evidence label")

    official = report.get("official")
    if not isinstance(official, dict) or official.get("metric_id") != "global_wp_clip2_v1":
        raise ValueError("grade report does not contain the frozen official metric")
    score = official.get("wp")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("grade report official score is missing or nonfinite")

    contrasts = report.get("contrasts")
    if not isinstance(contrasts, list):
        raise ValueError("grade report lacks its contrast list")
    if expected_reference is None:
        if contrasts:
            raise ValueError("the first H256x2 control must not be graded against an unregistered reference")
    elif (
        len(contrasts) != 1
        or not isinstance(contrasts[0], dict)
        or contrasts[0].get("reference") != expected_reference
    ):
        raise ValueError("grade report does not contain exactly its matched H256x2 reference")

    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite grade receipt: {receipt_path}")
    receipt = {
        "format": "wnn.student-frontier-fold0-grade-validation",
        "version": 1,
        "run": run,
        "candidate": expected_candidate,
        "official_metric": official["metric_id"],
        "official_wp": float(score),
        "split_manifest_sha256": manifest["split"]["sha256"],
        "valid_targets_sha256": target["sha256"],
        "evaluation_groups": 468,
        "reference": expected_reference,
        "grade_report_sha256": sha256_file(report_path),
        "validation": "passed; exposed F0 development panel, not a public forecast",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    """Validate one root-scored report and print a concise terminal receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate_grade(
        args.manifest.resolve(strict=True),
        args.manifest_sha256,
        args.report.resolve(strict=True),
        args.run,
        args.reference,
        args.out.resolve(),
    )
    print(json.dumps({"run": receipt["run"], "wp": receipt["official_wp"], "validation": receipt["validation"]}))


if __name__ == "__main__":
    main()
