#!/usr/bin/env python3
"""Finish report packaging from the already-persisted, frozen R1 grade.

The scoring process completed and wrote its JSON/moments before a renderer
KeyError. This successor validates those immutable outputs and writes only the
missing Markdown plus a terminal/finalization receipt; it does not rescore.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SPEC_SHA256 = "f6de6f79fad019300ce95a5bcd96fadc1fdeaceae19ae272be76f6d88258428b"
GRADE_JSON = HERE / "BASELINE_PAIR_R1.json"
MOMENTS = HERE / "BASELINE_PAIR_R1.moments.npz"
GRADE_SOURCE = HERE / "grade_fold1_baseline_pair_r1.py"
MARKDOWN = HERE / "BASELINE_PAIR_R1.md"
FAILURE_RECEIPT = HERE / "BASELINE_PAIR_R1_TERMINAL_FAILURE_R1.json"
FINALIZATION_RECEIPT = HERE / "BASELINE_PAIR_R1_FINALIZATION_R2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def render(grade: dict[str, Any], grade_sha: str) -> str:
    endpoints = grade["endpoints"]
    contrast = grade["primary_contrast"]
    low, high = contrast["ci95"]
    lines = [
        "# Fold-1 seed3/seed4 baseline-pair grade R1",
        "",
        f"Frozen spec SHA-256: `{grade['frozen_spec']['sha256']}`.",
        f"Metric: `{grade['primary_contrast']['metric_id']}`; fold 1 has "
        f"{grade['split']['evaluation_groups']} exposed groups and "
        f"{grade['split']['evaluation_components']} connected components "
        f"({grade['split']['selected_scored_rows']:,} selected rows).",
        "",
        "| Endpoint | WP | t0 | t1 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("seed3", "Seed 3"),
        ("seed4", "Seed 4"),
        ("equal_raw_prediction_mean_fp64_cast_fp32_then_metric_clip", "Raw equal mean (1:1)"),
    ):
        score = endpoints[key]
        lines.append(f"| {label} | {score['wp']:.9f} | {score['t0']:.9f} | {score['t1']:.9f} |")
    lines.extend(
        [
            "",
            "Primary contrast: equal mean minus seed 3.",
            f"Paired whole-component bootstrap: {contrast['mean_diff']:+.9f} WP; "
            f"95% percentile CI [{low:+.9f}, {high:+.9f}] across "
            f"{contrast['n_components']} components ({contrast['n_boot']} replicates, "
            f"seed {contrast['seed']}).",
            "",
            "This is a baseline diagnostic on a historically inspected fold, not a fresh holdout, "
            "new-best claim, or public-score estimate. Component resampling does not include "
            "training-seed variation or public transfer.",
            "",
            f"Machine-readable result: `{GRADE_JSON.relative_to(ROOT)}` ({grade_sha}).",
            f"Centered moments: `{MOMENTS.relative_to(ROOT)}` ({grade['moments_artifact']['sha256']}).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    require(GRADE_JSON.is_file() and MOMENTS.is_file() and GRADE_SOURCE.is_file(),
            "persisted grade JSON, moments, or source is missing")
    require(not MARKDOWN.exists(), "refusing to overwrite existing grade Markdown")
    require(not FAILURE_RECEIPT.exists(), "refusing to overwrite terminal-failure receipt")
    require(not FINALIZATION_RECEIPT.exists(), "refusing to overwrite finalization receipt")
    grade = json.loads(GRADE_JSON.read_text())
    require(grade.get("schema") == "wnn.fold1-baseline-pair.grade.r1", "wrong persisted grade schema")
    require(grade.get("status") == "evaluated", "persisted grade is not complete")
    require(grade.get("frozen_spec", {}).get("sha256") == SPEC_SHA256, "persisted grade spec pin changed")
    require(sha256(GRADE_SOURCE) == grade.get("grader_sha256"), "grader source differs from persisted score source")
    require(sha256(MOMENTS) == grade.get("moments_artifact", {}).get("sha256"), "centered moments hash mismatch")
    require(MOMENTS.stat().st_size == grade["moments_artifact"].get("bytes"), "centered moments size mismatch")
    require(grade.get("split", {}).get("evaluation_groups") == 457, "persisted evaluation group count mismatch")
    require(grade.get("split", {}).get("evaluation_components") == 149, "persisted component count mismatch")
    require(grade.get("primary_contrast", {}).get("n_boot") == 2000, "persisted bootstrap count mismatch")
    require(grade.get("primary_contrast", {}).get("seed") == 271828, "persisted bootstrap seed mismatch")

    failure = {
        "schema": "wnn.fold1-baseline-pair.grade-terminal-failure.r1",
        "observed_utc": grade["created_utc"],
        "attempted_command": (
            "scripts/limited .venv/bin/python scratch/fold1_gru384_successor_20261009/"
            "grading_r1/grade_fold1_baseline_pair_r1.py --root /home/vlad/development/wunder/"
            "wnn_connectome_starterpack --spec scratch/fold1_gru384_successor_20261009/"
            "grading_r1/FROZEN_BASELINE_PAIR_R1.json --expected-spec-sha256 " + SPEC_SHA256 +
            " --deadline-seconds 170"
        ),
        "process_exit_status": 1,
        "failure_stage": "post-score Markdown rendering",
        "exception": "KeyError: 'grade_json_path'",
        "diagnosis": "All 457 groups were scored and JSON plus centered-moment artifacts were persisted; renderer accessed report path fields that were set only after rendering. No scoring or bootstrap failure occurred.",
        "persisted_grade_json": {
            "path": str(GRADE_JSON.relative_to(ROOT)),
            "sha256": sha256(GRADE_JSON),
            "bytes": GRADE_JSON.stat().st_size,
        },
        "persisted_moments": {
            "path": str(MOMENTS.relative_to(ROOT)),
            "sha256": sha256(MOMENTS),
            "bytes": MOMENTS.stat().st_size,
        },
        "score_pass_runtime_seconds": grade["runtime_seconds"],
        "source_sha256": sha256(GRADE_SOURCE),
        "preservation": "R1 score JSON, moments, and grader source remain immutable; R2 only finalizes presentation and records the prior terminal error.",
    }
    failure_tmp = FAILURE_RECEIPT.with_suffix(".json.tmp")
    failure_tmp.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
    os.replace(failure_tmp, FAILURE_RECEIPT)

    markdown = render(grade, sha256(GRADE_JSON))
    markdown_tmp = MARKDOWN.with_suffix(".md.tmp")
    markdown_tmp.write_text(markdown)
    os.replace(markdown_tmp, MARKDOWN)
    finalization = {
        "schema": "wnn.fold1-baseline-pair.finalization.r2",
        "created_utc": datetime.now(UTC).isoformat(),
        "scoring_reused_without_rescore": True,
        "frozen_spec_sha256": SPEC_SHA256,
        "grade_source": {"path": str(GRADE_SOURCE.relative_to(ROOT)), "sha256": sha256(GRADE_SOURCE)},
        "grade_json": {"path": str(GRADE_JSON.relative_to(ROOT)), "sha256": sha256(GRADE_JSON)},
        "moments": {"path": str(MOMENTS.relative_to(ROOT)), "sha256": sha256(MOMENTS)},
        "markdown": {"path": str(MARKDOWN.relative_to(ROOT)), "sha256": sha256(MARKDOWN)},
        "terminal_failure_receipt": {
            "path": str(FAILURE_RECEIPT.relative_to(ROOT)),
            "sha256": sha256(FAILURE_RECEIPT),
        },
        "repair_source": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(Path(__file__))},
        "scope": "presentation and durable terminal provenance only; no fit, prediction, metric, or bootstrap rerun",
    }
    final_tmp = FINALIZATION_RECEIPT.with_suffix(".json.tmp")
    final_tmp.write_text(json.dumps(finalization, indent=2, allow_nan=False) + "\n")
    os.replace(final_tmp, FINALIZATION_RECEIPT)
    print(json.dumps(finalization, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
