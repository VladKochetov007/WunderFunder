from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.l4_teacher_capacity_20261104 import publish_r5_results as publisher


def test_results_body_uses_grade_wmse_and_fixed_blend_rows() -> None:
    grades = {
        "wide_gru": {
            "candidate_rows": [
                {
                    "name": "TL3_plus_wide_gru_s1001",
                    "score": {"wp": 0.701},
                    "wmse": {"wmse": 0.202},
                    "delta_to_tl3": 0.001,
                },
                {
                    "name": "TL3_plus_wide_gru_s1002",
                    "score": {"wp": 0.702},
                    "wmse": {"wmse": 0.203},
                    "delta_to_tl3": 0.002,
                },
            ],
            "matched_student_deltas": {
                "1001": {
                    "student_wp": 0.69,
                    "control_wp": 0.68,
                    "delta_student_minus_control_wp": 0.01,
                }
            },
        },
        "causal_transformer": {
            "candidate_rows": [
                {
                    "name": "TL3_plus_causal_transformer_s1001",
                    "score": {"wp": 0.703},
                    "wmse": {"wmse": 0.204},
                    "delta_to_tl3": 0.003,
                },
                {
                    "name": "TL3_plus_causal_transformer_s1002",
                    "score": {"wp": 0.704},
                    "wmse": {"wmse": 0.205},
                    "delta_to_tl3": 0.004,
                },
            ],
            "matched_student_deltas": {
                "1002": {
                    "student_wp": 0.67,
                    "control_wp": 0.68,
                    "delta_student_minus_control_wp": -0.01,
                }
            },
        },
    }
    body = publisher._grade_body(grades, {"html_url": "https://github.test/previous"})
    assert "0.701 | 0.202 | 0.001" in body
    assert "0.704 | 0.205 | 0.004" in body
    assert "0.69 | 0.68 | 0.01" in body
    assert "0.67 | 0.68 | -0.01" in body
    assert "90/10" in body
    assert "public score" in body


def test_status_body_keeps_fit_and_profile_counts_separate() -> None:
    row = {
        "family": "causal_transformer",
        "run": "teacher_s1001",
        "status": "running",
        "phase": "training, epoch 2",
        "epoch": 2,
        "valid_wp": 0.68,
        "valid_wmse": 0.21,
        "grad_norm": {"q50": 1.2},
        "clip": 1.0,
    }
    body = publisher._progress_body([row], "hourly", profile_counts={"completed": 13, "running": 2})
    assert "Scientific fits completed: 0/12" in body
    assert "training, epoch 2" in body
    assert "Profile receipts found: 15" in body
    assert "'completed': 13" in body


def test_archive_writer_reuses_only_matching_manifest_and_bytes(tmp_path: Path) -> None:
    source = tmp_path / "receipt.json"
    source.write_text('{"status":"complete"}\n', encoding="utf-8")
    archive_path = tmp_path / "evidence.zip"
    entries = [(source, "receipts/grade.json")]
    metadata = {"schema": "test.v1", "kind": "evidence"}

    first = publisher._verify_or_write_archive(archive_path, entries, metadata)
    second = publisher._verify_or_write_archive(archive_path, entries, metadata)
    assert first["sha256"] == second["sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
    assert manifest["files"][0]["path"] == "receipts/grade.json"

    source.write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from current verified inputs"):
        publisher._verify_or_write_archive(archive_path, entries, metadata)
