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


def test_native_transformer_owner_without_run_name_is_bound_by_frozen_cell(
    tmp_path: Path,
) -> None:
    source_sha, data_sha, split_sha, members_sha = (f"{letter}" * 64 for letter in "abcd")
    config_sha, source_git, panel_sha = "e" * 64, "f" * 40, "1" * 64
    output = tmp_path / "fit-output"
    command = ["python", "train_sequence.py", "--output", str(output)]
    panel = {
        "schema": "wnn.l4-teacher-capacity-transformer-panel.v1",
        "source_git": source_git,
        "identity": {
            "source_binding_sha256": source_sha,
            "data_binding_sha256": data_sha,
            "split_sha256": split_sha,
            "fit_members_sha256": members_sha,
        },
    }
    cell = {
        "cell_id": "tf_cell_s1001",
        "run_name": "tf_run_s1001",
        "fit_output": str(output),
        "config_sha256": config_sha,
        "normalized_config_sha256": "2" * 64,
    }
    receipt = {
        "command": command,
        "output_dir": str(output.resolve()),
        "owner": {
            "panel_sha256": panel_sha,
            "cell_id": cell["cell_id"],
            "source_binding_sha256": source_sha,
            "source_git": source_git,
            "data_binding_sha256": data_sha,
            "split_sha256": split_sha,
            "fit_members_sha256": members_sha,
            "config_sha256": config_sha,
        },
    }
    publisher._validate_native_tf_owner(panel, cell, receipt, panel_sha, command)

    receipt["output_dir"] = str(tmp_path / "wrong-output")
    with pytest.raises(ValueError, match="native Transformer fit identity differs"):
        publisher._validate_native_tf_owner(panel, cell, receipt, panel_sha, command)
    receipt["output_dir"] = str(output.resolve())
    receipt["owner"]["run_name"] = "other_run"
    with pytest.raises(ValueError, match="native Transformer fit identity differs"):
        publisher._validate_native_tf_owner(panel, cell, receipt, panel_sha, command)


def test_controller_receipt_completion_and_integer_deadline_handling(tmp_path: Path) -> None:
    assert publisher._controller_outcome({"status": "running"}) == "waiting"
    assert (
        publisher._controller_outcome({"status": "completed", "exit_code": 0, "timed_out": False})
        == "completed"
    )
    assert (
        publisher._controller_outcome({"status": "completed", "exit_code": 1, "timed_out": False})
        == "failed"
    )
    assert publisher._complete_deadline_matches(1_793_842_488, 1_793_842_488.8254418)
    assert not publisher._complete_deadline_matches(1_793_842_488.8254418, 1_793_842_488.8254418)
    receipt_path = tmp_path / "controller.json"
    receipt_path.write_text(
        json.dumps({"status": "running", "child": {"pid": 17}}), encoding="utf-8"
    )
    complete = {"controller_receipt": {"path": str(receipt_path), "child": {"pid": 17}}}
    assert publisher._r5_controller_outcome(complete) == "waiting"
    receipt_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "timed_out": False,
                "child": {"pid": 17},
            }
        ),
        encoding="utf-8",
    )
    assert publisher._r5_controller_outcome(complete) == "completed"
