from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_rkd_experiment import run_diagnostic, run_smoke_test


def test_synthetic_smoke_checks_geometry_without_training_or_score() -> None:
    report = run_smoke_test()
    assert report["fit_performed"] is False
    assert report["grader_or_score_used"] is False
    assert report["matched_relation_loss"] < 1e-7
    assert report["scale_translation_relation_loss"] < 1e-6
    assert report["time_permutation_relation_loss"] > 1e-3


def test_read_only_full_shape_cache_diagnostic_is_hashed_and_write_once(tmp_path: Path) -> None:
    rows = np.arange(20_000, dtype=np.float32)
    teacher = np.stack((np.sin(rows / 11), np.cos(rows / 17)), axis=-1)[None]
    student = 13.0 * teacher + np.array([8.0, -3.0], dtype=np.float32)
    valid = np.ones((1, 20_000), dtype=np.bool_)
    valid[:, :128] = False
    teacher_path = tmp_path / "main_source_train.npy"
    student_path = tmp_path / "student.npy"
    valid_path = tmp_path / "valid.npy"
    np.save(teacher_path, teacher)
    np.save(student_path, student.astype(np.float32))
    np.save(valid_path, valid)
    teacher_sha = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    metadata = {
        "format": "wnn.heterogeneous-distillation-cache",
        "version": 1,
        "source_coefficients": [{"coefficient": value} for value in (0.30, 0.20, 0.25, 0.0625)],
        "coefficient_sum": 0.8125,
        "files": {
            "main_source_train": {
                "path": teacher_path.name,
                "sha256": teacher_sha,
                "bytes": teacher_path.stat().st_size,
                "shape": [1, 20_000, 2],
                "dtype": "float32",
            }
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    output_json = tmp_path / "diagnostic.json"
    output_markdown = tmp_path / "diagnostic.md"

    metadata_before = (tmp_path / "metadata.json").read_bytes()
    with pytest.raises(ValueError, match="must not overwrite an input artifact"):
        run_diagnostic(
            teacher_path=teacher_path,
            student_path=student_path,
            valid_path=valid_path,
            output_json=tmp_path / "metadata.json",
            output_markdown=None,
            batch_size=1,
            time_chunk=20_000,
            seed=10,
            max_pairs=64,
            coefficient_total=0.8125,
        )
    assert (tmp_path / "metadata.json").read_bytes() == metadata_before

    report = run_diagnostic(
        teacher_path=teacher_path,
        student_path=student_path,
        valid_path=valid_path,
        output_json=output_json,
        output_markdown=output_markdown,
        batch_size=1,
        time_chunk=2_000,
        seed=10,
        max_pairs=64,
        coefficient_total=0.8125,
    )

    assert len(report["chunk_losses"]) == 10
    assert report["inputs"]["teacher_cache"]["shape"] == [1, 20_000, 2]
    assert report["inputs"]["teacher_cache"]["metadata_sha256"]
    assert report["inputs"]["valid_mask"]["dtype"] == "bool"
    assert report["configuration"]["read_mode"].startswith("numpy mmap_mode='r'")
    assert report["fit_performed"] is False
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["source_coefficient_total"] == 0.8125
    assert output_markdown.is_file()

    with pytest.raises(FileExistsError):
        run_diagnostic(
            teacher_path=teacher_path,
            student_path=student_path,
            valid_path=valid_path,
            output_json=output_json,
            output_markdown=output_markdown,
            batch_size=1,
            time_chunk=20_000,
            seed=10,
            max_pairs=64,
            coefficient_total=0.8125,
        )


def test_diagnostic_rejects_nonregistered_source_scale(tmp_path: Path) -> None:
    teacher_path = tmp_path / "main_source_train.npy"
    student_path = tmp_path / "student.npy"
    np.save(teacher_path, np.zeros((1, 20_000, 2), dtype=np.float32))
    np.save(student_path, np.zeros((1, 20_000, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="coefficient total must equal"):
        run_diagnostic(
            teacher_path=teacher_path,
            student_path=student_path,
            valid_path=None,
            output_json=tmp_path / "no-output.json",
            output_markdown=None,
            batch_size=1,
            time_chunk=20_000,
            seed=0,
            max_pairs=2,
            coefficient_total=1.0,
        )
    assert not (tmp_path / "no-output.json").exists()


def test_diagnostic_rejects_metadata_that_claims_the_wrong_source_total(tmp_path: Path) -> None:
    teacher_path = tmp_path / "main_source_train.npy"
    student_path = tmp_path / "student.npy"
    np.save(teacher_path, np.zeros((1, 20_000, 2), dtype=np.float32))
    np.save(student_path, np.zeros((1, 20_000, 2), dtype=np.float32))
    metadata = {
        "format": "wnn.heterogeneous-distillation-cache",
        "version": 1,
        "source_coefficients": [{"coefficient": 0.5}],
        "coefficient_sum": 0.5,
        "files": {
            "main_source_train": {
                "path": teacher_path.name,
                "sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                "bytes": teacher_path.stat().st_size,
                "shape": [1, 20_000, 2],
                "dtype": "float32",
            }
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="coefficients must sum to 0.8125"):
        run_diagnostic(
            teacher_path=teacher_path,
            student_path=student_path,
            valid_path=None,
            output_json=tmp_path / "no-output.json",
            output_markdown=None,
            batch_size=1,
            time_chunk=20_000,
            seed=0,
            max_pairs=2,
            coefficient_total=0.8125,
        )
    assert not (tmp_path / "no-output.json").exists()
