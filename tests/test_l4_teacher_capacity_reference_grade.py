"""Regression checks for PR60 reference-sidecar checkpoint layouts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.l4_teacher_capacity_20261104.continue_students import (
    _collect_pr60_reference_member,
)
from scripts.w18_l4_quality_grade_20261104 import SPLIT_SHA256
from wnn.splits import file_sha256


def test_pr60_reference_uses_sidecar_model_filename_for_epoch_one_and_three(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    row_order = [31]
    for run, epoch in (("w18_reference", 1), ("tiedlead_reference", 3)):
        prediction = project / "cache" / f"pred_{run}_valid.npy"
        sidecar = prediction.with_suffix(".json")
        run_root = project / "artifacts" / "runs" / run
        checkpoint = run_root / "model.pt"
        provenance = run_root / "provenance.json"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(parents=True, exist_ok=True)
        prediction.write_bytes(b"fixed prediction bytes")
        checkpoint.write_bytes(b"fixed checkpoint bytes")
        provenance.write_text("{}\n", encoding="utf-8")
        sidecar.write_text(
            json.dumps(
                {
                    "run": run,
                    "split": "valid",
                    "epoch": epoch,
                    "checkpoint_file": "model.pt",
                    "split_manifest_sha256": SPLIT_SHA256,
                    "inference_precision": "fp32",
                    "row_groups": row_order,
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "prediction_sha256": file_sha256(prediction),
                }
            ),
            encoding="utf-8",
        )
        reference = {
            "run": run,
            "prediction_path": str(prediction),
            "sidecar_path": str(sidecar),
            "checkpoint_sha256": file_sha256(checkpoint),
            "prediction_sha256": file_sha256(prediction),
        }

        record = _collect_pr60_reference_member(reference, epoch, row_order)

        assert record["kind"] == "legacy_bank"
        assert record["checkpoint"]["path"] == str(checkpoint.resolve())
        assert record["checkpoint"]["sha256"] == file_sha256(checkpoint)

