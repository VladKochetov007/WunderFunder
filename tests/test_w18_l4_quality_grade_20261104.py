"""Focused CPU checks for the frozen W18 L4 grading adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.w18_l4_quality_grade_20261104 import (
    collect_member,
    fixed_ensembles,
    matched_comparisons,
    reference_checkpoint_filename,
    validate_panel,
)
from wnn.splits import file_sha256


def test_fixed_ensembles_are_exactly_tl3_and_eight_fixed_nineten_blends() -> None:
    names = ["w18", "tiedlead_s5", "tiedlead_s6"] + [
        f"{configuration}_s{seed}"
        for configuration in ("control", "innovations", "alpha50", "family01")
        for seed in (1001, 1002)
    ]

    ensembles = fixed_ensembles(names)

    assert len(ensembles) == 9
    assert ensembles["TL3"] == {"w18": 0.34, "tiedlead_s5": 0.33, "tiedlead_s6": 0.33}
    assert len([name for name in ensembles if name.startswith("TL3_plus_")]) == 8
    assert all(
        weights["TL3"] == 0.9 and len(weights) == 2 and next(iter(weights.values())) == 0.9
        for name, weights in ensembles.items()
        if name.startswith("TL3_plus_")
    )


def test_panel_uses_project_symlink_root_for_datasets_and_cache(tmp_path: Path) -> None:
    project_root = tmp_path / "runtime" / "project"
    panel = {
        "schema": "wnn.w18-l4-cohort.v1",
        "status": "recipes_frozen",
        "identity": {"split_sha256": "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"},
        "paths": {"project_root": str(project_root), "split_manifest": str(tmp_path / "split.json")},
        "endpoint": {"checkpoint_epoch": 1, "epochs": 2},
        "cells": [
            {"configuration_id": config, "seed": seed, "run_name": f"{config}_{seed}"}
            for config in ("control", "innovations", "alpha50", "family01")
            for seed in (1001, 1002)
        ],
    }

    validated_project, data_root, _cells = validate_panel(panel, "f3e0327fee7de17e445f4ddb8ed5983af8348490ac3ab7a2887cf19deb5661b2")

    assert validated_project == project_root
    assert data_root == project_root


def test_reference_sidecar_accepts_only_exact_known_checkpoint_filenames(tmp_path: Path) -> None:
    model_path = tmp_path / "w18.json"
    model_path.write_text(json.dumps({"checkpoint_file": "model.pt"}), encoding="utf-8")
    epoch_path = tmp_path / "tiedlead.json"
    epoch_path.write_text(json.dumps({"checkpoint_file": "model_ep3.pt"}), encoding="utf-8")
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps({"checkpoint_file": "other.pt"}), encoding="utf-8")

    assert reference_checkpoint_filename(model_path, 1) == "model.pt"
    assert reference_checkpoint_filename(epoch_path, 3) == "model_ep3.pt"
    with pytest.raises(ValueError, match="unsupported checkpoint filename"):
        reference_checkpoint_filename(invalid_path, 3)


def test_collect_member_pins_sidecar_assets_and_checks_row_order(tmp_path: Path) -> None:
    run = "test_member"
    epoch = 1
    prediction = tmp_path / "prediction.npy"
    metadata_path = tmp_path / "prediction.json"
    checkpoint = tmp_path / "model_ep1.pt"
    provenance = tmp_path / "provenance.json"
    prediction.write_bytes(b"prediction bytes")
    checkpoint.write_bytes(b"checkpoint bytes")
    provenance.write_text("{}\n", encoding="utf-8")
    metadata = {
        "run": run,
        "split": "valid",
        "epoch": epoch,
        "checkpoint_file": checkpoint.name,
        "split_manifest_sha256": "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c",
        "inference_precision": "fp32",
        "row_groups": [17, 2],
        "checkpoint_sha256": file_sha256(checkpoint),
        "prediction_sha256": file_sha256(prediction),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    record = collect_member(
        run=run,
        epoch=epoch,
        prediction=prediction,
        metadata_path=metadata_path,
        checkpoint=checkpoint,
        provenance=provenance,
        row_order=[17, 2],
    )

    assert record["kind"] == "legacy_bank"
    assert record["predictions"]["sha256"] == file_sha256(prediction)
    assert record["checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert record["provenance"]["sha256"] == file_sha256(provenance)
    incomplete = collect_member(
        run=run,
        epoch=epoch,
        prediction=tmp_path / "missing.npy",
        metadata_path=metadata_path,
        checkpoint=checkpoint,
        provenance=provenance,
        row_order=[17, 2],
    )
    assert incomplete["state"] == "pending"
    assert "predictions" in incomplete["reason"]
    with pytest.raises(ValueError, match="row order"):
        collect_member(
            run=run,
            epoch=epoch,
            prediction=prediction,
            metadata_path=metadata_path,
            checkpoint=checkpoint,
            provenance=provenance,
            row_order=[2, 17],
        )


def test_matched_comparisons_leave_missing_scores_null_and_compute_deltas() -> None:
    report = {
        "member_scores": {
            "control_s1001": {"wp": 0.70},
            "innovations_s1001": {"wp": 0.72},
        },
        "rows": [
            {
                "name": "TL3_plus_innovations_s1001",
                "score": {"wp": 0.74},
                "delta_to_tl3": 0.01,
            },
            {"name": "TL3_plus_control_s1001", "score": {"wp": 0.735}},
        ],
    }

    rows = matched_comparisons(report)
    innovations = next(
        row for row in rows if row["configuration"] == "innovations" and row["seed"] == 1001
    )
    missing = next(
        row for row in rows if row["configuration"] == "alpha50" and row["seed"] == 1001
    )

    assert innovations["delta_wp_vs_control"] == pytest.approx(0.02)
    assert innovations["fixed_blend_delta_wp_vs_tl3"] == pytest.approx(0.01)
    assert innovations["fixed_blend_delta_wp_vs_control_blend"] == pytest.approx(0.005)
    assert missing["delta_wp_vs_control"] is None
    assert missing["status"] == "incomplete"
