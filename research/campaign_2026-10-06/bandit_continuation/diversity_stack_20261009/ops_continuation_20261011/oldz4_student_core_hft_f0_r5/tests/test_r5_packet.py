from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

PACKET_ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = PACKET_ROOT / "prepare_core_crop_r5.py"
STAGE_PATH = PACKET_ROOT / "run_hft_stage_r5.py"


def _import(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_checkpoint(root: Path) -> tuple[Path, Path]:
    source_dir = root / "artifacts/runs/fixture_source"
    source_dir.mkdir(parents=True)
    checkpoint_path = source_dir / "model_ep1.pt"
    provenance_path = source_dir / "provenance.json"
    checkpoint = {
        "epoch": 1,
        "model": {
            "head.2.weight": torch.arange(62, dtype=torch.float32).reshape(31, 2),
            "head.2.bias": torch.arange(31, dtype=torch.float32),
            "mean": torch.ones(3),
            "std": torch.full((3,), 2.0),
            "gru.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        },
        "args": {"distill_family_head": True, "distill_family_weight": 0.1},
        "provenance": {"args": {"distill_family_head": True, "distill_family_weight": 0.1}},
    }
    torch.save(checkpoint, checkpoint_path)
    provenance_path.write_text(
        json.dumps({"args": {"distill_family_head": True, "distill_family_weight": 0.1}}),
        encoding="utf-8",
    )
    return checkpoint_path, provenance_path


def test_frozen_crop_changes_only_the_family_output_rows(tmp_path: Path) -> None:
    prepare = _import(PREPARE_PATH, "prepare_core_crop_r5_test")
    source_checkpoint, source_provenance = _source_checkpoint(tmp_path)
    crop_source = PACKET_ROOT / "source_snapshots/core_crop_fe8/scripts/make_core_checkpoint.py"
    crop = prepare.import_crop_source(crop_source)
    crop.ROOT = tmp_path
    result = crop.derive_core(tmp_path, "fixture_source", 1, "fixture_core")
    output_dir = tmp_path / "artifacts/runs/fixture_core"
    verification = prepare.verify_crop(
        source_checkpoint,
        source_provenance,
        output_dir / "model_ep1.pt",
        output_dir / "provenance.json",
        source_run="fixture_source",
        source_epoch=1,
    )

    assert result["checkpoint_sha256"] == verification["output_checkpoint_sha256"]
    assert verification["head_rows_after"] == 23
    assert verification["dropped_family_rows"] == 8
    assert verification["unchanged_non_head_tensor_count"] == 3
    assert verification["normalization_tensor_count"] == 2


def test_core_crop_refuses_to_overwrite_a_destination(tmp_path: Path) -> None:
    prepare = _import(PREPARE_PATH, "prepare_core_crop_r5_no_overwrite_test")
    source_checkpoint, source_provenance = _source_checkpoint(tmp_path)
    crop_source = PACKET_ROOT / "source_snapshots/core_crop_fe8/scripts/make_core_checkpoint.py"
    crop = prepare.import_crop_source(crop_source)
    crop.ROOT = tmp_path
    crop.derive_core(tmp_path, "fixture_source", 1, "fixture_core")
    output_path = tmp_path / "artifacts/runs/fixture_core/model_ep1.pt"
    before = output_path.read_bytes()

    with pytest.raises(FileExistsError):
        crop.derive_core(tmp_path, "fixture_source", 1, "fixture_core")

    assert output_path.read_bytes() == before
    assert source_checkpoint.is_file()
    assert source_provenance.is_file()


def test_missing_root_profile_receipt_fails_closed(tmp_path: Path) -> None:
    stage = _import(STAGE_PATH, "run_hft_stage_r5_gate_test")

    with pytest.raises(FileNotFoundError, match="root-owned stage release"):
        stage._require_root_receipt(
            tmp_path / "missing.json",
            schema="wnn.oldz4-hft-profile-admission.v1",
            decision="PASS",
            bindings={"profile_run": "profile-run"},
        )


def test_root_receipt_binds_exact_stage_identity(tmp_path: Path) -> None:
    stage = _import(STAGE_PATH, "run_hft_stage_r5_identity_gate_test")
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema": "wnn.oldz4-hft-profile-admission.v1",
                "decision": "PASS",
                "profile_run": "expected-run",
            }
        ),
        encoding="utf-8",
    )

    receipt, digest = stage._require_root_receipt(
        path,
        schema="wnn.oldz4-hft-profile-admission.v1",
        decision="PASS",
        bindings={"profile_run": "expected-run"},
    )

    assert receipt["profile_run"] == "expected-run"
    assert len(digest) == 64
    with pytest.raises(ValueError, match="wrong profile_run"):
        stage._require_root_receipt(
            path,
            schema="wnn.oldz4-hft-profile-admission.v1",
            decision="PASS",
            bindings={"profile_run": "different-run"},
        )
