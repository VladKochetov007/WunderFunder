"""Focused tests for the one-field H233 export anchor guard correction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = CAMPAIGN_ROOT / "recipes" / "H233_EXPORT_ONLY_ANCHOR_GUARD_V1_20261008.py"
ACTUAL_SIDECAR_PATH = CAMPAIGN_ROOT.parents[3] / "cache/pred_bd26_c1v2_s3_f0_valid.json"


def _load_adapter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("h233_export_guard_under_test", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def _load_actual_sidecar() -> dict[str, Any]:
    sidecar = json.loads(ACTUAL_SIDECAR_PATH.read_text())
    assert sidecar["run"] == "bd26_c1v2_s3_f0"
    assert sidecar["split"] == "valid"
    assert sidecar["epoch"] == 1
    assert len(sidecar["row_groups"]) == 1873
    return sidecar


def _prepare_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar: dict[str, Any],
) -> tuple[ModuleType, Path, list[Path]]:
    adapter = _load_adapter()
    exporter = adapter._load_frozen_exporter()
    adapter.install_corrected_anchor_guard(exporter)
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    prediction = tmp_path / "cache" / "pred_bd26_c1v2_s3_f0_valid.npy"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"small fixture; the real exporter still hashes the full array")
    prediction.with_suffix(".json").write_text(json.dumps(sidecar))
    hashed: list[Path] = []

    def expected_prediction_hash(path: Path) -> str:
        hashed.append(path)
        return exporter.PARENT_VALID_PREDICTION_SHA256

    monkeypatch.setattr(exporter, "file_sha256", expected_prediction_hash)
    return exporter, prediction, hashed


def test_actual_h233_sidecar_uses_epoch_checkpoint_hash_and_keeps_full_hash_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = _load_actual_sidecar()
    adapter = _load_adapter()
    exporter = adapter._load_frozen_exporter()
    assert sidecar["checkpoint_sha256"] == exporter.CHECKPOINT_SHA256
    assert sidecar["checkpoint_sha256"] != exporter.PARENT_MODEL_PT_SHA256

    # Keep the exact producer metadata and group order, while replacing only
    # the large prediction-file hash operation with a checked call sentinel.
    exporter, prediction, hashed = _prepare_guard(tmp_path, monkeypatch, sidecar)
    result = exporter._validate_anchor_reference("valid", tuple(sidecar["row_groups"]))

    assert result == {
        "prediction_path": "cache/pred_bd26_c1v2_s3_f0_valid.npy",
        "prediction_sha256": exporter.PARENT_VALID_PREDICTION_SHA256,
        "checkpoint_sha256": exporter.CHECKPOINT_SHA256,
        "identity_check": "E511 verified model.pt tensors equal the pinned model_ep1.pt tensors",
    }
    assert hashed == [prediction]


def test_wrong_sidecar_checkpoint_hash_fails_before_prediction_array_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = _load_actual_sidecar()
    sidecar["checkpoint_sha256"] = "0" * 64
    exporter, _, hashed = _prepare_guard(tmp_path, monkeypatch, sidecar)

    with pytest.raises(ValueError, match="frozen model_ep1.pt"):
        exporter._validate_anchor_reference("valid", tuple(sidecar["row_groups"]))

    assert hashed == []
