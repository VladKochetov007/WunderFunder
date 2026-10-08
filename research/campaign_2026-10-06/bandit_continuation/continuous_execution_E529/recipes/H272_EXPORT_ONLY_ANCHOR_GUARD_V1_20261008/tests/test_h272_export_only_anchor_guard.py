"""Tests for the narrowly scoped H272 export-only parent-anchor correction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ADAPTER_PATH = Path(__file__).resolve().parents[1] / "h272_export_only_anchor_guard.py"
SIDECAR_PATH = Path(__file__).resolve().parent / "fixtures/h233_c1_valid_sidecar.json"
SIDECAR_SHA256 = "bb335392edc599f181c0bbcb6523fd430b036647454bc0d76facf5b592211ad3"


def _load_adapter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("h272_anchor_guard_under_test", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def _load_actual_h233_sidecar() -> dict[str, Any]:
    raw = SIDECAR_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SIDECAR_SHA256
    sidecar = json.loads(raw)
    assert sidecar["run"] == "bd26_c1v2_s3_f0"
    assert sidecar["split"] == "valid"
    assert sidecar["epoch"] == 1
    assert len(sidecar["row_groups"]) == 1_873
    return sidecar


def _prepare_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar: dict[str, Any],
) -> tuple[ModuleType, Path, list[Path]]:
    adapter = _load_adapter()
    exporter = adapter._load_frozen_exporter()
    adapter.install_corrected_parent_anchor_guard(exporter)
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    prediction = tmp_path / "cache/pred_bd26_c1v2_s3_f0_valid.npy"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"fixture prediction; export guard hashes full bytes")
    prediction.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    hashed: list[Path] = []

    def expected_prediction_hash(path: Path) -> str:
        hashed.append(path)
        return exporter.PARENT_VALID_PREDICTION_SHA256

    monkeypatch.setattr(exporter, "file_sha256", expected_prediction_hash)
    return exporter, prediction, hashed


def test_actual_h233_sidecar_binds_epoch_checkpoint_and_preserves_hash_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = _load_actual_h233_sidecar()
    adapter = _load_adapter()
    observed = adapter.verify_frozen_h272_source()
    exporter = adapter._load_frozen_exporter()
    assert len(observed) == 56
    assert sidecar["checkpoint_sha256"] == exporter.CHECKPOINT_SHA256
    assert sidecar["checkpoint_sha256"] != exporter.PARENT_MODEL_PT_SHA256

    exporter, prediction, hashed = _prepare_guard(tmp_path, monkeypatch, sidecar)
    result = exporter._validate_anchor_reference("valid", tuple(sidecar["row_groups"]))

    assert result is not None
    assert result["prediction_path"] == "cache/pred_bd26_c1v2_s3_f0_valid.npy"
    assert result["prediction_sha256"] == exporter.PARENT_VALID_PREDICTION_SHA256
    assert result["checkpoint_sha256"] == exporter.CHECKPOINT_SHA256
    assert hashed == [prediction]


def test_wrong_producer_checkpoint_fails_before_prediction_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = _load_actual_h233_sidecar()
    sidecar["checkpoint_sha256"] = "0" * 64
    exporter, _, hashed = _prepare_guard(tmp_path, monkeypatch, sidecar)

    with pytest.raises(ValueError, match="frozen model_ep1.pt"):
        exporter._validate_anchor_reference("valid", tuple(sidecar["row_groups"]))

    assert hashed == []


def test_train_split_does_not_read_parent_anchor() -> None:
    adapter = _load_adapter()
    exporter = adapter._load_frozen_exporter()
    adapter.install_corrected_parent_anchor_guard(exporter)

    assert exporter._validate_anchor_reference("train", ()) is None


@pytest.mark.parametrize("subcommand", ("profile", "fit"))
def test_adapter_refuses_profile_and_fit_subcommands(
    monkeypatch: pytest.MonkeyPatch, subcommand: str
) -> None:
    adapter = _load_adapter()
    monkeypatch.setattr("sys.argv", [str(ADAPTER_PATH), subcommand])

    with pytest.raises(SystemExit, match="only the export subcommand"):
        adapter.main()


@pytest.mark.parametrize(
    "arguments",
    (
        ("--split", "train"),
        ("--split=train",),
        ("--split", "valid", "--split", "train"),
    ),
)
def test_adapter_refuses_nonvalid_export_splits(
    monkeypatch: pytest.MonkeyPatch, arguments: tuple[str, ...]
) -> None:
    adapter = _load_adapter()
    monkeypatch.setattr("sys.argv", [str(ADAPTER_PATH), "export", *arguments])

    with pytest.raises(SystemExit, match="only the valid export split"):
        adapter.main()


def test_adapter_preserves_valid_split_default() -> None:
    adapter = _load_adapter()

    adapter._require_valid_split([])
    adapter._require_valid_split(["--split", "valid"])
    adapter._require_valid_split(["--split=valid"])
