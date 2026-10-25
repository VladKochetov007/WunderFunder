"""Receiver-only checks for portable metadata export without model loading."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PACKET = (
    Path(__file__).resolve().parents[1] / "research/experiment_drafts/"
    "b300-transformer-distillation-runtime/recovery_r5"
)
sys.path.insert(0, str(PACKET))
_spec = importlib.util.spec_from_file_location("recovery_export", PACKET / "export_midas.py")
assert _spec is not None and _spec.loader is not None
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


def fixture_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide synthetic metadata; this is not a trained model or quality evidence."""
    fit = root / "runs/cell/fit"
    fit.mkdir(parents=True)
    config = {
        "epochs": 30,
        "expected_fold": 0,
        "seed": 201001,
        "model": {},
        "precision_mode": "bf16_bf16x9",
    }
    (root / "config.json").write_text(json.dumps(config))
    split = {"fold": 0, "schema": "wnn.sequence-inner-split.v1"}
    split_raw = exporter.encoded(split)
    split_sha = exporter.hashlib.sha256(split_raw).hexdigest()
    (root / "split.json").write_bytes(split_raw)
    normalized = json.loads(json.dumps(config))
    normalized["model"]["precision_mode"] = config["precision_mode"]
    metadata = {
        "source": {"sha256": "a" * 64},
        "config": normalized,
        "provenance": {"data": {"split_sha256": split_sha, "fold": 0}},
    }
    (fit / "run_metadata.json").write_bytes(exporter.encoded(metadata))
    campaign = {"campaign_sha256": "b" * 64, "runtime_source_sha256": "a" * 64}
    cell = {
        "run_dir": "runs/cell",
        "config": "config.json",
        "split_sha256": split_sha,
        "split_manifest": str(root / "split.json"),
        "id": "cell",
        "variant": "lr_1e-4",
        "fold": 0,
        "seed": 201001,
    }
    monkeypatch.setattr(exporter, "load_recovery", lambda _: (campaign, {}))
    monkeypatch.setattr(exporter, "get_recovery_cell", lambda *_: cell)
    return fit


def test_missing_terminal_does_not_claim_live_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A metadata snapshot cannot prove that a GPU process is running or completed."""
    fixture_run(tmp_path, monkeypatch)
    path = exporter.export_bundle(tmp_path, "cell", tmp_path / "bundle")
    receipt = json.loads(path.read_text())
    assert receipt["execution"] == {"status": "partial", "started_at": None, "finished_at": None}
    assert receipt["source"]["commit"] is None
    assert receipt["metrics"] == []
    assert receipt["checkpoints"] == []


def test_source_mismatch_blocks_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An artifact from a different runtime cannot be relabeled as the planned cell."""
    fit = fixture_run(tmp_path, monkeypatch)
    metadata = json.loads((fit / "run_metadata.json").read_text())
    metadata["source"]["sha256"] = "c" * 64
    (fit / "run_metadata.json").write_bytes(exporter.encoded(metadata))
    with pytest.raises(ValueError, match="source/split"):
        exporter.export_bundle(tmp_path, "cell", tmp_path / "bundle")


def test_bundle_is_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing human-transferred bundle remains byte-identical on accidental replay."""
    fixture_run(tmp_path, monkeypatch)
    path = exporter.export_bundle(tmp_path, "cell", tmp_path / "bundle")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        exporter.export_bundle(tmp_path, "cell", tmp_path / "bundle")
    assert path.read_bytes() == before
