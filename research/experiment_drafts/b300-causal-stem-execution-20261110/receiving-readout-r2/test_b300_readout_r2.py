from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).with_name("b300_readout_r2.py")
SPEC = importlib.util.spec_from_file_location("b300_readout_r2_fixture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
readout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readout)


def test_score_agreement_checks_each_channel_at_fixed_tolerance() -> None:
    monitor = {"valid/wp": 0.6, "valid/wp/t0": 0.5, "valid/wp/t1": 0.7}
    close = {"score": {"wp": 0.600004, "t0": 0.500004, "t1": 0.700004}}
    mismatch = {"score": {"wp": 0.6, "t0": 0.500011, "t1": 0.7}}
    nonfinite = {"score": {"wp": 0.6, "t0": 0.5, "t1": float("nan")}}

    assert readout._score_agrees_with_monitor(close, monitor)
    assert not readout._score_agrees_with_monitor(mismatch, monitor)
    assert not readout._score_agrees_with_monitor(nonfinite, monitor)


def test_fit_events_require_complete_json_lines(tmp_path: Path) -> None:
    journal = tmp_path / "monitor_evaluations.jsonl"
    journal.write_text('{"epoch":1}\n', encoding="utf-8")
    assert readout._read_fit_events(journal) == [{"epoch": 1}]

    journal.write_text('{"epoch":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="partial trailing record"):
        readout._read_fit_events(journal)


def test_fit_inventory_is_hash_bound(tmp_path: Path) -> None:
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    checkpoint = fit_dir / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    core = SimpleNamespace(
        sha256_file=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    )
    original = readout._artifact_inventory(core, fit_dir)
    checkpoint.write_bytes(b"changed")
    assert original != readout._artifact_inventory(core, fit_dir)


def test_fit_inventory_rejects_linked_directory(tmp_path: Path) -> None:
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    linked_fit = tmp_path / "linked-fit"
    try:
        linked_fit.symlink_to(fit_dir, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable in this environment: {error}")
    core = SimpleNamespace(
        sha256_file=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="missing or linked"):
        readout._artifact_inventory(core, linked_fit)
