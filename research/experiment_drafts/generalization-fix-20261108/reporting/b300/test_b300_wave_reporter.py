from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from b300_wave_evidence import (
    MANIFEST,
    NORMALIZED_CONFIGS,
    RAW_CONFIGS,
    SOURCE,
    SPLIT,
    completion,
    full_valid,
    load_panel,
    snapshot,
)
from b300_wave_report import render


def panel_fixture(path: Path) -> tuple[dict, str]:
    value = {
        "expected_training_source_sha256": SOURCE,
        "source_manifest_sha256": MANIFEST,
        "cells": [
            {
                "cell": name,
                "config_sha256": sha,
                "normalized_config_sha256": NORMALIZED_CONFIGS[name],
            }
            for name, sha in RAW_CONFIGS.items()
        ],
    }
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return value, hashlib.sha256(raw).hexdigest()


def test_panel_refuses_alias_source_config_and_byte_digest(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    value, digest = panel_fixture(path)
    assert set(load_panel(path, digest)["cells"]) == set(RAW_CONFIGS)
    with pytest.raises(ValueError, match="digest"):
        load_panel(path, "0" * 64)
    value["cells"][0]["cell"] = "labels_s101001"
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="four-cell"):
        load_panel(path, hashlib.sha256(raw).hexdigest())
    value, _ = panel_fixture(path)
    value["cells"][0]["normalized_config_sha256"] = "0" * 64
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="four-cell"):
        load_panel(path, hashlib.sha256(raw).hexdigest())


def test_exact_four_epoch_complete_coverage_and_profile_refusal(tmp_path: Path) -> None:
    context = {"split_sha256": SPLIT, "evaluation_member_count": 1873}
    metrics = {"valid/sequence_count": 1873, "valid/selected_rows": 3528273} | {
        key: 0.6
        for key in (
            "valid/wp",
            "valid/wp/t0",
            "valid/wp/t1",
            "valid/z/wp",
            "valid/z/wp/t0",
            "valid/z/wp/t1",
        )
    }
    events = [
        {
            "kind": "evaluation",
            "epoch": epoch,
            "step": 820 * epoch,
            "context": context,
            "metrics": metrics,
        }
        for epoch in range(1, 5)
    ]
    assert all(full_valid(event, context) for event in events)
    ids = {"source": SOURCE, "config": "c" * 64, "data": "d" * 64}
    identity = {f"{key}_sha256": value for key, value in ids.items()} | {"world_size": 2}
    terminal = {
        "status": "completed",
        "completed_epochs": 4,
        "global_updates": 3280,
        "model_sha256": "f" * 64,
        "identity": identity,
    }
    assert completion(terminal, ids, events)
    assert not completion(terminal | {"completed_epochs": 30}, ids, events)
    assert not completion(terminal | {"global_updates": 3279}, ids, events)
    assert not completion(terminal | {"status": "profile_complete"}, ids, events)
    assert not completion(terminal, ids, events[:-1])
    assert not completion(terminal, ids, events[:-1] + [events[-2]])
    assert not full_valid(events[0] | {"metrics": metrics | {"valid/z/wp": float("nan")}}, context)
    path = tmp_path / "plan.json"
    _, digest = panel_fixture(path)
    panel = load_panel(path, digest)
    (tmp_path / "b300-labels_s101101-profile").mkdir()
    value = snapshot(tmp_path, panel, 100)
    assert "0/4 fits complete" in render(value)
    assert "0/4 fixed4E" in render(value)


def test_b300_dry_run_has_no_api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import report_b300_wave

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run constructed a live API client")

    path = tmp_path / "plan.json"
    _, digest = panel_fixture(path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"native_root": str(tmp_path)}))
    monkeypatch.setattr(report_b300_wave, "ProgressGitHub", refuse)
    monkeypatch.setattr(report_b300_wave.os, "write", lambda *args: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--mirror-config",
            str(config),
            "--receiver-plan",
            str(path),
            "--receiver-plan-sha256",
            digest,
            "--pr",
            "70",
            "--state",
            str(tmp_path / "state.json"),
            "--dry-run",
            "--once",
        ],
    )
    assert report_b300_wave.main() == 0
    assert not (tmp_path / "state.json").exists()
