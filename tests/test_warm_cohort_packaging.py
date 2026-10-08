"""Offline checks for the hash-bound warm-cohort source successor."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "kaggle_warm_cohort"))
import kernel_driver
import stage_warm_cohort


def test_hash_bound_source_stage_uses_manifest_and_opaque_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main = tmp_path / "main"
    parent = main / "artifacts" / "runs" / "parent"
    parent.mkdir(parents=True)
    (parent / "model_ep3.pt").write_bytes(b"model")
    (parent / "provenance.json").write_text('{"run_id":"parent"}\n')
    monkeypatch.setattr(stage_warm_cohort, "MAIN", main)
    monkeypatch.setattr(stage_warm_cohort, "PARENT_RUN", "parent")
    monkeypatch.setattr(stage_warm_cohort, "PARENT_FILES", ("model_ep3.pt", "provenance.json"))
    monkeypatch.setattr(
        stage_warm_cohort,
        "write_archive",
        lambda path, _specs: path.write_bytes(b"sealed source bundle"),
    )

    upload = tmp_path / "packet" / "source_upload"
    stage_warm_cohort.stage_hash_bound_source(upload, [{"run_id": "smoke-v3"}])

    manifest = json.loads((upload / "assets-manifest.json").read_text())
    assert manifest["dataset"] == "vladkochetov07/wnn-warm-cohort-src-20261008-v2"
    assert len(manifest["files"]) == 3
    assert {entry["path"] for entry in manifest["files"]} == {
        "wnn_src.tar.gz",
        "artifacts/runs/parent/model_ep3.pt",
        "artifacts/runs/parent/provenance.json",
    }
    for entry in manifest["files"]:
        payload = (upload / entry["filename"]).read_bytes()
        assert entry["filename"].endswith(".wnn")
        assert len(payload) == entry["size"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    assert (upload / "dataset-metadata.json").is_file()
    assert len(list(upload.iterdir())) == 5


def test_queue_specs_are_idempotent_across_driver_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "working"
    monkeypatch.setattr(kernel_driver, "WORK", work)
    spec = {"run_id": "resume-me", "gpu": 0, "steps": [{"name": "fit"}]}

    assert kernel_driver.write_queues([spec]) == {0: 1, 1: 0}
    pending = work / "queues" / "gpu0" / "pending" / "resume-me.json"
    running = work / "queues" / "gpu0" / "running" / pending.name
    done = work / "queues" / "gpu0" / "done" / pending.name
    encoded = pending.read_bytes()

    assert kernel_driver.write_queues([spec]) == {0: 1, 1: 0}
    running.parent.mkdir(parents=True)
    pending.rename(running)
    assert kernel_driver.write_queues([spec]) == {0: 1, 1: 0}
    assert not pending.exists() and running.read_bytes() == encoded
    done.parent.mkdir(parents=True)
    running.rename(done)
    assert kernel_driver.write_queues([spec]) == {0: 0, 1: 0}
    assert not pending.exists() and done.read_bytes() == encoded

    changed = {**spec, "steps": [{"name": "different"}]}
    with pytest.raises(RuntimeError, match="differs from frozen spec"):
        kernel_driver.write_queues([changed])


def test_boot_marker_is_a_durable_file_before_input_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "working"
    monkeypatch.setattr(kernel_driver, "WORK", work)
    monkeypatch.setattr(kernel_driver, "INPUT", tmp_path / "input")

    kernel_driver.write_boot_marker()

    marker = json.loads((work / "boot_marker.json").read_text())
    assert marker["event"] == "kernel_boot"
    assert marker["input_root"] == str(tmp_path / "input")
