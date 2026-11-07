"""Synthetic replay/immutability checks and a CPU-only existing-plotter smoke test."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from epoch_snapshot import SPLIT_SHA, capture, complete_events
from watch_epoch_plots import EpochPlots

from wnn.multistage_search import canonical_json_sha256, file_sha256


def fixture_runs(root: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    """Write explicitly synthetic native/view journals with two recorded epochs."""
    run = root / "live-dashboard/runs/l4-fixture-fit"
    native = root / "live-dashboard/native/l4-fixture-fit"
    run.mkdir(parents=True)
    native.mkdir(parents=True)
    source_files = {"src/synthetic_fixture.py": hashlib.sha256(b"synthetic test only").hexdigest()}
    data = {
        "split_sha256": SPLIT_SHA,
        "fit_members": [["train", index] for index in range(10325)],
        "evaluation_members": [["valid", index] for index in range(1873)],
    }
    metadata = {
        "config": {"fixture": "synthetic test only"},
        "source": {"files": source_files, "sha256": canonical_json_sha256(source_files)},
        "provenance": {"data": data, "data_sha256": canonical_json_sha256(data)},
    }
    evaluations = [
        {
            "schema": "wnn.monitor.v1",
            "kind": "evaluation",
            "epoch": epoch,
            "step": epoch * 10,
            "wall_time": f"2026-11-07T00:00:0{epoch}+00:00",
            "metrics": {
                "train_eval/loss": 0.5,
                "valid/loss": 0.6,
                "train_eval/wp": 0.6,
                "valid/wp": 0.61,
                "train_eval/wp/t0": 0.6,
                "train_eval/wp/t1": 0.6,
                "valid/wp/t0": 0.62,
                "valid/wp/t1": 0.60,
                "valid/sequence_count": 1873,
                "valid/selected_rows": 3528273,
            },
            "context": {"fixture": "synthetic only", "evaluation_exposure": "development exposed"},
        }
        for epoch in (1, 2)
    ]
    train = {
        "schema": "wnn.monitor.v1",
        "kind": "train_step",
        "epoch": 0,
        "step": 0,
        "wall_time": "2026-11-07T00:00:00+00:00",
        "metrics": {
            "train/loss": 0.8,
            "optimizer/lr/main": 0.0003,
            "optimizer/gradient_norm_preclip": 0.2,
        },
        "context": {"fixture": "synthetic only"},
    }
    payloads = {
        "run_metadata.json": json.dumps(metadata).encode(),
        ".sequence_mirror_owner.json": json.dumps(
            {"host": "l4", "source_directory": "/synthetic-fixture-only/fit"}
        ).encode(),
        "monitor_evaluations.jsonl": b"".join(
            (json.dumps(event) + "\n").encode() for event in evaluations
        ),
        "monitor_events.jsonl": b"".join(
            (json.dumps(event) + "\n").encode() for event in [train, *evaluations]
        ),
        "monitor_observer.jsonl": (
            json.dumps(
                train | {"kind": "resources", "metrics": {"gpu/device_0/utilization_pct": 20}}
            )
            + "\n"
        ).encode(),
        "rank0.jsonl": b'{"step":1,"total_loss":0.8}\n{"step":11,"total_loss":0.7}\n',
    }
    for name, payload in payloads.items():
        (run / name).write_bytes(payload)
        (native / name).write_bytes(payload)
    (native / "collection_receipt.json").write_text(json.dumps({"fixture": "synthetic only"}))
    return run, native, evaluations


def test_delayed_replay_preserves_each_exact_epoch_prefix_and_existing_snapshot(
    tmp_path: Path,
) -> None:
    run, native, _ = fixture_runs(tmp_path)
    events = complete_events(run / "monitor_evaluations.jsonl")
    first = capture(run, native, tmp_path, *events[0])
    assert first is not None
    hashes = {path.name: file_sha256(path) for path in first.iterdir()}
    second = capture(run, native, tmp_path, *events[1])
    assert second is not None and second != first
    assert len(complete_events(first / "monitor_evaluations.jsonl")) == 1
    assert len(complete_events(second / "monitor_evaluations.jsonl")) == 2
    assert (first / "rank0.jsonl").read_bytes() == b'{"step":1,"total_loss":0.8}\n'
    assert capture(run, native, tmp_path, *events[0]) == first
    assert hashes == {path.name: file_sha256(path) for path in first.iterdir()}
    changed = events[0][0] | {"step": 9}
    with pytest.raises(ValueError, match="changed"):
        capture(run, native, tmp_path, changed, (json.dumps(changed) + "\n").encode())


def test_wrong_full_valid_mask_never_creates_snapshot(tmp_path: Path) -> None:
    run, _, events = fixture_runs(tmp_path)
    events[0]["metrics"]["valid/selected_rows"] -= 1
    (run / "monitor_evaluations.jsonl").write_text(json.dumps(events[0]) + "\n")
    with pytest.raises(ValueError, match="1873"):
        complete_events(run / "monitor_evaluations.jsonl")
    assert not (tmp_path / "snapshots").exists()


def test_long_B300_watch_does_not_extend_l4_discovery(tmp_path: Path) -> None:
    """The ROOT nine-day B300 allowance does not extend L4's24-hour discovery."""
    fixture_runs(tmp_path)
    watcher = EpochPlots(tmp_path, Path(__file__).resolve().parents[4], Path("unused-python"))
    watcher.scan(include_l4=False)
    assert not watcher.jobs
    assert not (tmp_path / "snapshots").exists()


@pytest.mark.skipif(
    "PLOT_PYTHON" not in os.environ, reason="set the provisioned CPU plotting Python"
)
def test_existing_plot_helper_creates_both_formats_with_single_child_and_replay(
    tmp_path: Path,
) -> None:
    fixture_runs(tmp_path)
    source = Path(__file__).resolve().parents[4]
    watcher = EpochPlots(tmp_path, source, Path(os.environ["PLOT_PYTHON"]))
    watcher.scan()
    assert len(watcher.jobs) == 2 and not watcher.errors
    watcher.start_next(120)
    assert watcher.active is not None
    pid = watcher.active[0].pid
    watcher.start_next(120)
    assert watcher.active[0].pid == pid
    deadline = time.monotonic() + 120
    while watcher.active is not None and time.monotonic() < deadline:
        watcher.finish()
        time.sleep(0.1)
    assert watcher.active is None
    job = next(iter(watcher.jobs.values()))
    assert job["status"] == "completed", job
    plots = Path(job["plots"])
    assert list(plots.glob("*.png")) and list(plots.glob("*.svg"))
    hashes = {path.name: file_sha256(path) for path in plots.iterdir()}
    replay = EpochPlots(tmp_path, source, Path(os.environ["PLOT_PYTHON"]))
    replay.scan()
    assert next(iter(replay.jobs.values()))["status"] == "completed"
    assert hashes == {path.name: file_sha256(path) for path in plots.iterdir()}
