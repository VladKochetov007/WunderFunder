from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest


def _load_worker(candidate: Path) -> ModuleType:
    path = candidate / "code/scripts/loss_pair_gpu_worker.py"
    spec = importlib.util.spec_from_file_location("d02_r4_gpu_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_actual_gate_helper_runs_two_loss_profiles_with_distinct_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = Path(__file__).resolve().parents[1]
    worker = _load_worker(candidate)
    worker.WORK = tmp_path / "working"
    worker.OUT = worker.WORK / "out"
    worker.ROOT = tmp_path / "repo"
    worker.WORK.mkdir()
    worker.OUT.mkdir()
    worker.ROOT.mkdir()
    (worker.OUT / "logs").mkdir()
    (worker.OUT / "shared").mkdir()
    worker._SAMPLES.clear()
    worker._COMMANDS.clear()
    worker._STARTED = time.monotonic()

    child = r"""
import json
import os
from pathlib import Path
import sys

import numpy as np

gate = Path(os.environ["WNN_LOSS_PAIR_ARM_DIR"])
prefit = gate / "prefit"
prefit.mkdir(parents=True)
for name, contents in (
    ("initial_state.npz", b"initial"),
    ("normalization.npz", b"normalization"),
    ("normalization.json", b"{}\n"),
    ("sample_orders.json", b"{}\n"),
):
    (prefit / name).write_bytes(contents)
np.savez_compressed(
    prefit / "profile_inputs.npz",
    inputs=np.zeros((32, 2000, 211), dtype=np.float32),
    targets=np.zeros((32, 2000, 2), dtype=np.float32),
    row_mask=np.zeros((32, 2000), dtype=np.bool_),
)
(gate / "device_gate_reload.json").write_text(
    json.dumps({"status": "pass", "device_type": "cuda"}) + "\n"
)
(gate / "update_trace.jsonl").write_text("{}\n")
print("PROFILE_UPDATE_EVIDENCE " + json.dumps({
    "valid_for_throughput": True,
    "optimizer_attempts": 10,
    "optimizer_completed_updates": 10,
    "amp_skipped_updates": 0,
    "loss_finite": True,
    "gradient_finite": True,
}))
print("MUON_PRECISION " + json.dumps({
    "selected_orthogonalization_dtype": "float32",
    "model_precision": "fp32",
}))
"""
    monkeypatch.setattr(
        worker,
        "_device_gate_argv",
        lambda run_name, loss, manifest, validation, validation_sha: [
            sys.executable,
            "-c",
            child,
            loss,
        ],
    )
    monkeypatch.setattr(worker, "_guard", lambda sample, deadline, startup=False: None)

    inventory = [{"physical_index": 0, "name": "fixture GPU"}]
    receipts = {
        loss: worker._run_device_shape_update_gate(
            tmp_path / "manifest.json",
            tmp_path / "validation.json",
            "a" * 64,
            inventory,
            time.monotonic() + 60,
            loss,
        )
        for loss in ("wmse", "huber")
    }

    assert set(receipts) == {"wmse", "huber"}
    expected_logs = {
        "wmse": "logs/device_shape_update_gate_wmse.log",
        "huber": "logs/device_shape_update_gate_huber.log",
    }
    for loss, receipt in receipts.items():
        assert receipt["loss"] == loss
        assert receipt["log"]["path"] == expected_logs[loss]
        assert receipt["log"]["size_bytes"] > 0
        assert (worker.OUT / expected_logs[loss]).is_file()
        assert (worker.OUT / "shared" / f"device_profile_gate_{loss}.json").is_file()
    assert [row["label"] for row in worker._COMMANDS] == [
        "device_shape_update_gate_wmse",
        "device_shape_update_gate_huber",
    ]
    assert not (worker.OUT / "logs/device_shape_update_gate.log").exists()
