from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
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


def _load_train_script(candidate: Path) -> ModuleType:
    """Load the exact source producer that emits loss-pair pre-fit artifacts."""
    sys.path.insert(0, str(candidate / "code/src"))
    path = candidate / "code/scripts/train_rnn.py"
    spec = importlib.util.spec_from_file_location("d02_r5_train_rnn", path)
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


def _assert_terminal_receipt_pin(receipt: dict[str, object], outputs: Path, relative: str) -> Path:
    """Check a captured R4 gate file against its terminal SHA/size inventory."""
    files = receipt["terminal_collection"]["files"]
    row = files[f"attempt_001/outputs/{relative}"]
    path = outputs / relative
    assert path.stat().st_size == row["size_bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    return path


def test_actual_r4_consecutive_gate_outputs_replay_huber_first_batch_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay the R4 WMSE/Huber outputs through the corrected actual producer contract."""
    import numpy as np
    import torch

    candidate = Path(__file__).resolve().parents[1]
    r4_root = candidate.parent / "d02_muon_loss_successor_r4_admitted_r2"
    receipt_path = r4_root / "GPU_D02_R4_TERMINAL_OPERATOR_RECEIPT_R1.json"
    receipt_bytes = receipt_path.read_bytes()
    assert hashlib.sha256(receipt_bytes).hexdigest() == (
        "546c267622e6968d318e6ff5d27bb1e4011e3766c3431d9f34420e9b3bafeba8"
    )
    receipt = json.loads(receipt_bytes)
    assert receipt["terminal_status"] == "KernelWorkerStatus.ERROR"
    assert receipt["terminal_collection"]["collection_complete"] is True

    outputs = r4_root / "terminal_collect_r1/attempt_001/outputs"
    phase_path = _assert_terminal_receipt_pin(receipt, outputs, "phase.jsonl")
    phases = [json.loads(line) for line in phase_path.read_text().splitlines() if line]
    gate_events = [
        (row.get("phase"), row.get("loss"))
        for row in phases
        if row.get("phase") in {
            "device_shape_update_gate_start",
            "device_shape_update_gate_pass",
        }
    ]
    assert gate_events == [
        ("device_shape_update_gate_start", "wmse"),
        ("device_shape_update_gate_pass", "wmse"),
        ("device_shape_update_gate_start", "huber"),
    ]

    wmse_profile = _assert_terminal_receipt_pin(
        receipt,
        outputs,
        "out/profile_gate/wmse/prefit/profile_inputs.npz",
    )
    huber_profile = outputs / "out/profile_gate/huber/prefit/profile_inputs.npz"
    assert not huber_profile.exists(), "the historical R4 failure must remain visible"
    for relative in (
        "initial_state.npz",
        "normalization.npz",
        "sample_orders.json",
    ):
        left = _assert_terminal_receipt_pin(
            receipt, outputs, f"out/profile_gate/wmse/prefit/{relative}"
        )
        right = _assert_terminal_receipt_pin(
            receipt, outputs, f"out/profile_gate/huber/prefit/{relative}"
        )
        if relative.endswith(".json"):
            assert json.loads(left.read_text()) == json.loads(right.read_text())
        else:
            with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
                assert set(a.files) == set(b.files)
                assert all(np.array_equal(a[name], b[name]) for name in a.files)
    wmse_schedule = _assert_terminal_receipt_pin(
        receipt, outputs, "out/profile_gate/wmse/schedule_trace.json"
    )
    huber_schedule = _assert_terminal_receipt_pin(
        receipt, outputs, "out/profile_gate/huber/schedule_trace.json"
    )
    assert json.loads(wmse_schedule.read_text()) == json.loads(huber_schedule.read_text())

    worker = _load_worker(candidate)
    trainer = _load_train_script(candidate)
    replay = tmp_path / "reconstructed-consumer-test-fixture"
    replay_wmse = replay / "wmse"
    replay_huber = replay / "huber"
    for gate_name, replay_gate in (("wmse", replay_wmse), ("huber", replay_huber)):
        source_gate = outputs / "out/profile_gate" / gate_name
        prefit = replay_gate / "prefit"
        prefit.mkdir(parents=True)
        for filename in (
            "initial_state.npz",
            "normalization.npz",
            "sample_orders.json",
        ):
            shutil.copyfile(source_gate / "prefit" / filename, prefit / filename)
        shutil.copyfile(source_gate / "schedule_trace.json", replay_gate / "schedule_trace.json")
    shutil.copyfile(wmse_profile, replay_wmse / "prefit/profile_inputs.npz")

    assert trainer.should_export_loss_pair_profile_inputs("wmse", replay_wmse, 0, 0)
    assert trainer.should_export_loss_pair_profile_inputs("huber", replay_huber, 0, 0)
    assert not trainer.should_export_loss_pair_profile_inputs("huber", replay_huber, 1, 0)
    assert not trainer.should_export_loss_pair_profile_inputs("huber", None, 0, 0)
    monkeypatch.delenv("WNN_D02_CONTROL_CONTRACT", raising=False)
    with np.load(wmse_profile, allow_pickle=False) as arrays:
        trainer.export_loss_pair_profile_inputs(
            torch.from_numpy(arrays["inputs"]),
            torch.from_numpy(arrays["targets"]),
            torch.from_numpy(arrays["row_mask"]),
            replay_huber,
        )

    # This is a reconstructed expected Huber artifact from the verified shared first
    # batch, not an observation that the failed R4 Huber process wrote such a file.
    assert (replay_huber / "prefit/profile_inputs.npz").is_file()
    worker._verify_prefit_pair(
        {"wmse": replay_wmse, "huber_d1": replay_huber}, expected_steps=10
    )
