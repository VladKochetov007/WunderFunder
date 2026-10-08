from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE / "src"))

from wnn import d02_controls as controls  # noqa: E402


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Small binaries in the real producer's schema; actual exporter is checked separately."""
    control = tmp_path / "control"
    control.mkdir()
    np.savez(control / "initial_state.npz", weight=np.arange(12, dtype=np.float32).reshape(3, 4))
    np.savez(
        control / "normalization.npz", mean=np.zeros(4, np.float32), std=np.ones(4, np.float32)
    )
    np.savez(
        control / "profile_inputs.npz",
        inputs=np.ones((2, 12, 4), np.float32),
        targets=np.ones((2, 12, 2), np.float32),
        row_mask=np.ones((2, 12), bool),
    )
    orders = {
        "schema": "wnn.loss-pair.sample-orders.v1",
        "seed": 20261009,
        "batch_size": 32,
        "drop_last": False,
        "membership_sha256": controls.MEMBERSHIP_SHA,
        "epoch_seeds": {"-1": 20261009, "0": 20262018, "1": 20263027},
        "epochs": {str(epoch): [f"train:{index}" for index in range(350)] for epoch in (-1, 0, 1)},
        "evaluation_ids": [f"valid:{index}" for index in range(468)],
    }
    (control / "sample_orders.json").write_text(json.dumps(orders))
    rows = [
        {
            "attempt_index": index,
            "epoch_index": index // 110,
            "batch_index": (index % 110) // 10,
            "chunk_index": index % 10,
            "scheduler_step_pre": index,
            "optimizer_groups": [{"group_index": 0, "lr": 0.00004, "beta1": 0.95}],
        }
        for index in range(220)
    ]
    (control / "onecycle_trace.json").write_text(
        json.dumps({"schema": "wnn.loss-pair.onecycle-trace.v1", "total_steps": 220, "rows": rows})
    )
    artifacts = {}
    for role in controls.ROLES:
        path = control / (
            role
            + (".npz" if role in {"initial_state", "normalization", "profile_inputs"} else ".json")
        )
        artifacts[role] = {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": controls.digest(path),
        }
    contract = {
        "schema": "wnn.loss-pair.training-contract.v1",
        "format_version": 1,
        "state": "complete",
        "cache": {
            "manifest_sha256": controls.CACHE_SHA,
            "membership_sha256": controls.MEMBERSHIP_SHA,
        },
        "fit": {"optimizer": "AdamW"},
        "artifacts": artifacts,
    }
    path = control / "training_contract.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setenv("WNN_D02_CONTROL_CONTRACT", str(path))
    monkeypatch.setenv("WNN_D02_CONTROL_CONTRACT_SHA256", controls.digest(path))
    controls.controls_from_env.cache_clear()
    controls.control_schedule.cache_clear()
    return control


def test_exact_prefit_first_batch_and_both_group_trace(tmp_path: Path, monkeypatch) -> None:
    control = fixture(tmp_path, monkeypatch)
    actual = tmp_path / "actual"
    shutil.copytree(control, actual)
    assert controls.verify_prefit(actual)["status"] == "pass"
    controls.verify_first_batch(actual / "profile_inputs.npz")
    expected = controls.control_schedule()[0]
    row = {
        **expected,
        "optimizer_groups": [
            dict(expected["optimizer_groups"][0], group_index=index) for index in range(2)
        ],
    }
    controls.verify_schedule_row(row)
    changed = copy.deepcopy(row)
    changed["optimizer_groups"][1]["lr"] *= 3
    with pytest.raises(ValueError, match="LR/beta1"):
        controls.verify_schedule_row(changed)
    orders = json.loads((actual / "sample_orders.json").read_text())
    orders["epochs"]["0"].reverse()
    (actual / "sample_orders.json").write_text(json.dumps(orders))
    with pytest.raises(ValueError, match="sequence order"):
        controls.verify_prefit(actual)
    np.savez(actual / "initial_state.npz", weight=np.zeros((3, 4), np.float32))
    with pytest.raises(ValueError, match="tensor differs"):
        controls.verify_prefit(actual)


def test_mutated_control_bytes_fail_before_updates(tmp_path: Path, monkeypatch) -> None:
    control = fixture(tmp_path, monkeypatch)
    (control / "initial_state.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact changed"):
        controls.controls_from_env()
