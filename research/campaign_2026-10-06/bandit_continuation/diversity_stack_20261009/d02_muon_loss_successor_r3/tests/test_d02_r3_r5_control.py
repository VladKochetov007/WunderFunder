from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from wnn import d02_controls  # noqa: E402

R5_CONTRACT_SHA = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"


def test_actual_r5_training_contract_and_shared_artifact_roles() -> None:
    output_raw = os.environ.get("WNN_R5_CONTROL_OUTPUT")
    if not output_raw:
        pytest.skip("set WNN_R5_CONTROL_OUTPUT to the collected R5 output root")

    output_root = Path(output_raw).resolve(strict=True)
    contract_path = output_root / "out" / "training_contract.json"
    contract, resolved = d02_controls.load_controls(contract_path, R5_CONTRACT_SHA)

    assert contract["state"] == "complete"
    assert contract["fit"]["optimizer"] == "AdamW"
    assert contract["cache"]["fit_train_count"] == 350
    assert contract["cache"]["fit_valid_count_not_used_for_fit"] == 28
    assert contract["cache"]["evaluation_valid_count"] == 468
    assert set(resolved) == set(d02_controls.ROLES)
    assert json.loads(resolved["sample_orders"].read_text())["membership_sha256"] == (
        d02_controls.MEMBERSHIP_SHA
    )
    trace = json.loads(resolved["onecycle_trace"].read_text())
    assert trace["schema"] == "wnn.loss-pair.onecycle-trace.v1"
    assert trace["total_steps"] == 220
    assert len(trace["rows"]) == 220


def test_r5_parent_role_inventory_matches_the_actual_contract() -> None:
    output_raw = os.environ.get("WNN_R5_CONTROL_OUTPUT")
    if not output_raw:
        pytest.skip("set WNN_R5_CONTROL_OUTPUT to the collected R5 output root")

    packet = json.loads((ROOT / "gpu_loss_pair.packet.json").read_text(encoding="utf-8"))
    contract_path = Path(output_raw).resolve(strict=True) / "out" / "training_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert packet["r5_control_parent"]["shared_artifacts"] == contract["artifacts"]
