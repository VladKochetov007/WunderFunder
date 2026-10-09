"""Audit real completed-run settings after systemd unloads a transient unit."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "terminal_audit_r4", BASE / "fold1_gru384_terminal_audit_r4.py"
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def actual_inputs() -> tuple[dict, dict, dict]:
    packet = json.loads((BASE / "RECOVERY_GATE_PACKET_R3.json").read_text())
    root = Path(packet["root"])
    attempt = json.loads(
        (
            root / packet["runtime_relative"] / "attempts" / "fold1-gru384-s3-resume-r3.json"
        ).read_text()
    )
    effective = json.loads((BASE / "ROOT_R3_EFFECTIVE_UNIT_PROPERTIES_R1.json").read_text())
    return attempt, packet, effective


def test_actual_terminal_run_retains_resource_evidence() -> None:
    attempt, packet, effective = actual_inputs()
    assert attempt["status"] == "finished"
    assert attempt["exit_status"] == 0
    assert (
        AUDIT.recorded_resource_properties(attempt, packet, effective)["MemoryHigh"] == "8000000000"
    )


@pytest.mark.parametrize("field", ["MemoryMax", "MemorySwapMax", "CPUQuotaPerSecUSec"])
def test_wrong_start_settings_are_rejected(field: str) -> None:
    attempt, packet, effective = actual_inputs()
    changed = copy.deepcopy(attempt)
    changed["attempt_properties_at_start"][field] = "infinity"
    with pytest.raises(ValueError, match="recorded R3 resource properties"):
        AUDIT.recorded_resource_properties(changed, packet, effective)


def test_missing_effective_high_observation_is_rejected() -> None:
    attempt, packet, effective = actual_inputs()
    del effective["MemoryHigh"]
    with pytest.raises(ValueError, match="root observed effective"):
        AUDIT.recorded_resource_properties(attempt, packet, effective)


def test_wrong_cgroup_is_rejected() -> None:
    attempt, packet, effective = actual_inputs()
    attempt["attempt_properties_at_start"]["ControlGroup"] = "/other-job"
    with pytest.raises(ValueError, match="cgroup identity"):
        AUDIT.recorded_resource_properties(attempt, packet, effective)
