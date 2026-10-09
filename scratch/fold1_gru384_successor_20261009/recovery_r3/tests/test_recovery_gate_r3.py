from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "fold1_gru384_recovery_gate_r3.py"
SPEC = importlib.util.spec_from_file_location("fold1_recovery_r3", SOURCE)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)
ROOT = Path(__file__).resolve().parents[4]
RESOURCE_SPEC = importlib.util.spec_from_file_location(
    "bandit_resources_source_manifest", ROOT / "scripts" / "bandit_resources.py"
)
assert RESOURCE_SPEC is not None and RESOURCE_SPEC.loader is not None
RESOURCE = importlib.util.module_from_spec(RESOURCE_SPEC)
RESOURCE_SPEC.loader.exec_module(RESOURCE)


def command_fixture() -> tuple[dict[str, object], dict[str, object]]:
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=20s",
        "14400",
        "/project/.venv/bin/python",
        "scripts/train_rnn.py",
        "--name",
        "bd26_w384raw_s3_f1_resume_r2",
        "--epochs",
        "4",
        "--raw-only",
        "--save-training-state",
        "--resume-state",
        "/project/artifacts/runs/bd26_w384raw_s3_f1/training_state.pt",
    ]
    request = {"resume_command": command}
    packet = {
        "root": "/project",
        "r2": {
            "output_path": "artifacts/runs/bd26_w384raw_s3_f1_resume_r2",
            "source_training_state_path": "artifacts/runs/bd26_w384raw_s3_f1/training_state.pt",
        },
        "resume": {
            "timeout_seconds": 7200,
            "run_name": "bd26_w384raw_s3_f1_resume_r3",
            "resume_from_path": "artifacts/runs/bd26_w384raw_s3_f1_resume_r2/training_state.pt",
        },
    }
    return request, packet


def test_r3_command_changes_only_wrapper_name_and_resume_checkpoint() -> None:
    request, packet = command_fixture()
    result = GATE.resume_command(request, packet)
    assert result[3] == "7200"
    assert result[result.index("--name") + 1] == "bd26_w384raw_s3_f1_resume_r3"
    assert result[result.index("--epochs") + 1] == "4"
    assert result[result.index("--resume-state") + 1] == (
        "/project/artifacts/runs/bd26_w384raw_s3_f1_resume_r2/training_state.pt"
    )
    before = request["resume_command"]
    assert isinstance(before, list)
    allowed = {3, before.index("--name") + 1, before.index("--resume-state") + 1}
    assert [
        index for index, (a, b) in enumerate(zip(before, result, strict=True)) if a != b
    ] == sorted(allowed)


def test_r3_command_fails_closed_when_schedule_horizon_changes() -> None:
    request, packet = command_fixture()
    command = list(request["resume_command"])
    command[command.index("--epochs") + 1] = "3"
    request["resume_command"] = command
    with pytest.raises(ValueError, match="four-epoch schedule horizon"):
        GATE.resume_command(request, packet)


def test_missing_or_nonzero_oom_delta_fails_closed() -> None:
    names = (
        "memory_event_deltas",
        "final_memory_event_deltas",
        "parent_memory_event_deltas",
        "final_parent_memory_event_deltas",
        "parent_hierarchical_memory_event_deltas",
        "final_parent_hierarchical_memory_event_deltas",
    )
    record = {name: {"oom": 0, "oom_kill": 0, "oom_group_kill": 0} for name in names}
    GATE.require_zero_oom_deltas(record)
    record["final_parent_hierarchical_memory_event_deltas"]["oom_kill"] = 1
    with pytest.raises(ValueError, match="missing/nonzero OOM evidence"):
        GATE.require_zero_oom_deltas(record)
    del record["parent_memory_event_deltas"]
    with pytest.raises(ValueError, match="missing/nonzero OOM evidence"):
        GATE.require_zero_oom_deltas(record)


def test_controller_attempt_receipt_uses_exact_source_manifest_subset() -> None:
    packet_path = SOURCE.with_name("RECOVERY_GATE_PACKET_R3.json")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    attempt_paths, loader_paths = GATE.source_pin_partition(packet)
    assert len(attempt_paths) == 6
    assert loader_paths == [
        "research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/"
        "sol_continuation/FOLD1_DEPENDENCY_INVENTORY_R1.json"
    ]
    # Use the production controller's manifest builder on the frozen command.
    # It captures source trees, command-path files, and the attempt contract;
    # the packet-loader inventory JSON is deliberately outside that receipt.
    request_path = SOURCE.with_name("RESUME_REQUEST_R3.json")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source_receipt = RESOURCE.source_manifest(ROOT, request["resume_command"], request_path)
    assert loader_paths[0] not in source_receipt
    GATE.verify_packet_source_pins(ROOT, packet)
    GATE.verify_attempt_source_receipts({"source_sha256": source_receipt}, packet)

    source_receipt["src/wnn/optim.py"] = "e" * 64
    with pytest.raises(ValueError, match="attempt source receipt differs"):
        GATE.verify_attempt_source_receipts({"source_sha256": source_receipt}, packet)

    invalid_packet = dict(packet)
    invalid_packet["attempt_source_receipt_paths"] = attempt_paths + loader_paths
    invalid_packet["packet_loader_only_source_paths"] = []
    with pytest.raises(ValueError, match="only the exact dependency inventory"):
        GATE.source_pin_partition(invalid_packet)
