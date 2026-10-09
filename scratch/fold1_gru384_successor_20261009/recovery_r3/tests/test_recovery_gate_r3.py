from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "fold1_gru384_recovery_gate_r3.py"
SPEC = importlib.util.spec_from_file_location("fold1_recovery_r3", SOURCE)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


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
