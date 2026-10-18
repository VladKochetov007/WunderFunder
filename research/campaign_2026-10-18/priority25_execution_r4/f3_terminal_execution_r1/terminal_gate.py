from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

PLAN_SHA256 = "b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8"
FOLD_SHA256 = "21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f"
STAGE1 = {
    "wide": {
        "collection": "2af09d13ea5d36c90f84c52e1b6e3b3606f6af143f27280bd1e39d5b29e7cfdd",
        "verification": "2a274447676b657f30280631e30a1ff3a67acdec02497b2355486d84a789aeb4",
    },
    "deep": {
        "collection": "99595b13eaa0f7fc7dac54ac74e09f6b55551e4085219d6dba7e05a95af5266b",
        "verification": "223725c151ec40f6f26e71332b1cf1b0f6fbe26a2c749ce6421f0b49d06a3efa",
    },
}
CONTROLLER = "computer-vision-homework-r01-f3-sequence.service"


class GateError(ValueError):
    """A live terminal-support prerequisite is absent or inconsistent."""


def require(condition: bool, label: str) -> None:
    """Raise a named failure instead of relaxing a root gate."""
    if not condition:
        raise GateError(label)


def sha256_bytes(raw: bytes) -> str:
    """Hash one already bounded byte string."""
    return hashlib.sha256(raw).hexdigest()


def validate_pointer(
    plan: Mapping[str, object], sequence: Mapping[str, object], arm: str
) -> dict[str, object] | None:
    """Require the real sequence to be past this arm's successful native fit."""
    require(arm in ("wide", "deep"), "unknown terminal arm")
    require(sequence.get("plan_sha256") == PLAN_SHA256, "sequence plan SHA")
    steps = cast(Sequence[Mapping[str, object]], plan.get("steps"))
    require(len(steps) == 7, "plan must have seven steps")
    native_index = 2 if arm == "wide" else 3
    native = steps[native_index]
    require(native.get("step_id") == f"continuation_{arm}", "native continuation identity")
    require(native.get("kind") == "continue_fixed_six_epoch_fit", "native continuation kind")
    status = sequence.get("status")
    index = sequence.get("index")
    require(type(index) is int, "sequence index is not an integer")
    if status == "finished":
        require(index == 6, "finished sequence is not at terminal grade index")
        require(sequence.get("attempt") == steps[6].get("attempt"), "finished sequence grade id")
        return None
    require(status == "running", "sequence is neither running nor finished")
    minimum_index = native_index + 1
    allowed = range(minimum_index, 7)
    require(index in allowed, "sequence has not advanced past native terminal fit")
    current = steps[index]
    require(sequence.get("attempt") == current.get("attempt"), "sequence attempt/plan mismatch")
    expected_kinds = {
        3: ("continuation_deep", "continue_fixed_six_epoch_fit"),
        4: ("export_wide", "export_fixed_epoch5"),
        5: ("export_deep", "export_fixed_epoch5"),
        6: ("grade", "paired_official_grade"),
    }
    require(
        (current.get("step_id"), current.get("kind")) == expected_kinds[index],
        "current step identity",
    )
    return dict(current)


def validate_successful_exit(record: Mapping[str, object], expected_id: str) -> None:
    """Require native attempt evidence to report the exact successful exit."""
    require(record.get("id") == expected_id, "native attempt id differs")
    require(record.get("status") == "finished", "native attempt is not finished")
    require(type(record.get("exit_status")) is int, "native exit status is not an integer")
    require(record.get("exit_status") == 0, "native attempt exit status is nonzero")


def expected_children(current: Mapping[str, object] | None) -> dict[str, tuple[float, int]]:
    """Return exact populated service roots and their hierarchical caps."""
    if current is None:
        return {}
    kind = current.get("kind")
    worker_memory = 4_000_000_000 if kind == "paired_official_grade" else 8_000_000_000
    attempt = current.get("attempt")
    require(isinstance(attempt, str) and attempt, "current step attempt missing")
    return {
        CONTROLLER: (0.1, 512_000_000),
        f"wnn-bandit-{attempt}.service": (1.0, worker_memory),
    }


def validate_resource_snapshot(
    snapshot: Mapping[str, object],
    current: Mapping[str, object] | None,
    operation: str,
) -> dict[str, object]:
    """Count every populated descendant under known units and reserve support."""
    require(operation in ("collector", "verifier"), "unknown support operation")
    props = cast(Mapping[str, str], snapshot.get("parent_properties"))
    require(props.get("CPUQuotaPerSecUSec") == "2s", "parent CPU quota")
    require(props.get("MemoryHigh") == "18000000000", "parent memory.high")
    require(props.get("MemoryMax") == "20000000000", "parent memory.max")
    require(props.get("MemorySwapMax") == "0", "parent swap cap")
    require(snapshot.get("parent_cgroup_procs") == "", "parent cgroup has direct processes")
    direct = cast(Sequence[Mapping[str, object]], snapshot.get("top_units"))
    actual = {str(child.get("unit")): child for child in direct}
    expected = expected_children(current)
    if current is None:
        # A finished controller can be in its short post-write tail, or its
        # systemd unit can already be empty. Both are known states; a worker
        # or any other populated child remains a hard failure.
        require(
            set(actual) in (set(), {CONTROLLER}),
            "finished sequence has an unexpected populated direct child",
        )
        if CONTROLLER in actual:
            expected = {CONTROLLER: (0.1, 512_000_000)}
    else:
        require(set(actual) == set(expected), "populated direct-child set differs")
    for unit, (cpu, memory) in expected.items():
        child = actual[unit]
        require(abs(float(child.get("cpu_cores", -1)) - cpu) < 0.001, f"{unit} CPU cap")
        require(int(child.get("memory_max", -1)) == memory, f"{unit} memory cap")
        require(int(child.get("swap_max", -1)) == 0, f"{unit} swap cap")
        require(child.get("populated") is True, f"{unit} is not populated")
    populated_paths = cast(Sequence[str], snapshot.get("populated_paths"))
    allowed_roots = set(expected)
    for relative in populated_paths:
        parts = Path(relative).parts
        require(parts and parts[0] in allowed_roots, f"unknown populated descendant {relative}")
    if expected:
        require(
            {Path(path).parts[0] for path in populated_paths} == allowed_roots,
            "descendant root set differs",
        )
    else:
        require(not populated_paths, "finished sequence has populated descendants")
    support_cpu, support_memory = (
        (0.25, 512_000_000) if operation == "collector" else (0.5, 1_500_000_000)
    )
    parent_current = int(props.get("MemoryCurrent", "-1"))
    children_current = sum(int(child.get("memory_current", -1)) for child in direct)
    require(parent_current >= 0 and children_current >= 0, "invalid memory.current")
    unattributed = max(0, parent_current - children_current)
    committed_memory = (
        unattributed + sum(memory for _, memory in expected.values()) + support_memory
    )
    committed_cpu = sum(cpu for cpu, _ in expected.values()) + support_cpu
    require(committed_memory < 17_000_000_000, "committed memory reaches 17 GB")
    require(committed_cpu <= 2.0, "aggregate CPU commitments exceed two cores")
    return {
        "populated_paths": list(populated_paths),
        "known_top_units": sorted(actual),
        "committed_cpu_cores": committed_cpu,
        "committed_memory_bytes": committed_memory,
        "support_cpu_cores": support_cpu,
        "support_memory_bytes": support_memory,
        "hierarchical_descendants_counted_once_at_unit_cap": True,
    }


def decode_raw_files(raw_files: Mapping[str, object]) -> dict[str, bytes]:
    """Decode and hash-check a bounded remote capture before any local write."""
    require(1 <= len(raw_files) <= 8, "raw capture file count")
    decoded: dict[str, bytes] = {}
    total = 0
    for name, item in raw_files.items():
        require(Path(name).name == name, "raw file name must be a basename")
        record = cast(Mapping[str, object], item)
        raw = base64.b64decode(str(record.get("base64", "")), validate=True)
        require(len(raw) == int(record.get("bytes", -1)), f"{name} byte length")
        require(sha256_bytes(raw) == record.get("sha256"), f"{name} SHA256")
        require(len(raw) <= 2_000_000, f"{name} exceeds per-file bound")
        json.loads(raw)
        total += len(raw)
        decoded[name] = raw
    require(total <= 8_000_000, "raw capture exceeds total bound")
    return decoded


def write_exclusive(path: Path, raw: bytes) -> None:
    """Create evidence once and leave collisions untouched."""
    with path.open("xb") as stream:
        stream.write(raw)
