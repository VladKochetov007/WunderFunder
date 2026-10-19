"""Observe all existing execution owners and reserve their GPU slots through export."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from feature_common import read, sha
from panel import GPU_UUIDS

OLD_UNITS = ("tabdata-r01-execution-v4", "tabdata-r01-density4-v1", "tabdata-pr15-execution-v2")
OWN_UNIT = "tabdata-native-feature-execution-v1"
MEMORY_MAX = 192 * 1024**3
# Identities copied from the retained density4 live-status-v2 receipt
# (SHA-256 5c1de14f...13e02500), not a reusable PID allowlist.
PRESERVED_FOREIGN_RECEIPT_SHA256 = (
    "5c1de14fdda31eca810fecdcb63d7858d0d5db58851a7ea86c558105413e02500"
)
PRESERVED_FOREIGN = {
    4060404: (GPU_UUIDS[0], 207469348),
    4086591: (GPU_UUIDS[2], 208123403),
    4061037: (GPU_UUIDS[4], 207474461),
}


def process_start_ticks(pid: int, proc_root: Path = Path("/proc")) -> int:
    """Read Linux stat field 22, handling spaces and parentheses in comm."""
    stat = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    closing = stat.rfind(")")
    if closing < 0:
        raise ValueError("Malformed Linux process stat record")
    fields = stat[closing + 2 :].split()
    if len(fields) <= 19:
        raise ValueError("Truncated Linux process stat record")
    return int(fields[19])


def is_preserved_foreign(pid: int, gpu_uuid: str) -> bool:
    """Exempt only the recorded external process incarnation on its recorded GPU."""
    expected = PRESERVED_FOREIGN.get(pid)
    if expected is None or expected[0] != gpu_uuid:
        return False
    try:
        return process_start_ticks(pid) == expected[1]
    except (OSError, ValueError):
        return False


def validate_device_map(devices: dict[str, dict]) -> None:
    """Require the complete panel GPU UUID/index map before making reservations."""
    if set(devices) != set(GPU_UUIDS) or any(
        devices[uuid].get("index") != index for index, uuid in enumerate(GPU_UUIDS)
    ):
        raise ValueError("Observed GPU UUID/index map differs from the frozen eight-card panel")


def unit_status(name: str) -> dict:
    """Read one user-unit's actual kernel accounting path and hard limits."""
    args = [
        "systemctl",
        "--user",
        "show",
        name,
        "-p",
        "ActiveState",
        "-p",
        "SubState",
        "-p",
        "Result",
        "-p",
        "MemoryCurrent",
        "-p",
        "MemoryMax",
        "-p",
        "MemorySwapMax",
        "-p",
        "TasksMax",
        "-p",
        "ControlGroup",
    ]
    text = subprocess.check_output(args, text=True, timeout=15)
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def cgroup_members(unit: dict, cgroup_root: Path = Path("/sys/fs/cgroup")) -> list[int]:
    """Treat a collected inactive cgroup as absent, not as observed zero resource caps."""
    relative = unit.get("ControlGroup", "")
    if not relative:
        if unit.get("ActiveState") == "active":
            raise ValueError("Active execution owner has no cgroup identity")
        return []
    base = cgroup_root / relative.lstrip("/")
    if not base.exists():
        if unit.get("ActiveState") == "active":
            raise ValueError("Active execution owner has no process accounting")
        return []
    members = set()
    for path in base.rglob("cgroup.procs"):
        members.update(int(value) for value in path.read_text().split())
    return sorted(members)


def cuda_actors() -> tuple[dict[str, list[int]], dict[str, dict]]:
    """Read device memory and live CUDA PIDs without importing Torch or changing a job."""
    devices = {}
    observed_indices = set()
    for line in subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    ).splitlines():
        index, uuid, used, total = [x.strip() for x in line.split(",")]
        index = int(index)
        if uuid in devices or index in observed_indices:
            raise ValueError("nvidia-smi returned duplicate GPU identity")
        observed_indices.add(index)
        devices[uuid] = dict(index=int(index), used_mib=int(used), total_mib=int(total))
    validate_device_map(devices)
    if observed_indices != set(range(len(GPU_UUIDS))):
        raise ValueError("Observed GPU indices differ from the frozen eight-card panel")
    actors = {uuid: [] for uuid in devices}
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=20,
    ).splitlines():
        uuid, pid = [x.strip() for x in line.split(",")]
        if uuid not in actors:
            raise ValueError("CUDA process row references a GPU outside the frozen panel")
        actors[uuid].append(int(pid))
    return actors, devices


def controller_terminal(root: Path, family: str, unit: dict) -> bool:
    """Free a legacy reservation only after full durable completion and an empty owner cgroup."""
    if unit.get("ActiveState") in {"active", "activating", "deactivating"} or cgroup_members(unit):
        return False
    if family == "r01":
        original = unit_status(OLD_UNITS[0])
        if original.get("ActiveState") in {
            "active",
            "activating",
            "deactivating",
        } or cgroup_members(original):
            return False
        state = root / "state/r01-density4-v1"
        p = state / "SUMMARY.json"
        if not p.is_file():
            return False
        summary = read(p)
        if (
            summary.get("schema") != "tabdata-r01-density4-successor-summary.v1"
            or summary.get("status") != "complete"
            or summary.get("export_count_verified") != 60
            or summary.get("grade_pair_count") != 30
            or summary.get("pending_fit_count") != 0
            or summary.get("old_unit_retirement_eligible") is not True
        ):
            return False
        tail = (state / "events.jsonl").read_text().splitlines()[-5:]
        return any(
            json.loads(line).get("event") == "controller_finished"
            and json.loads(line).get("summary_sha256") == sha(p)
            for line in tail
        )
    state = root / "state/pr15-execution-v2"
    p = state / "COMPLETE.json"
    if not p.is_file() or read(p) != {
        "completed_successor_cells": 104,
        "preserved_original_fits": 16,
    }:
        return False
    results = read(state / "RESULTS.json")
    expected = {
        f"pr15_b300_v1_{arm}_f{fold}_s{seed}"
        for arm in ("baseline", "moments", "dictionary", "combined")
        for fold in range(4)
        for seed in range(1002, 1009 if fold < 2 else 1008)
    }
    return set(results) == expected and all(
        row.get("status") == "completed" for row in results.values()
    )


def snapshot(root: Path, *, require_own: bool = True) -> dict:
    """Account for all four execution units and leave two legacy slots per active family."""
    units = {name: unit_status(name) for name in (*OLD_UNITS, OWN_UNIT)}
    terminal = {
        "r01": controller_terminal(root, "r01", units[OLD_UNITS[1]]),
        "pr15": controller_terminal(root, "pr15", units[OLD_UNITS[2]]),
    }
    errors = []
    current_bytes = 0
    for name, unit in units.items():
        optional = (name in OLD_UNITS[:2] and terminal["r01"]) or (
            name == OLD_UNITS[2] and terminal["pr15"]
        )
        if name == OWN_UNIT and not require_own:
            continue
        if optional:
            continue
        expected_tasks = 1024 if name == OWN_UNIT else 512
        if (
            unit.get("ActiveState") != "active"
            or unit.get("MemoryMax") != str(MEMORY_MAX)
            or unit.get("MemorySwapMax") != "0"
            or unit.get("TasksMax") != str(expected_tasks)
        ):
            errors.append("Execution unit state/caps differ: " + name)
        value = unit.get("MemoryCurrent", "")
        if value.isdigit():
            current_bytes += int(value)
        else:
            errors.append("Missing memory accounting: " + name)
        unit["observed_members"] = cgroup_members(unit)
        relative = unit.get("ControlGroup", "")
        base = Path("/sys/fs/cgroup") / relative.lstrip("/")
        p = base / "memory.events"
        if p.is_file():
            events = dict(line.split() for line in p.read_text().splitlines())
            unit["observed_memory_events"] = events
            if any(int(events.get(k, 0)) for k in ("oom", "oom_kill", "oom_group_kill")):
                errors.append("Observed unit OOM: " + name)
            if not {"oom", "oom_kill"}.issubset(events):
                errors.append("Incomplete OOM counters: " + name)
        else:
            errors.append("Missing memory events: " + name)
        ancestors = []
        for parent in (base, *base.parents):
            if parent == Path("/sys/fs/cgroup"):
                break
            values = {
                key: (parent / key).read_text().strip()
                for key in ("memory.max", "memory.swap.max", "pids.max")
                if (parent / key).is_file()
            }
            if len(values) != 3:
                errors.append("Missing ancestor accounting: " + name)
            if values.get("memory.max", "max") != "max" and int(values["memory.max"]) < MEMORY_MAX:
                errors.append("Ancestor memory below declared owner cap: " + name)
            ancestors.append(dict(path=str(parent), limits=values))
        unit["observed_ancestor_limits"] = ancestors
    meminfo = dict(
        (line.split(":", 1)[0], line.split(":", 1)[1].strip())
        for line in Path("/proc/meminfo").read_text().splitlines()
    )
    available = int(meminfo["MemAvailable"].split()[0]) * 1024
    free = shutil.disk_usage(root).free
    if min(available, free) < 64 * 1024**3:
        errors.append("Host memory or physical-disk reserve below64GiB")
    actors, devices = cuda_actors()
    owned = set().union(*(set(unit.get("observed_members", ())) for unit in units.values()))
    preserved_by_gpu = {
        uuid: {pid for pid in pids if is_preserved_foreign(pid, uuid)}
        for uuid, pids in actors.items()
    }
    for uuid, pids in actors.items():
        preserved = preserved_by_gpu[uuid]
        unexpected = set(pids) - owned - preserved
        if unexpected:
            errors.append("Unaccounted CUDA actor on " + uuid)
        campaign = set(pids) - preserved
        if len(campaign) > 5:
            errors.append("Observed campaign density exceeds five on " + uuid)
    legacy_cuda = {"r01": [], "pr15": []}
    for pids in actors.values():
        for pid in pids:
            try:
                command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if str(root / "runtime/b300-r01-v1/code/gpu_entrypoint.py").encode() in command:
                legacy_cuda["r01"].append(pid)
            if str(root / "state/pr15-execution-").encode() in command:
                legacy_cuda["pr15"].append(pid)
    for family in terminal:
        if terminal[family] and legacy_cuda[family]:
            terminal[family] = False
            errors.append("Terminal owner still has a CUDA actor: " + family)
    for device in devices.values():
        if device["total_mib"] - device["used_mib"] < 12 * 1024:
            errors.append("Insufficient device memory reserve")
    return dict(
        schema="tabdata-four-owner-resource-snapshot.v1",
        time_unix=time.time(),
        verified=not errors,
        errors=errors,
        units=units,
        terminal=terminal,
        reserved_legacy_slots_per_gpu=2 * (not terminal["r01"]) + 2 * (not terminal["pr15"]),
        cgroup_memory_current_total=current_bytes,
        host_available_bytes=available,
        disk_free_bytes=free,
        cuda_actors=actors,
        preserved_foreign_identities_sha256=PRESERVED_FOREIGN_RECEIPT_SHA256,
        preserved_foreign_pids_by_gpu={
            uuid: sorted(pids) for uuid, pids in preserved_by_gpu.items() if pids
        },
        legacy_cuda=legacy_cuda,
        devices=devices,
    )
