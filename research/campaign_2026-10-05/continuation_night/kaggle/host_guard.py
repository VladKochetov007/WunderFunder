from __future__ import annotations

import ctypes
import json
import logging
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

ProcessRecord = dict[str, int | str]


def cgroup_ancestors(root: Path, relative: str) -> list[Path]:
    """Include every mounted ancestor that can constrain the current process."""
    leaf = root / relative.lstrip("/")
    if ".." in leaf.parts:
        raise ValueError("unexpected parent traversal in the kernel cgroup path")
    result = []
    while leaf != root:
        result.append(leaf)
        leaf = leaf.parent
    return [*result, root]


def sample_v1_hierarchy(
    ancestors: list[Path], capacity: int, available: int
) -> tuple[int, int, dict[str, dict[str, object]]]:
    """Apply every visible v1 memory limit, failing closed on a missing level."""
    records: dict[str, dict[str, object]] = {}
    for path in dict.fromkeys(ancestors):
        try:
            limit = int((path / "memory.limit_in_bytes").read_text().strip())
            current = int((path / "memory.usage_in_bytes").read_text().strip())
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"detected cgroup v1 memory hierarchy is not fully observable at {path}"
            ) from error
        if limit <= 0 or current < 0:
            raise RuntimeError(f"invalid cgroup v1 memory values at {path}")
        capacity = min(capacity, limit)
        available = min(available, limit - current)
        record: dict[str, object] = {"current_bytes": current, "limit": limit}
        for name in ("memory.stat", "memory.oom_control"):
            stat_path = path / name
            if stat_path.is_file():
                record[name] = {
                    key: int(value)
                    for key, value in (
                        line.split() for line in stat_path.read_text().splitlines()
                    )
                }
        records[str(path)] = record
    if not records:
        raise RuntimeError("detected cgroup v1 memory hierarchy has no visible ancestors")
    return capacity, available, records


def write_json(path: Path, value: object) -> None:
    """Publish a complete telemetry record atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def host_sample() -> dict[str, object]:
    """Measure physical and cgroup pressure; retain anon/file/OOM diagnostics."""
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        memory[name] = int(value.strip().split()[0]) * 1024
    capacity = memory["MemTotal"]
    available = memory["MemAvailable"]
    cgroup: dict[str, object] = {}
    # Intermediate limits constrain descendants even when the leaf says max.
    bases = [Path("/sys/fs/cgroup")]
    v1_root = Path("/sys/fs/cgroup/memory")
    v1_bases = [v1_root]
    v1_detected = False
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            bases.extend(cgroup_ancestors(Path("/sys/fs/cgroup"), relative))
        if "memory" in controllers.split(","):
            v1_detected = True
            v1_bases.extend(cgroup_ancestors(v1_root, relative))
    for base in dict.fromkeys(bases):
        limit = base / "memory.max"
        if not limit.is_file():
            continue
        raw_limit = limit.read_text().strip()
        current = int((base / "memory.current").read_text())
        if raw_limit != "max":
            capacity = min(capacity, int(raw_limit))
            available = min(available, int(raw_limit) - current)
        record: dict[str, object] = {"current_bytes": current, "limit": raw_limit}
        for name in ("memory.stat", "memory.events"):
            path = base / name
            if path.exists():
                record[name] = {
                    key: int(value)
                    for key, value in (line.split() for line in path.read_text().splitlines())
                }
        cgroup[str(base)] = record
    if v1_detected:
        capacity, available, v1_records = sample_v1_hierarchy(
            v1_bases, capacity, available
        )
        cgroup.update(v1_records)
    return {
        "capacity_bytes": capacity,
        "available_bytes": max(0, min(capacity, available)),
        "physical_available_bytes": memory["MemAvailable"],
        "cgroups": cgroup,
    }


def tree_sample(root_pid: int) -> list[ProcessRecord]:
    """Inventory only the launched process and its live descendants."""
    pending, rows = [root_pid], []
    while pending:
        pid = pending.pop()
        try:
            raw = Path(f"/proc/{pid}/stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text()
            rows.append(
                {
                    "pid": pid,
                    "start_ticks": int(fields[19]),
                    "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
                    "group": int(fields[2]),
                    "state": fields[0],
                }
            )
            pending.extend(int(child) for child in children.split())
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rows


def direct_children() -> list[int]:
    """Read this dedicated supervisor's immediate and newly adopted children."""
    pid = os.getpid()
    return [int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]


def set_subreaper(value: int) -> int:
    """Contain double-forked workers even if an intermediate parent exits."""
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER")
    if libc.prctl(36, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
    return previous.value


def owned_children(root_pid: int, excluded: set[int]) -> list[ProcessRecord]:
    """Include the root tree and adopted descendants absent before launch."""
    rows = tree_sample(root_pid)
    for pid in direct_children():
        if pid not in excluded and pid != root_pid:
            rows.extend(tree_sample(pid))
    return list({int(row["pid"]): row for row in rows}.values())


def gpu_sample() -> list[dict[str, object]]:
    """Observe physical VRAM outside the framework allocator."""
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    devices = []
    for line in text.strip().splitlines():
        uuid, name, total, used = (part.strip() for part in line.split(","))
        devices.append(
            {
                "uuid": uuid,
                "name": name,
                "total_bytes": int(total) * 1024**2,
                "used_bytes": int(used) * 1024**2,
            }
        )
    return devices


def oom_counts(sample: dict[str, object]) -> dict[str, int]:
    """Read available cgroup OOM counters without treating absence as zero."""
    return {
        f"{path}/{source}/{key}": count
        for path, record in sample["cgroups"].items()
        for source in ("memory.events", "memory.oom_control")
        for key, count in record.get(source, {}).items()
        if key in {"oom", "oom_kill", "oom_group_kill"}
    }


def stop_tree(
    process: subprocess.Popen[bytes], rows: list[ProcessRecord], excluded: set[int]
) -> dict[str, object]:
    """Terminate the captured descendants, including nested process sessions."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        rows = list(
            {
                int(row["pid"]): row for row in [*rows, *owned_children(process.pid, excluded)]
            }.values()
        )
        groups = set()
        for row in reversed(rows):
            current = tree_sample(int(row["pid"]))
            if current and current[0]["start_ticks"] == row["start_ticks"]:
                group = int(current[0]["group"])
                if group != os.getpgrp():
                    groups.add(group)
                with suppress(ProcessLookupError):
                    os.kill(int(row["pid"]), sig)
        for group in groups:
            with suppress(ProcessLookupError):
                os.killpg(group, sig)
        if sig == signal.SIGTERM:
            time.sleep(2)
    process.wait(timeout=10)
    deadline = time.monotonic() + 5
    while True:
        remaining = owned_children(process.pid, excluded)
        for row in remaining:
            with suppress(ChildProcessError):
                os.waitpid(int(row["pid"]), os.WNOHANG)
        remaining = owned_children(process.pid, excluded)
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    return {
        "status": "complete" if not remaining else "failed",
        "remaining": remaining,
        "captured_processes": rows,
        "root_returncode": process.returncode,
    }


def supervised(
    argv: list[str],
    cwd: Path,
    output: Path,
    environment: dict[str, str],
    *,
    timeout: int,
    rss_limit: int,
    reserve: int = 2 * 1024**3,
    gpu_uuid: str | None = None,
    gpu_stop: int = 13 * 1024**3,
) -> dict[str, object]:
    """Run one child with host pressure stops and durable startup breadcrumbs."""
    output.mkdir(parents=True, exist_ok=False)
    initial = host_sample()
    write_json(output / "startup.json", {"argv": argv, "host": initial, "time": time.time()})
    capacity = int(initial["capacity_bytes"])
    budget = min(int(capacity * 0.70), int(initial["available_bytes"]) - int(capacity * 0.30))
    if rss_limit + reserve > budget:
        write_json(
            output / "result.json",
            {
                "status": "rejected_before_launch",
                "reason": "host_target_admission",
                "host": initial,
                "required_bytes": rss_limit + reserve,
                "admissible_bytes": budget,
            },
        )
        raise RuntimeError("host target admission failed before child launch")
    started = time.monotonic()
    result: dict[str, object] = {"status": "running", "peak_rss_bytes": 0}
    process = None
    rows: list[ProcessRecord] = []
    seen: dict[int, ProcessRecord] = {}
    last_logged = -60.0
    initial_oom = oom_counts(initial)
    excluded = set(direct_children())
    previous_subreaper = set_subreaper(1)
    try:
        with (output / "worker.log").open("wb") as log:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                rows = owned_children(process.pid, excluded)
                seen.update((int(row["pid"]), row) for row in rows)
                rss = sum(int(row["rss_bytes"]) for row in rows)
                host = host_sample()
                elapsed = time.monotonic() - started
                result["peak_rss_bytes"] = max(int(result["peak_rss_bytes"]), rss)
                sample = {
                    "time": time.time(),
                    "elapsed": elapsed,
                    "host": host,
                    "processes": rows,
                    "rss_bytes": rss,
                    "returncode": process.poll(),
                }
                devices = gpu_sample() if gpu_uuid else []
                sample["devices"] = devices
                if devices:
                    observed = [
                        int(device["used_bytes"])
                        for device in devices
                        if device["uuid"] == gpu_uuid
                    ]
                    result["peak_gpu_bytes"] = max(
                        [int(result.get("peak_gpu_bytes", 0)), *observed]
                    )
                write_json(output / "heartbeat.json", sample)
                with (output / "telemetry.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample, sort_keys=True) + "\n")
                if elapsed - last_logged >= 60:
                    logging.info(
                        json.dumps(
                            {
                                "event": "transport_heartbeat",
                                "elapsed": elapsed,
                                "rss_bytes": rss,
                                "host_available_bytes": host["available_bytes"],
                            }
                        )
                    )
                    last_logged = elapsed
                reason = None
                if rss >= rss_limit:
                    reason = "process_tree_rss"
                if int(host["available_bytes"]) <= int(host["capacity_bytes"]) * 0.15:
                    reason = "host_85_percent_stop"
                if elapsed >= timeout:
                    reason = "timeout"
                if any(
                    count > initial_oom.get(key, count) for key, count in oom_counts(host).items()
                ):
                    reason = "cgroup_oom_counter_increased"
                if gpu_uuid:
                    selected = [device for device in devices if device["uuid"] == gpu_uuid]
                    if len(selected) != 1:
                        reason = "selected_gpu_disappeared"
                    elif int(selected[0]["used_bytes"]) >= min(
                        gpu_stop, int(selected[0]["total_bytes"]) * 0.90
                    ):
                        reason = "gpu_90_percent_or_stricter_stop"
                if reason:
                    result.update(status="stopped", reason=reason)
                    raise RuntimeError(f"resource stop: {reason}")
                if process.returncode is not None:
                    survivors = [
                        row
                        for row in owned_children(process.pid, excluded)
                        if int(row["pid"]) != process.pid
                    ]
                    if survivors:
                        result.update(status="failed", reason="descendants_after_root_exit")
                        raise RuntimeError("worker exited with surviving descendants")
                    result.update(
                        status="complete" if process.returncode == 0 else "failed",
                        returncode=process.returncode,
                    )
                    break
                time.sleep(5)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if process is not None:
            teardown = stop_tree(process, list(seen.values()), excluded)
            result["teardown"] = teardown
            if teardown["status"] != "complete":
                result["status"] = "failed"
        set_subreaper(previous_subreaper)
        result.update(elapsed_seconds=time.monotonic() - started, final_host=host_sample())
        write_json(output / "result.json", result)
    if result["returncode"] != 0:
        raise RuntimeError(f"child failed: {result['returncode']}")
    if result["status"] != "complete":
        raise RuntimeError("child teardown could not be verified")
    return result
