#!/usr/bin/env python3
"""Durable, aggregate-limited local jobs for the 2026-10-06 bandit campaign.

Run this CLI through ``scripts/limited`` in ``wnn-campaign.slice``. Supply a
profile identifier and a RAM reservation derived from a completed representative
profile; this controller records that evidence but does not guess a reservation.

The parent slice is the hard aggregate boundary. This controller also admits
jobs only when their profiled reservations fit the 10 CPU / 18 GB working
allowance and records every launch and terminal outcome.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, TypeVar

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime"
SLICE = "wnn-campaign.slice"
CPU_CAP = 10
RAM_WORKING_BYTES = 18_000_000_000
RAM_PARENT_BYTES = 20_000_000_000
RAM_HIGH_BYTES = 18_000_000_000
GPU_ALLOCATOR_BYTES = 4_300_000_000
GPU_OBSERVED_BYTES = 5_000_000_000
ACTIVE = {"reserved", "waiting", "running"}
UNRESOLVED = {"preflight", "launch_uncertain"}
ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
CODE_SUFFIXES = {".py", ".sh", ".rs", ".toml", ".yaml", ".yml", ".json"}
POLL_SECONDS = 5.0
OBSERVER_RETRIES = 3
OBSERVER_RETRY_SECONDS = 0.25
T = TypeVar("T")


def utc_now() -> str:
    """Return a UTC timestamp suitable for an immutable attempt record."""
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """Hash one source file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Durably replace a JSON record and its containing directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path) -> dict[str, object]:
    """Read a required JSON object without silently repairing bad evidence."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


@contextmanager
def locked(state: Path) -> Iterator[None]:
    """Serialize all controller decisions across CLI processes."""
    state.mkdir(parents=True, exist_ok=True)
    with (state / "controller.lock").open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def attempt_path(state: Path, attempt: str) -> Path:
    """Return the durable evidence path for one validated attempt ID."""
    return state / "attempts" / f"{attempt}.json"


def job_records(state: Path) -> list[dict[str, object]]:
    """Read the complete attempt ledger; malformed records block admission."""
    paths = sorted((state / "attempts").glob("*.json"))
    return [read_json(path) for path in paths]


def parent_properties() -> dict[str, str]:
    """Inspect the verified aggregate cgroup with a bounded systemctl call."""
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            SLICE,
            "--property=CPUQuotaPerSecUSec,MemoryHigh,MemoryMax,MemorySwapMax,ControlGroup",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def verify_parent(properties: dict[str, str]) -> str:
    """Require the exact authorized aggregate limits and return its cgroup path."""
    expected = {
        "CPUQuotaPerSecUSec": "10s",
        "MemoryHigh": str(RAM_HIGH_BYTES),
        "MemoryMax": str(RAM_PARENT_BYTES),
        "MemorySwapMax": "0",
    }
    mismatches = {
        name: (properties.get(name), value)
        for name, value in expected.items()
        if properties.get(name) != value
    }
    group = properties.get("ControlGroup", "")
    if mismatches or not group.endswith(f"/{SLICE}"):
        raise RuntimeError(f"aggregate parent limits not verified: {mismatches}, group={group!r}")
    return group


def current_cgroup(pid: int | str = "self") -> str:
    """Return the cgroup-v2 path for a process."""
    rows = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    entries = [row.split(":", 2)[2] for row in rows if row.startswith("0::")]
    if len(entries) != 1:
        raise RuntimeError(f"cannot identify one cgroup-v2 membership for pid {pid}")
    return entries[0]


def verify_parent_membership(parent_group: str, pid: int | str = "self") -> None:
    """Reject a controller or worker that is outside the campaign aggregate."""
    actual = current_cgroup(pid)
    prefix = parent_group.rstrip("/") + "/"
    if actual != parent_group and not actual.startswith(prefix):
        raise RuntimeError(f"process is outside {SLICE}: {actual}")


def verify_controller_membership(pid: int | str = "self") -> None:
    """Require the controller CLI itself to inherit the aggregate parent slice."""
    if os.environ.get("WNN_SLICE") != SLICE:
        raise RuntimeError(f"run this CLI inside {SLICE} via scripts/limited")
    if SLICE not in current_cgroup(pid).split("/"):
        raise RuntimeError(f"process is outside {SLICE}")


def cgroup_dir(group: str) -> Path:
    """Resolve a systemd cgroup-v2 path below the mounted hierarchy."""
    path = (Path("/sys/fs/cgroup") / group.lstrip("/")).resolve()
    if not path.is_relative_to(Path("/sys/fs/cgroup").resolve()):
        raise RuntimeError(f"cgroup escaped the unified hierarchy: {group}")
    return path


def memory_current(group: str) -> int:
    """Return current cgroup memory use in exact bytes."""
    value = (cgroup_dir(group) / "memory.current").read_text().strip()
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"invalid memory.current for {group}: {value!r}")
    return int(value)


def memory_peak(group: str) -> int:
    """Read the kernel-maintained high-water mark for one attempt cgroup."""
    value = (cgroup_dir(group) / "memory.peak").read_text().strip()
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"invalid memory.peak for {group}: {value!r}")
    return int(value)


def memory_events(group: str) -> dict[str, int]:
    """Read OOM counters from the parent or attempt cgroup."""
    path = cgroup_dir(group) / "memory.events"
    values = {
        name: int(count) for name, count in (line.split() for line in path.read_text().splitlines())
    }
    required = {"oom", "oom_kill", "oom_group_kill"}
    if not required <= values.keys() or any(value < 0 for value in values.values()):
        raise ValueError(f"incomplete OOM counters for {group}")
    return values


def deltas(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Compute nonnegative counter deltas, preserving counter resets as evidence."""
    result = {key: after.get(key, 0) - value for key, value in before.items()}
    if any(value < 0 for value in result.values()):
        raise RuntimeError(f"cgroup counters decreased: {result}")
    return result


def observe(
    name: str,
    reader: Callable[[], T],
    record: dict[str, object],
    record_path: Path,
) -> T | None:
    """Retry transient observations and persist unknown status without killing work."""
    last_error: str | None = None
    for attempt in range(1, OBSERVER_RETRIES + 1):
        try:
            value = reader()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < OBSERVER_RETRIES:
                time.sleep(OBSERVER_RETRY_SECONDS)
            continue
        record.setdefault("observer_status", {})
        statuses = record["observer_status"]
        if isinstance(statuses, dict):
            statuses[name] = "ok"
        return value
    event = {
        "at": utc_now(),
        "observer": name,
        "state": "unknown",
        "retries": OBSERVER_RETRIES,
        "error": last_error,
    }
    events = record.setdefault("observer_events", [])
    if isinstance(events, list):
        events.append(event)
        del events[:-32]
    statuses = record.setdefault("observer_status", {})
    if isinstance(statuses, dict):
        statuses[name] = "unknown"
    atomic_json(record_path, record)
    print(
        f"observer {name}=unknown after {OBSERVER_RETRIES} retries: {last_error}", file=sys.stderr
    )
    return None


def gpu_memory_bytes() -> int:
    """Observe and sum used memory across every device reported by nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows or any(not row.isascii() or not row.isdigit() for row in rows):
        raise RuntimeError(f"unparseable nvidia-smi memory output: {result.stdout!r}")
    # nvidia-smi reports MiB. Round each device upward to stay conservative.
    return sum(int(row) + 1 for row in rows) * 1024**2


def systemd_properties(unit: str) -> dict[str, str]:
    """Read one attempt unit's active state and effective cgroup path."""
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=ActiveState,SubState,Result,ExecMainStatus,ControlGroup,CPUQuotaPerSecUSec,MemoryMax,MemorySwapMax,OOMPolicy,KillMode",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def request_close(state: Path, reason: str) -> dict[str, object]:
    """Persist a new-admission barrier without signaling healthy workers."""
    marker = state / "STOP_ADMISSION.json"
    if marker.exists():
        return read_json(marker)
    active: list[str] = []
    unreadable: list[str] = []
    for path in sorted((state / "attempts").glob("*.json")):
        try:
            item = read_json(path)
        except (OSError, ValueError):
            unreadable.append(path.name)
            continue
        if item.get("status") in ACTIVE | UNRESOLVED:
            active.append(str(item.get("id", path.stem)))
    value: dict[str, object] = {
        "id": uuid.uuid4().hex,
        "closed_utc": utc_now(),
        "reason": reason,
        "drain_attempts": active,
        "unreadable_attempt_records": unreadable,
    }
    atomic_json(marker, value)
    return value


def close_admission(state: Path, reason: str) -> dict[str, object]:
    """Atomically block future admissions and record the drain set."""
    with locked(state):
        return request_close(state, reason)


def ensure_admission_open(state: Path) -> None:
    """Raise before any reservation is written when a drain barrier exists."""
    marker = state / "STOP_ADMISSION.json"
    if marker.exists():
        raise RuntimeError(
            f"new admission is closed: {read_json(marker).get('reason', 'no reason')}"
        )


def source_manifest(root: Path, command: list[str], contract: Path) -> dict[str, str]:
    """Freeze campaign/runtime code plus explicit command source files."""
    root = root.resolve()
    paths = {Path(__file__).resolve(), root / "scripts/limited"}
    for directory in (root / "src/wnn", root / "submission", root / "scripts"):
        if directory.exists():
            paths.update(path for path in directory.rglob("*.py") if path.is_file())
    paths.add(contract.resolve())
    for token in command:
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.suffix.lower() in CODE_SUFFIXES and candidate.is_file():
            paths.add(candidate.resolve())
    result: dict[str, str] = {}
    for path in sorted(paths):
        key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        result[key] = sha256_file(path)
    return result


def active_reservations(state: Path, *, exclude: str | None = None) -> tuple[float, int]:
    """Sum CPU and RAM reservations for admitted jobs in the ledger."""
    cpus = 0.0
    ram = 0
    for item in job_records(state):
        if exclude is not None and item.get("id") == exclude:
            continue
        if item.get("status") in UNRESOLVED:
            raise RuntimeError(f"unreconciled attempt blocks admission: {item.get('id')}")
        if item.get("status") in ACTIVE:
            limits = item.get("reservation")
            if not isinstance(limits, dict):
                raise ValueError(f"attempt has no reservation: {item.get('id')}")
            cpus += int(limits["cpus"])
            ram += int(limits["ram_bytes"])
    return cpus, ram


def support_cpu_reservations(state: Path, parent_group: str) -> dict[str, float]:
    """Account active parent-slice services and scopes outside this attempt ledger."""
    known_units = {str(item["unit"]) for item in job_records(state) if item.get("status") in ACTIVE}
    result: dict[str, float] = {}
    for group in cgroup_dir(parent_group).iterdir():
        if not group.is_dir() or group.name in known_units:
            continue
        events_path = group / "cgroup.events"
        cpu_path = group / "cpu.max"
        try:
            populated = dict(line.split() for line in events_path.read_text().splitlines()).get(
                "populated"
            )
            if populated == "0":
                continue
            parts = cpu_path.read_text().split()
        except OSError as error:
            raise RuntimeError(f"cannot inspect support cgroup {group.name}: {error}") from error
        if (
            len(parts) != 2
            or not parts[1].isascii()
            or not parts[1].isdigit()
            or int(parts[1]) <= 0
        ):
            raise ValueError(f"invalid cpu.max for support cgroup {group.name}: {parts}")
        if parts[0] == "max":
            result[group.name] = float(CPU_CAP)
        elif parts[0].isascii() and parts[0].isdigit():
            result[group.name] = int(parts[0]) / int(parts[1])
        else:
            raise ValueError(f"invalid cpu.max for support cgroup {group.name}: {parts}")
    return result


def verify_capacity(
    state: Path,
    parent_group: str,
    cpus: int,
    ram_bytes: int,
    *,
    exclude: str | None = None,
    current_memory_bytes: int | None = None,
) -> dict[str, object]:
    """Check both committed reservations and measured parent use before admission."""
    active_cpus, active_ram = active_reservations(state, exclude=exclude)
    support = support_cpu_reservations(state, parent_group)
    support_cpus = sum(support.values())
    if active_cpus + support_cpus + cpus > CPU_CAP:
        raise RuntimeError(f"CPU reservation would exceed {CPU_CAP} aggregate cores")
    if active_ram + ram_bytes > RAM_WORKING_BYTES:
        raise RuntimeError("profiled RAM reservations would exceed the 18 GB working allowance")
    measured = (
        memory_current(parent_group) if current_memory_bytes is None else current_memory_bytes
    )
    if measured + ram_bytes > RAM_WORKING_BYTES:
        raise RuntimeError(
            f"measured parent RAM plus profile reserve exceeds 18 GB: {measured}+{ram_bytes}"
        )
    return {
        "parent_memory_current_bytes": measured,
        "profile_ram_reserve_bytes": ram_bytes,
        "committed_cpu_cores": active_cpus,
        "support_cpu_reserve_cores": support_cpus,
        "support_cpu_reservations": support,
        "committed_ram_reserve_bytes": active_ram,
    }


def verify_attempt_id(attempt: str) -> None:
    """Reject attempt IDs that could escape their single-file ledger name."""
    if not ATTEMPT_ID.fullmatch(attempt):
        raise ValueError("attempt ID must contain 1-80 letters, digits, underscores or dashes")


def service_command(
    root: Path,
    state: Path,
    attempt: str,
    kind: str,
    cpus: int,
    ram_bytes: int,
    python: str,
    admission_timeout: int,
) -> list[str]:
    """Build a bounded systemd service command for one admitted worker."""
    unit = f"wnn-bandit-{attempt}.service"
    env = {
        "WNN_SLICE": SLICE,
        "WNN_THREADS": str(cpus),
        "WNN_CPU_QUOTA": f"{cpus * 100}%",
        "WNN_CPU_PCT": str(min(100, cpus * 10)),
        "WNN_MEM_MAX": str(ram_bytes),
        "WNN_LOADER_WORKERS": "0",
        "OMP_NUM_THREADS": str(cpus),
        "MKL_NUM_THREADS": str(cpus),
        "OPENBLAS_NUM_THREADS": str(cpus),
        "NUMBA_NUM_THREADS": str(cpus),
        "POLARS_MAX_THREADS": str(cpus),
        "ARROW_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(root / "src"),
        "CUDA_VISIBLE_DEVICES": "0" if kind == "gpu" else "",
    }
    for key in ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "VIRTUAL_ENV"):
        if key in os.environ:
            env[key] = os.environ[key]
    if kind == "gpu":
        env["WNN_GPU_MEMORY_BYTES"] = str(GPU_ALLOCATOR_BYTES)
        env["WNN_VRAM_GB"] = "4.3"
    command = [
        "systemd-run",
        "--user",
        "--no-block",
        "--quiet",
        f"--unit={unit}",
        f"--slice={SLICE}",
        "-p",
        "Type=exec",
        "-p",
        f"TimeoutStartSec={admission_timeout + 10}",
        "-p",
        f"CPUQuota={cpus * 100}%",
        "-p",
        f"MemoryMax={ram_bytes}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=continue",
        "-p",
        "KillMode=control-group",
        "-p",
        "Restart=no",
        f"--working-directory={root}",
    ]
    command.extend(f"--setenv={key}={value}" for key, value in env.items())
    command.extend(
        [
            python,
            str(Path(__file__).resolve()),
            "--root",
            str(root),
            "--state",
            str(state),
            "worker",
            "--attempt",
            attempt,
            "--admission-timeout",
            str(admission_timeout),
        ]
    )
    return command


def launch(args: argparse.Namespace) -> dict[str, object]:
    """Reserve aggregate resources and submit one durable systemd attempt."""
    verify_attempt_id(args.attempt)
    if args.kind not in {"cpu", "gpu"}:
        raise ValueError("local job kind must be cpu or gpu")
    if not 1 <= args.cpus <= CPU_CAP:
        raise ValueError(f"worker CPU reservation must be in [1, {CPU_CAP}]")
    if not 1 <= args.reserve_bytes <= RAM_WORKING_BYTES:
        raise ValueError("profile reserve must be a positive byte count within 18 GB")
    if args.admission_timeout <= 0:
        raise ValueError("admission timeout must be positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("a literal worker command is required after --")
    contract = args.contract.resolve(strict=True)
    manifest = source_manifest(args.root, command, contract)
    contract_key = (
        str(contract.relative_to(args.root))
        if contract.is_relative_to(args.root)
        else str(contract)
    )
    state = args.state.resolve()
    record_path = attempt_path(state, args.attempt)
    unit = f"wnn-bandit-{args.attempt}.service"
    record: dict[str, object] = {
        "id": args.attempt,
        "unit": unit,
        "kind": args.kind,
        "profile": args.profile,
        "command": command,
        "working_directory": str(args.root),
        "contract": str(contract),
        "contract_sha256": manifest[contract_key],
        "source_sha256": manifest,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "parent_slice": SLICE,
        "reservation": {"cpus": args.cpus, "ram_bytes": args.reserve_bytes},
        "phase": "preflight",
        "status": "preflight",
        "created_utc": utc_now(),
        "heartbeat_utc": utc_now(),
        "elapsed_seconds": 0.0,
        "peak_ram_bytes": 0,
        "peak_parent_ram_bytes": 0,
        "peak_observed_vram_bytes": 0,
        "observer_events": [],
    }
    with locked(state):
        ensure_admission_open(state)
        if record_path.exists():
            raise FileExistsError(f"attempt evidence already exists: {record_path}")
        atomic_json(record_path, record)
        try:
            verify_controller_membership()
            props = observe("parent_systemd_launch", parent_properties, record, record_path)
            if props is None:
                raise RuntimeError("parent systemd limits are unknown after bounded retries")
            parent_group = verify_parent(props)
            verify_parent_membership(parent_group)
            initial_vram = None
            if args.kind == "gpu":
                initial_vram = observe("nvidia_smi_launch", gpu_memory_bytes, record, record_path)
                if initial_vram is None:
                    raise RuntimeError("initial GPU memory is unknown after bounded retries")
                if initial_vram >= GPU_OBSERVED_BYTES:
                    raise RuntimeError(
                        f"device use is already at the {GPU_OBSERVED_BYTES} byte ceiling"
                    )
            measured = observe(
                "parent_memory_admission",
                lambda: memory_current(parent_group),
                record,
                record_path,
            )
            parent_events = observe(
                "parent_memory_events_admission",
                lambda: memory_events(parent_group),
                record,
                record_path,
            )
            if measured is None or parent_events is None:
                raise RuntimeError("aggregate memory/OOM counters are unknown at admission")
            resources = verify_capacity(
                state,
                parent_group,
                args.cpus,
                args.reserve_bytes,
                exclude=args.attempt,
                current_memory_bytes=measured,
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            record.update(
                status="not_started",
                phase="admission_failed",
                failure=f"{type(error).__name__}: {error}",
                finished_utc=utc_now(),
            )
            atomic_json(record_path, record)
            raise
        record.update(
            parent_properties_at_launch=props,
            admission_measurement=resources,
            parent_memory_events_at_launch=parent_events,
            observed_vram_cap_bytes=GPU_OBSERVED_BYTES if args.kind == "gpu" else 0,
            initial_observed_vram_bytes=initial_vram,
            allocator_admission_bytes=GPU_ALLOCATOR_BYTES if args.kind == "gpu" else 0,
            peak_parent_ram_bytes=resources["parent_memory_current_bytes"],
            phase="reserved",
            status="reserved",
            service_submit_requested_utc=utc_now(),
            heartbeat_utc=utc_now(),
        )
        atomic_json(record_path, record)
        launch_command = service_command(
            args.root,
            state,
            args.attempt,
            args.kind,
            args.cpus,
            args.reserve_bytes,
            args.python,
            args.admission_timeout,
        )
        try:
            subprocess.run(launch_command, check=True, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as error:
            latest = read_json(record_path)
            latest["launch_error"] = f"{type(error).__name__}: {error}"
            latest["launch_error_utc"] = utc_now()
            if latest.get("status") == "reserved":
                latest["status"] = "launch_uncertain"
            atomic_json(record_path, latest)
            raise RuntimeError(
                f"systemd launch is uncertain; retain reservation and reconcile {args.attempt}"
            ) from error
    return {"id": args.attempt, "unit": unit, "status": "reserved", "command": command}


def stop_child(child: subprocess.Popen[bytes]) -> None:
    """Stop only this attempt's process group after an actual resource breach."""
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def wait_for_gpu_lock(
    state: Path, timeout: int, record: dict[str, object], path: Path
) -> TextIO | None:
    """Serialize all admitted local GPU workers, including workers queued at drain."""
    lock_path = state / "gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    handle = lock_path.open("a")
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.monotonic() - started >= timeout:
                handle.close()
                record.update(
                    status="not_started",
                    phase="gpu_lock_timeout",
                    failure=f"GPU lock unavailable for {timeout} seconds",
                    finished_utc=utc_now(),
                )
                atomic_json(path, record)
                return None
            record.update(phase="waiting_for_gpu", heartbeat_utc=utc_now())
            atomic_json(path, record)
            time.sleep(min(POLL_SECONDS, timeout - (time.monotonic() - started)))


def _run_worker(root: Path, state: Path, attempt: str, admission_timeout: int) -> int:
    """Run, monitor and durably finish a command inside its own campaign service."""
    verify_attempt_id(attempt)
    record_path = attempt_path(state, attempt)
    record = read_json(record_path)
    if record.get("status") != "reserved":
        raise RuntimeError(f"attempt is not an unstarted reservation: {attempt}")
    parent = observe("parent_systemd_start", parent_properties, record, record_path)
    if parent is None:
        raise RuntimeError("parent systemd properties are unknown at worker start")
    parent_group = verify_parent(parent)
    own_properties = observe(
        "attempt_systemd_start",
        lambda: systemd_properties(str(record["unit"])),
        record,
        record_path,
    )
    if own_properties is None:
        raise RuntimeError("attempt systemd properties are unknown at worker start")
    group = own_properties.get("ControlGroup", "")
    expected_prefix = parent_group.rstrip("/") + "/"
    if not group.startswith(expected_prefix) or current_cgroup() != group:
        raise RuntimeError(f"worker is outside its verified attempt cgroup: {group}")
    reservation = record.get("reservation")
    if not isinstance(reservation, dict):
        raise ValueError("attempt reservation is malformed")
    cpus, ram_bytes = int(reservation["cpus"]), int(reservation["ram_bytes"])
    expected_properties = {
        "CPUQuotaPerSecUSec": f"{cpus}s",
        "MemoryMax": str(ram_bytes),
        "MemorySwapMax": "0",
        "OOMPolicy": "continue",
        "KillMode": "control-group",
    }
    wrong = {
        key: (own_properties.get(key), expected)
        for key, expected in expected_properties.items()
        if own_properties.get(key) != expected
    }
    if wrong:
        raise RuntimeError(f"attempt resource properties are not verified: {wrong}")
    verify_parent_membership(parent_group)
    attempt = str(record["id"])
    command = record.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) for arg in command)
    ):
        raise ValueError("reserved command is malformed")
    manifest = record.get("source_sha256")
    if not isinstance(manifest, dict):
        raise ValueError("source manifest is missing")
    contract = Path(str(record["contract"]))
    actual_manifest = source_manifest(root, command, contract)
    if actual_manifest != manifest:
        raise RuntimeError("frozen source changed between reservation and worker start")
    kind = str(record["kind"])
    record.update(
        parent_properties_at_start=parent,
        attempt_properties_at_start=own_properties,
        phase="preflight",
        heartbeat_utc=utc_now(),
    )
    lock_file: TextIO | None = None
    if kind == "gpu":
        lock_file = wait_for_gpu_lock(state, admission_timeout, record, record_path)
        if lock_file is None:
            return 0
        initial_vram = observe("nvidia_smi_admission", gpu_memory_bytes, record, record_path)
        if initial_vram is None:
            record.update(
                status="not_started", phase="gpu_observer_unknown", finished_utc=utc_now()
            )
            atomic_json(record_path, record)
            lock_file.close()
            return 1
        if initial_vram >= GPU_OBSERVED_BYTES:
            record.update(
                status="not_started",
                phase="gpu_budget_preflight",
                failure=f"device was already at {initial_vram} observed bytes",
                finished_utc=utc_now(),
            )
            atomic_json(record_path, record)
            lock_file.close()
            return 1
    measured_parent = observe(
        "parent_memory_admission_recheck",
        lambda: memory_current(parent_group),
        record,
        record_path,
    )
    if measured_parent is None:
        record.update(
            status="not_started",
            phase="admission_observer_unknown",
            failure="parent memory is unknown after bounded retries",
            finished_utc=utc_now(),
        )
        atomic_json(record_path, record)
        if lock_file is not None:
            lock_file.close()
        return 1
    try:
        before = verify_capacity(
            state,
            parent_group,
            cpus,
            ram_bytes,
            exclude=attempt,
            current_memory_bytes=measured_parent,
        )
    except RuntimeError as error:
        record.update(
            status="not_started",
            phase="admission_recheck_failed",
            failure=str(error),
            finished_utc=utc_now(),
        )
        atomic_json(record_path, record)
        if lock_file is not None:
            lock_file.close()
        return 1
    start_events = observe(
        "attempt_memory_events_start", lambda: memory_events(group), record, record_path
    )
    parent_start_events = observe(
        "parent_memory_events_start", lambda: memory_events(parent_group), record, record_path
    )
    if start_events is None or parent_start_events is None:
        record.update(
            status="not_started",
            phase="oom_observer_unknown",
            failure="could not establish baseline OOM counters",
            finished_utc=utc_now(),
        )
        atomic_json(record_path, record)
        if lock_file is not None:
            lock_file.close()
        return 1
    record.update(
        initial_memory_events=start_events,
        initial_parent_memory_events=parent_start_events,
    )
    if source_manifest(root, command, contract) != manifest:
        record.update(
            status="not_started",
            phase="source_changed_after_admission",
            failure="frozen source changed while waiting for resources",
            finished_utc=utc_now(),
        )
        atomic_json(record_path, record)
        if lock_file is not None:
            lock_file.close()
        return 1
    log_path = state / "logs" / f"{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    failure: str | None = None
    status = "failed"
    breach = False
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONPATH": str(root / "src"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "WNN_THREADS": str(cpus),
        "WNN_CPU_QUOTA": f"{cpus * 100}%",
        "WNN_CPU_PCT": str(min(100, cpus * 10)),
        "WNN_MEM_MAX": str(ram_bytes),
        "WNN_SLICE": SLICE,
        "WNN_LOADER_WORKERS": "0",
        "CUDA_VISIBLE_DEVICES": "0" if kind == "gpu" else "",
    }
    for key in ("LD_LIBRARY_PATH", "VIRTUAL_ENV"):
        if key in os.environ:
            env[key] = os.environ[key]
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        env[key] = str(cpus)
    env["ARROW_NUM_THREADS"] = "1"
    if kind == "gpu":
        env["WNN_GPU_MEMORY_BYTES"] = str(GPU_ALLOCATOR_BYTES)
        env["WNN_VRAM_GB"] = "4.3"
    try:
        with log_path.open("xb", buffering=0) as log:
            record.update(
                status="running",
                phase="running",
                started_utc=utc_now(),
                started_unix=time.time(),
                heartbeat_utc=utc_now(),
                admission_measurement_at_start=before,
                log=str(log_path),
            )
            atomic_json(record_path, record)
            child = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while child.poll() is None:
                elapsed = time.monotonic() - started
                observed_parent = observe(
                    "parent_systemd_runtime", parent_properties, record, record_path
                )
                if observed_parent is not None:
                    try:
                        observed_group = verify_parent(observed_parent)
                    except RuntimeError as error:
                        failure = f"aggregate parent changed during attempt: {error}"
                        breach = True
                        break
                    parent_group = observed_group
                attempt_props = observe(
                    "attempt_systemd_runtime",
                    lambda: systemd_properties(str(record["unit"])),
                    record,
                    record_path,
                )
                group_current = observe(
                    "attempt_memory_current",
                    lambda: memory_current(group),
                    record,
                    record_path,
                )
                parent_current = observe(
                    "parent_memory_current",
                    lambda group_path=parent_group: memory_current(group_path),
                    record,
                    record_path,
                )
                events = observe(
                    "attempt_memory_events", lambda: memory_events(group), record, record_path
                )
                parent_events = observe(
                    "parent_memory_events",
                    lambda group_path=parent_group: memory_events(group_path),
                    record,
                    record_path,
                )
                observed_vram = None
                if kind == "gpu":
                    observed_vram = observe(
                        "nvidia_smi_runtime", gpu_memory_bytes, record, record_path
                    )
                if group_current is not None:
                    record["peak_ram_bytes"] = max(int(record["peak_ram_bytes"]), group_current)
                    if group_current > ram_bytes:
                        failure = (
                            f"attempt memory use exceeded its {ram_bytes} byte profile reserve"
                        )
                        breach = True
                if parent_current is not None:
                    record["peak_parent_ram_bytes"] = max(
                        int(record["peak_parent_ram_bytes"]), parent_current
                    )
                    if parent_current > RAM_WORKING_BYTES:
                        failure = f"aggregate working RAM exceeded {RAM_WORKING_BYTES} bytes"
                        breach = True
                if observed_vram is not None:
                    record["peak_observed_vram_bytes"] = max(
                        int(record["peak_observed_vram_bytes"]), observed_vram
                    )
                    if observed_vram >= GPU_OBSERVED_BYTES:
                        failure = f"observed VRAM exceeded {GPU_OBSERVED_BYTES} bytes"
                        breach = True
                if events is not None:
                    delta = deltas(start_events, events)
                    record["memory_event_deltas"] = delta
                    if any(delta.get(key, 0) for key in ("oom", "oom_kill", "oom_group_kill")):
                        failure = f"attempt cgroup OOM counters increased: {delta}"
                        breach = True
                if parent_events is not None:
                    delta = deltas(parent_start_events, parent_events)
                    record["parent_memory_event_deltas"] = delta
                    if any(delta.get(key, 0) for key in ("oom", "oom_kill", "oom_group_kill")):
                        failure = f"aggregate parent OOM counters increased: {delta}"
                        breach = True
                if attempt_props is not None:
                    active_state = attempt_props.get("ActiveState")
                    record["attempt_active_state"] = active_state or "unknown"
                    if active_state not in {"active", "activating", "deactivating"}:
                        record["attempt_active_state"] = "unknown"
                        record.setdefault("observer_events", []).append(
                            {
                                "at": utc_now(),
                                "observer": "attempt_systemd_runtime",
                                "state": "unknown",
                                "error": f"unexpected ActiveState={active_state!r}",
                            }
                        )
                record.update(
                    elapsed_seconds=elapsed,
                    heartbeat_utc=utc_now(),
                    phase="resource_breach" if breach else "running",
                )
                atomic_json(record_path, record)
                if breach:
                    close_admission(state, f"resource budget breach in {attempt}: {failure}")
                    break
                time.sleep(POLL_SECONDS)
        if breach and child is not None:
            stop_child(child)
        if child is not None:
            code = child.wait()
            record["exit_status"] = code
            if breach:
                status = "failed"
            elif code == 0:
                status = "finished"
            else:
                failure = f"worker exited with status {code}"
                status = "failed"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        failure = f"{type(error).__name__}: {error}"
        if child is not None and child.poll() is None:
            stop_child(child)
        if child is not None:
            record["exit_status"] = child.returncode
    finally:
        elapsed = time.monotonic() - started
        final_events = observe(
            "attempt_memory_events_final", lambda: memory_events(group), record, record_path
        )
        final_parent_events = observe(
            "parent_memory_events_final", lambda: memory_events(parent_group), record, record_path
        )
        if final_events is not None:
            final_delta = deltas(start_events, final_events)
            record["final_memory_event_deltas"] = final_delta
            if any(final_delta.get(key, 0) for key in ("oom", "oom_kill", "oom_group_kill")):
                status = "failed"
                failure = f"attempt cgroup OOM counters increased at exit: {final_delta}"
                close_admission(state, f"resource budget breach in {attempt}: {failure}")
        if final_parent_events is not None:
            final_parent_delta = deltas(parent_start_events, final_parent_events)
            record["final_parent_memory_event_deltas"] = final_parent_delta
            if any(final_parent_delta.get(key, 0) for key in ("oom", "oom_kill", "oom_group_kill")):
                status = "failed"
                failure = f"aggregate parent OOM counters increased at exit: {final_parent_delta}"
                close_admission(state, f"resource budget breach in {attempt}: {failure}")
        peak = observe("attempt_memory_peak_final", lambda: memory_peak(group), record, record_path)
        if peak is not None:
            record["peak_ram_bytes"] = max(int(record["peak_ram_bytes"]), peak)
        final_parent_current = observe(
            "parent_memory_current_final",
            lambda group_path=parent_group: memory_current(group_path),
            record,
            record_path,
        )
        if final_parent_current is not None:
            record["peak_parent_ram_bytes"] = max(
                int(record["peak_parent_ram_bytes"]), final_parent_current
            )
            if final_parent_current > RAM_WORKING_BYTES:
                status = "failed"
                failure = f"aggregate working RAM exceeded {RAM_WORKING_BYTES} bytes at exit"
                close_admission(state, f"resource budget breach in {attempt}: {failure}")
        if kind == "gpu":
            final_vram = observe("nvidia_smi_final", gpu_memory_bytes, record, record_path)
            if final_vram is not None:
                record["peak_observed_vram_bytes"] = max(
                    int(record["peak_observed_vram_bytes"]), final_vram
                )
                if final_vram >= GPU_OBSERVED_BYTES:
                    status = "failed"
                    failure = f"observed VRAM reached {GPU_OBSERVED_BYTES} bytes at exit"
                    close_admission(state, f"resource budget breach in {attempt}: {failure}")
        if (
            child is not None
            and not breach
            and child.returncode is not None
            and child.returncode != 0
        ):
            status = "failed"
        record.update(
            status=status,
            phase="terminal",
            failure=failure,
            elapsed_seconds=elapsed,
            heartbeat_utc=utc_now(),
            finished_utc=utc_now(),
        )
        atomic_json(record_path, record)
        if lock_file is not None:
            lock_file.close()
    return 0 if status == "finished" else 1


def worker(root: Path, state: Path, attempt: str, admission_timeout: int) -> int:
    """Turn preflight exceptions into terminal records instead of stale reservations."""
    try:
        return _run_worker(root, state, attempt, admission_timeout)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        path = attempt_path(state, attempt)
        with suppress(OSError, ValueError):
            record = read_json(path)
            if record.get("status") in ACTIVE | {"launch_uncertain"}:
                record.update(
                    status="not_started" if "started_utc" not in record else "failed",
                    phase="preflight_failed" if "started_utc" not in record else "monitor_failed",
                    failure=f"{type(error).__name__}: {error}",
                    finished_utc=utc_now(),
                )
                atomic_json(path, record)
        print(f"attempt {attempt} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


def reconcile(state: Path) -> list[str]:
    """Mark systemd-ended attempts with missing terminal writes as lost evidence."""
    changed: list[str] = []
    with locked(state):
        for record in job_records(state):
            if record.get("status") not in ACTIVE | UNRESOLVED:
                continue
            if record.get("status") == "preflight":
                record.update(
                    status="not_started",
                    phase="controller_lost_during_preflight",
                    failure="controller ended before systemd service submission",
                    finished_utc=utc_now(),
                )
                atomic_json(attempt_path(state, str(record["id"])), record)
                changed.append(str(record["id"]))
                continue
            unit = str(record["unit"])
            props = observe(
                "reconcile_systemd",
                lambda unit_name=unit: systemd_properties(unit_name),
                record,
                attempt_path(state, str(record["id"])),
            )
            if props is None:
                print(f"{unit}: terminal state unknown after bounded retries", file=sys.stderr)
                continue
            if props.get("ActiveState") in {"active", "activating", "deactivating"}:
                continue
            record.update(
                status="lost",
                phase="lost_monitor",
                failure="service ended without a trusted terminal attempt record",
                systemd_terminal=props,
                finished_utc=utc_now(),
            )
            atomic_json(attempt_path(state, str(record["id"])), record)
            request_close(state, f"attempt {record['id']} lost its monitor")
            changed.append(str(record["id"]))
    return changed


def drain(state: Path, reason: str, poll_seconds: float = POLL_SECONDS) -> list[dict[str, object]]:
    """Close new admission and wait for all admitted work to finish naturally."""
    close_admission(state, reason)
    while True:
        records = job_records(state)
        active = [item for item in records if item.get("status") in ACTIVE | UNRESOLVED]
        if not active:
            return records
        reconcile(state)
        records = job_records(state)
        active = [item for item in records if item.get("status") in ACTIVE | UNRESOLVED]
        if not active:
            return records
        print(
            "draining active attempts: " + ", ".join(str(item["id"]) for item in active),
            file=sys.stderr,
        )
        time.sleep(poll_seconds)


def resume(state: Path, review: str) -> None:
    """Reopen only after a written review and retain the closed barrier as history."""
    if not review.strip():
        raise ValueError("a review note is required to reopen admission")
    with locked(state):
        active = [item for item in job_records(state) if item.get("status") in ACTIVE | UNRESOLVED]
        if active:
            raise RuntimeError("cannot reopen while attempts remain active or uncertain")
        marker = state / "STOP_ADMISSION.json"
        if not marker.exists():
            raise FileNotFoundError("admission is already open")
        value = read_json(marker)
        value["reopened_utc"] = utc_now()
        value["review"] = review
        history = state / "barriers" / f"{value['id']}.json"
        atomic_json(history, value)
        marker.unlink()
        dir_fd = os.open(state, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def status(state: Path) -> dict[str, object]:
    """Return ledger, admission barrier and verified parent snapshot."""
    value: dict[str, object] = {
        "state": str(state),
        "admission_closed": (state / "STOP_ADMISSION.json").exists(),
        "attempts": job_records(state),
    }
    try:
        props = parent_properties()
        value["parent_verified"] = verify_parent(props)
        value["parent_properties"] = props
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        value["parent_verified"] = False
        value["parent_observer_error"] = f"{type(error).__name__}: {error}"
    return value


def parser() -> argparse.ArgumentParser:
    """Define the launch, worker, status, reconcile, drain and resume commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("launch", help="reserve resources and submit a durable attempt")
    start.add_argument("--attempt", required=True)
    start.add_argument("--kind", choices=("cpu", "gpu"), required=True)
    start.add_argument("--cpus", type=int, required=True)
    start.add_argument("--reserve-bytes", type=int, required=True)
    start.add_argument("--profile", required=True)
    start.add_argument("--contract", type=Path, required=True)
    start.add_argument("--python", default=sys.executable)
    start.add_argument("--admission-timeout", type=int, default=900)
    start.add_argument("command", nargs=argparse.REMAINDER)
    run = commands.add_parser("worker", help=argparse.SUPPRESS)
    run.add_argument("--attempt", required=True)
    run.add_argument("--admission-timeout", type=int, default=900)
    commands.add_parser("status", help="show durable attempts and parent limits")
    commands.add_parser(
        "reconcile", help="record attempts whose systemd monitor ended unexpectedly"
    )
    stop = commands.add_parser("drain", help="close new admission and drain admitted work")
    stop.add_argument("--reason", required=True)
    reopen = commands.add_parser(
        "resume", help="reopen after all jobs end and a review is recorded"
    )
    reopen.add_argument("--review", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the controller or one systemd-hosted attempt worker."""
    args = parser().parse_args(argv)
    root = args.root.resolve()
    state = args.state.resolve() if args.state.is_absolute() else (root / args.state).resolve()
    verify_controller_membership()
    if args.action == "launch":
        args.root = root
        args.state = state
        print(json.dumps(launch(args), indent=2))
    elif args.action == "worker":
        print(worker(root, state, args.attempt, args.admission_timeout))
        return (
            0
            if read_json(attempt_path(state, args.attempt)).get("status")
            in {"finished", "not_started"}
            else 1
        )
    elif args.action == "status":
        print(json.dumps(status(state), indent=2))
    elif args.action == "reconcile":
        print(json.dumps({"lost_attempts": reconcile(state)}, indent=2))
    elif args.action == "drain":
        records = drain(state, args.reason)
        print(json.dumps({"drained": True, "attempts": len(records)}, indent=2))
    elif args.action == "resume":
        resume(state, args.review)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
