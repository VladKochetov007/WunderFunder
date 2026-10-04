from __future__ import annotations

import fcntl
import os
import signal
import socket
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from types import FrameType

from wnn.campaign import (
    ACTIVE,
    AdmissionBusy,
    BudgetExhausted,
    atomic_json,
    attempts,
    locked,
    pause,
    pause_locked,
    read_json,
    ready,
    reserve_attempt,
    runtime_identity,
)
from wnn.campaign_spec import (
    RAM_MAX,
    SLICE,
    ExperimentSpec,
    content_file,
    digest,
    rooted,
    validate_local,
    validate_spec,
    verify_files,
)


def capture(argv: list[str]) -> str:
    """Inspect local state with a deadline; failed monitoring is never idle."""
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=5).stdout


def properties(unit: str) -> dict[str, str]:
    """Read resource and lifecycle evidence for an exact systemd unit."""
    output = capture(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "LoadState,ActiveState,Result,ExecMainStatus,ControlGroup,CPUQuotaPerSecUSec,MemoryMax,MemorySwapMax",
        ]
    )
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def check_parent(value: dict[str, str]) -> None:
    """Reject absent, oversized or uninspectable parent resource controls."""
    quota = value.get("CPUQuotaPerSecUSec", "")
    if not quota.endswith("s") or not quota[:-1].replace(".", "", 1).isdigit():
        raise RuntimeError("cannot interpret parent CPU quota")
    if not 0 < float(quota[:-1]) <= 12:
        raise RuntimeError("parent CPU quota exceeds 12 CPUs")
    if not 0 < int(value.get("MemoryMax", "0")) <= RAM_MAX or value.get("MemorySwapMax") != "0":
        raise RuntimeError("parent RAM/swap limits are not confirmed")
    if not value.get("ControlGroup", "").endswith(f"/{SLICE}"):
        raise RuntimeError("parent control group is absent")


def device_memory() -> int:
    """Read total local device usage, including allocations outside PyTorch."""
    output = capture(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    rows = output.splitlines()
    if len(rows) != 1 or not rows[0].strip().isdigit():
        raise RuntimeError("local adapter requires one inspectable GPU")
    # nvidia-smi reports MiB; add one MiB to avoid treating a rounded value as exact.
    return (int(rows[0]) + 1) * 1024**2


def check_orphans(state: Path) -> None:
    """Refuse unmanaged services and stale ledger entries instead of duplicating work."""
    records = attempts(state)
    known = {str(item["unit"]): item for item in records}
    units = capture(
        [
            "systemctl",
            "--user",
            "list-units",
            "--type=service",
            "--state=active,activating,deactivating",
            "--no-legend",
            "--plain",
            "wnn-c04-*.service",
        ]
    )
    for line in units.splitlines():
        unit = line.split()[0]
        if (
            unit in known
            and known[unit]["status"] not in ACTIVE
            and time.time() - float(str(known[unit].get("finished_unix", 0))) < 15
        ):
            raise AdmissionBusy(f"previous service is completing teardown: {unit}")
        if unit not in known or known[unit]["status"] not in ACTIVE:
            raise RuntimeError(f"unmanaged campaign service: {unit}")
    for record in records:
        if record["status"] in ACTIVE:
            current = properties(str(record["unit"]))
            if current.get("ActiveState") not in {"active", "activating", "deactivating"}:
                raise RuntimeError(f"stale attempt needs reconciliation: {record['id']}")
            if (
                record["status"] == "running"
                and time.time() - float(str(record["heartbeat_unix"])) > 30
            ):
                raise RuntimeError(f"worker heartbeat is stale: {record['id']}")


def worker_environment(spec: ExperimentSpec) -> dict[str, str]:
    """Build explicit resource variables without forwarding controller credentials."""
    count = str(spec["limits"]["cpus"])
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(Path.home()),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "WNN_THREADS": count,
        "WNN_GPU_MEMORY_BYTES": str(spec["limits"]["gpu_memory_bytes"]),
        "CUDA_VISIBLE_DEVICES": "0" if spec["kind"] == "gpu" else "",
    }
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        env[key] = count
    env["ARROW_NUM_THREADS"] = "1"
    env.update(spec["environment"])
    return env


def service_command(
    root: Path, state: Path, record: dict[str, object], interpreter: str
) -> list[str]:
    """Create one bounded service containing both monitor and execution children."""
    spec = validate_spec(record["spec"])
    limits = spec["limits"]
    return [
        "systemd-run",
        "--user",
        "--no-block",
        "--quiet",
        f"--unit={record['unit']}",
        f"--slice={SLICE}",
        "-p",
        "Type=notify",
        "-p",
        "NotifyAccess=main",
        "-p",
        f"CPUQuota={limits['cpus'] * 100}%",
        "-p",
        f"MemoryMax={limits['ram_bytes']}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=kill",
        "-p",
        "KillMode=control-group",
        "-p",
        "TimeoutStartSec=900",
        "-p",
        f"RuntimeMaxSec={limits['timeout_seconds'] + 20}",
        "-p",
        "TimeoutStopSec=10",
        "-p",
        "Restart=no",
        f"--working-directory={root}",
        "--setenv=PYTHONDONTWRITEBYTECODE=1",
        interpreter,
        str(root / "scripts/run_campaign.py"),
        "--root",
        str(root),
        "--state",
        str(state),
        "worker",
        str(record["id"]),
        "--python",
        interpreter,
    ]


def launch(
    root: Path, state: Path, spec: ExperimentSpec, attempt: str, interpreter: str
) -> dict[str, object]:
    """Verify content and atomically reserve a single durable local attempt."""
    validate_local(spec)
    # Large inputs are hashed once in the bounded worker, not in the caller's shell.
    verify_files(root, spec["code"])
    if any(rooted(root, name).exists() for name in spec["artifacts"]):
        raise FileExistsError("required output already exists; choose a fresh run identity")
    record = None
    try:
        with locked(state):
            ready(state, root)
            check_parent(properties(SLICE))
            check_orphans(state)
            if spec["kind"] == "gpu" and device_memory() >= spec["limits"]["gpu_stop_bytes"]:
                raise RuntimeError("device already exceeds operational threshold")
            record = reserve_attempt(state, root, spec, attempt)
            command = service_command(root, state, record, interpreter)
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (AdmissionBusy, BudgetExhausted, FileExistsError):
        raise
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        detail = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail += f"; stderr: {error.stderr.strip()}"
        if record is not None:
            record.update(
                status="launch_uncertain",
                failure=detail,
                charged_seconds=record["reserved_seconds"],
                finished_unix=time.time(),
            )
            atomic_json(state / "attempts" / f"{attempt}.json", record)
        if not (state / "PAUSE.json").exists():
            pause(state, f"admission failed: {detail}", attempt)
        raise
    return record


def notify_ready() -> None:
    """Start the independent execution timeout after bounded content preflight."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        raise RuntimeError("worker must run in its systemd notify service")
    if address.startswith("@"):
        address = "\0" + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notifier:
        notifier.settimeout(5)
        notifier.connect(address)
        notifier.sendall(b"READY=1")


def memory_events(group: Path) -> dict[str, int]:
    """Read cgroup OOM counters while the service still exists."""
    return {
        key: int(value)
        for key, value in (
            line.split() for line in (group / "memory.events").read_text().splitlines()
        )
    }


def stop_child(child: subprocess.Popen[bytes]) -> None:
    """Terminate the command's process group, including surviving descendants."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=5)
        return
    with suppress(subprocess.TimeoutExpired):
        child.wait(timeout=5)
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=5)


def run_worker(root: Path, state: Path, attempt: str, interpreter: str) -> int:
    """Monitor one scoped command; preserve failures and block further admission."""
    record_path = state / "attempts" / f"{attempt}.json"
    record = read_json(record_path)
    spec = validate_spec(record["spec"])
    validate_local(spec)
    signals: list[int] = []

    def interrupted(signum: int, frame: FrameType | None) -> None:
        signals.append(signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    child = None
    start = None
    accounting_start = time.monotonic()
    failure = None
    group = None
    lock_handle = None
    try:
        ready(state, root)
        if record["status"] != "reserved" or record["spec_sha256"] != digest(spec):
            raise ValueError("attempt is not an unstarted matching reservation")
        if record["runtime_sha256"] != runtime_identity(root):
            raise ValueError("reserved runner content changed")
        parent = properties(SLICE)
        check_parent(parent)
        own = properties(str(record["unit"]))
        expected = f"{parent['ControlGroup']}/{record['unit']}"
        if (
            own.get("ControlGroup") != expected
            or expected not in Path("/proc/self/cgroup").read_text().splitlines()[0]
        ):
            raise RuntimeError("worker is outside its reserved service cgroup")
        group = Path("/sys/fs/cgroup") / expected.lstrip("/")
        if (group / "memory.oom.group").read_text().strip() != "1":
            raise RuntimeError("worker cgroup does not kill the entire group on OOM")
        initial_oom = memory_events(group).get("oom_kill", 0)
        if spec["kind"] == "gpu":
            lock_handle = (state / "gpu.lock").open("a")
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_files(root, spec["code"] + spec["inputs"])
        if signals or (state / "PAUSE.json").exists():
            raise RuntimeError("execution cancelled during preflight")
        if spec["kind"] == "gpu" and device_memory() >= spec["limits"]["gpu_stop_bytes"]:
            raise RuntimeError("device over operational threshold before execution")
        argv = [
            interpreter if part == "{python}" and index == 0 else part
            for index, part in enumerate(spec["argv"])
        ]
        env = worker_environment(spec)
        env["PYTHONPATH"] = str(root / "src")
        record.update(
            status="running",
            started_unix=time.time(),
            argv=argv,
            heartbeat_unix=time.time(),
            parent_properties=parent,
            unit_properties=own,
            environment=env,
            peak_observed_vram_bytes=0,
            peak_ram_bytes=0,
        )
        atomic_json(record_path, record)
        notify_ready()
        start = time.monotonic()
        log_path = state / "logs" / f"{attempt}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("xb") as log:
            child = subprocess.Popen(
                argv,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while child.poll() is None:
                elapsed = time.monotonic() - start
                if signals or (state / "PAUSE.json").exists():
                    raise RuntimeError(f"interrupted or campaign paused: {signals}")
                if elapsed >= spec["limits"]["timeout_seconds"]:
                    raise TimeoutError("attempt execution timeout")
                observed = device_memory() if spec["kind"] == "gpu" else 0
                ram = int((group / "memory.current").read_text())
                counters = memory_events(group)
                record.update(
                    heartbeat_unix=time.time(),
                    elapsed_seconds=elapsed,
                    peak_observed_vram_bytes=max(
                        int(str(record["peak_observed_vram_bytes"])), observed
                    ),
                    peak_ram_bytes=max(int(str(record["peak_ram_bytes"])), ram),
                    memory_events=counters,
                )
                atomic_json(record_path, record)
                if observed and observed >= spec["limits"]["gpu_stop_bytes"]:
                    raise RuntimeError("observed VRAM operational threshold exceeded")
                if counters.get("oom_kill", 0) > initial_oom:
                    raise RuntimeError("cgroup OOM kill observed")
                check_parent(properties(SLICE))
                time.sleep(0.5)
            if child.returncode != 0:
                raise RuntimeError(f"command exited {child.returncode}")
            record["artifacts"] = [content_file(root, name) for name in spec["artifacts"]]
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        if signals:
            failure = f"interrupted by signal(s): {signals}; {failure}"
        if child is not None:
            try:
                stop_child(child)
            except (OSError, subprocess.SubprocessError) as error:
                failure = f"{failure}; teardown: {error}"
        elapsed = time.monotonic() - accounting_start
        record.update(
            status="finished" if failure is None else "failed",
            failure=failure,
            exit_code=None if child is None else child.returncode,
            elapsed_seconds=elapsed,
            charged_seconds=elapsed,
            finished_unix=time.time(),
            signals=signals,
        )
        if group is not None:
            try:
                record["final_memory_events"] = memory_events(group)
            except OSError as error:
                failure = f"{failure}; final cgroup observation: {error}"
                record.update(status="failed", failure=failure)
        with locked(state):
            atomic_json(record_path, record)
            if failure is not None:
                pause_locked(state, failure, attempt)
        if lock_handle is not None:
            lock_handle.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0 if failure is None else 1


def reconcile(state: Path) -> None:
    """Preserve terminal systemd evidence and conservatively charge lost monitors."""
    with locked(state):
        for record in attempts(state):
            if record["status"] not in ACTIVE | {"launch_uncertain"}:
                continue
            value = properties(str(record["unit"]))
            if value.get("ActiveState") in {"active", "activating", "deactivating"}:
                continue
            record.update(
                status="lost",
                finished_unix=time.time(),
                systemd_terminal=value,
                charged_seconds=max(
                    float(str(record["elapsed_seconds"])), float(str(record["reserved_seconds"]))
                ),
                failure="service ended without a trusted terminal worker record",
            )
            atomic_json(state / "attempts" / f"{record['id']}.json", record)
            pause_locked(state, str(record["failure"]), str(record["id"]))
