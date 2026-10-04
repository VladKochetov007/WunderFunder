"""Durable, resource-limited campaign jobs with GPU wall-time accounting."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research" / "campaign_2026-09-29"
JOBS = CAMPAIGN / "jobs"
GPU_CAP = 400_000.0  # 2026-10-02 15:55: user authorised continued research (8 CPU / 5 GB VRAM / 14 GB RAM enforced by the slice and WNN_VRAM_GB); 72,000 s cap silently stopped the queue at 15:06
VRAM_CAP = int(os.environ.get("WNN_CAMPAIGN_VRAM_BYTES", "5000000000"))
if not 0 < VRAM_CAP <= 6_000_000_000:
    raise ValueError("observed VRAM limit must be positive and at most 6 GB")
STOP_NEW_JOBS = CAMPAIGN / "STOP_NEW_JOBS"
CONTINUATION = ROOT / "research" / "campaign_2026-10-03" / "continuation"
QUEUE_PAUSE = ROOT / "research" / "campaign_2026-10-01" / "QUEUE_PAUSE"
GPU_SLICE = "wnn-campaign-gpu.slice"
ADMISSION_TIMEOUT = 900.0


def request_stop(reason: str) -> None:
    """Persist the launch barrier until an intentional campaign resumption."""
    try:
        with STOP_NEW_JOBS.open("x") as marker:
            marker.write(f"{timestamp()}: {reason}\n")
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError:
        pass


def check_launch_allowed(*, continuation: bool = False) -> None:
    """Reject launches and delayed service starts while the barrier exists."""
    if STOP_NEW_JOBS.exists():
        raise RuntimeError(f"campaign launch blocked by {STOP_NEW_JOBS.relative_to(ROOT)}")
    if continuation:
        if not (CONTINUATION / "READY").is_file():
            raise RuntimeError("continuation admission is not ready")
        for marker in (CONTINUATION / "RESOURCE_FAILURE", QUEUE_PAUSE):
            if marker.exists():
                raise RuntimeError(f"continuation launch blocked by {marker.relative_to(ROOT)}")


def continuation_job(args: argparse.Namespace) -> bool:
    """Gate future GPU jobs and explicit continuation CPU entrypoints."""
    prefixes = (f"{CONTINUATION}/", f"{CONTINUATION.relative_to(ROOT)}/")
    return args.kind == "gpu" or any(part.startswith(prefixes) for part in args.command)


def timestamp() -> str:
    """Return a UTC timestamp for auditable job records."""
    return datetime.now(UTC).isoformat()


def write_record(path: Path, record: dict) -> None:
    """Atomically publish a job's latest durable accounting record."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n")
    temp.replace(path)


def records() -> list[dict]:
    """Read accounting records without discarding malformed evidence."""
    return [json.loads(path.read_text()) for path in sorted(JOBS.glob("*.json"))]


def source_fingerprints() -> dict[str, str]:
    """Record the exact research/runtime source bytes seen when a job starts."""
    paths = [
        *sorted((ROOT / "src" / "wnn").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "submission").glob("*.py")),
        *sorted((ROOT / "native" / "src").glob("*.rs")),
        ROOT / "scripts" / "limited",
        ROOT / "scripts" / "ft_gclp.sh",
        ROOT / "utils.py",
        *sorted(CONTINUATION.glob("*.sh")),
        ROOT / "research" / "campaign_2026-10-01" / "jobs" / "dcurve_fold.sh",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }


def slice_evidence(slice_name: str) -> dict[str, str]:
    """Capture effective family limits; fail closed if inspection fails."""
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            slice_name,
            "--property=CPUQuotaPerSecUSec,MemoryMax,MemorySwapMax,ControlGroup",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines())
    if slice_name == GPU_SLICE and (
        properties.get("CPUQuotaPerSecUSec") != "8s"
        or properties.get("MemoryMax") != str(24 * 1024**3)
        or properties.get("MemorySwapMax") != "0"
        or not properties.get("ControlGroup", "").endswith(f"/{GPU_SLICE}")
    ):
        raise RuntimeError(f"GPU family limits not confirmed: {properties}")
    return properties


def notify_ready() -> None:
    """Start systemd's execution lifetime only after bounded admission."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        raise RuntimeError("timed execution requires a systemd notify service")
    if address.startswith("@"):
        address = "\0" + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notifier:
        notifier.settimeout(5)
        notifier.connect(address)
        notifier.sendall(b"READY=1")


def charged_seconds(record: dict) -> float:
    """Conservatively account jobs whose monitor ended unexpectedly."""
    elapsed = float(record.get("elapsed_seconds", 0))
    if record["status"] == "running":
        active = unit_active(record["unit"])
        if active:
            elapsed = max(elapsed, time.time() - record["started_unix"])
        else:
            elapsed = max(elapsed + 5, float(record.get("timeout_seconds", elapsed)) + 15)
    return elapsed


def budget() -> dict:
    """Return cumulative GPU-job wall time, including failures and active work."""
    jobs = records()
    used = sum(charged_seconds(job) for job in jobs if job["kind"] == "gpu")
    return {
        "gpu_seconds_used": used,
        "gpu_seconds_remaining": max(0.0, GPU_CAP - used),
        "gpu_seconds_cap": GPU_CAP,
        "jobs": jobs,
    }


def device_memory() -> int:
    """Read total observed device memory, including non-PyTorch allocations."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return max(int(line.strip()) for line in result.stdout.splitlines()) * 1024**2


def unit_active(unit: str) -> bool:
    """Check a known campaign unit with a bounded systemctl call."""
    return (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            check=False,
            timeout=5,
        ).returncode
        == 0
    )


def stop_scope(unit: str) -> None:
    """Stop a campaign-owned scope, then confirm it has exited."""
    if unit_active(unit):
        subprocess.run(["systemctl", "--user", "stop", unit], check=True, timeout=10)
    if unit_active(unit):
        raise RuntimeError(f"campaign scope remains active: {unit}")


def stop_child(child: subprocess.Popen) -> None:
    """Reap the leader and stop its group, including surviving descendants."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=5)
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            child.wait(timeout=2)
            return
        time.sleep(0.1)
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=2)


def run_job(args: argparse.Namespace) -> int:
    """Run one job under a durable service and preserve failure evidence."""
    received_signals: list[int] = []

    def on_signal(signum: int, frame: FrameType | None) -> None:
        received_signals.append(signum)
        request_stop(f"{args.name} received {signal.Signals(signum).name}")

    previous_handlers = {
        signum: signal.signal(signum, on_signal) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    JOBS.mkdir(parents=True, exist_ok=True)
    record_path = JOBS / f"{args.name}.json"
    if record_path.exists():
        raise ValueError(f"job evidence already exists: {record_path}")
    admission_start = time.monotonic()
    record = {
        "name": args.name,
        "unit": args.unit,
        "scope_unit": f"wnn-top1-{args.name}-scope.scope",
        "phase": args.phase,
        "kind": args.kind,
        "command": args.command,
        "admission_started_at": timestamp(),
        "admission_timeout_seconds": args.admission_timeout,
        "timeout_seconds": args.timeout,
        "status": "waiting",
        "elapsed_seconds": 0,
        "peak_observed_vram_bytes": 0,
        "observed_vram_cap_bytes": VRAM_CAP,
        "source_sha256": source_fingerprints(),
    }
    write_record(record_path, record)
    log_path = CAMPAIGN / "logs" / f"{args.name}.log"
    start = None
    lock = None
    child = None
    failure = None
    try:
        check_launch_allowed(continuation=continuation_job(args))
        lock = (CAMPAIGN / "gpu.lock").open("a+")
        if args.kind == "gpu":
            while True:
                check_launch_allowed(continuation=True)
                if received_signals:
                    raise RuntimeError("admission interrupted")
                if time.monotonic() - admission_start >= args.admission_timeout:
                    raise TimeoutError("GPU admission deadline exceeded")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.5)
            for old in records():
                scope = old.get("scope_unit")
                if old["kind"] == "gpu" and scope and unit_active(scope):
                    raise RuntimeError(f"previous GPU scope remains active: {scope}")
        check_launch_allowed(continuation=continuation_job(args))
        remaining = budget()["gpu_seconds_remaining"]
        allowance = min(args.timeout, remaining - 15) if args.kind == "gpu" else args.timeout
        if allowance <= 0:
            raise ValueError("GPU budget exhausted")
        family = GPU_SLICE if args.kind == "gpu" else "wnn-campaign.slice"
        if os.environ.get("WNN_SLICE") != family:
            raise RuntimeError("execution scope must inherit the monitor's family slice")
        record["runtime_provenance"] = {
            "slice": family,
            "effective_slice_properties": slice_evidence(family),
            "environment": {
                key: value
                for key, value in sorted(os.environ.items())
                if key.startswith(
                    (
                        "WNN_",
                        "OMP_",
                        "MKL_",
                        "OPENBLAS_",
                        "NUMBA_",
                        "POLARS_",
                        "ARROW_",
                        "CUBLAS_",
                        "CUDA_",
                        "PYTORCH_",
                    )
                )
                or key == "PYTHONHASHSEED"
            },
        }
        record["source_sha256"] = source_fingerprints()
        record.update(
            admission_elapsed_seconds=time.monotonic() - admission_start,
            started_at=timestamp(),
            started_unix=time.time(),
            timeout_seconds=allowance,
            status="running",
        )
        write_record(record_path, record)
        notify_ready()
        start = time.monotonic()
        with log_path.open("x") as log:
            if received_signals:
                failure = f"signal:{signal.Signals(received_signals[0]).name}"
            else:
                check_launch_allowed(continuation=continuation_job(args))
            if failure is not None:
                return 1
            child = subprocess.Popen(
                args.command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while child.poll() is None:
                elapsed = time.monotonic() - start
                if received_signals:
                    failure = f"signal:{signal.Signals(received_signals[0]).name}"
                    break
                if args.kind == "gpu":
                    observed = device_memory()
                    record["peak_observed_vram_bytes"] = max(
                        record["peak_observed_vram_bytes"], observed
                    )
                    if observed > VRAM_CAP:
                        failure = "observed_vram_limit"
                        break
                record["elapsed_seconds"] = elapsed
                write_record(record_path, record)
                if elapsed >= allowance:
                    failure = "time_budget_limit"
                    break
                time.sleep(0.5)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        failure = repr(exc)
    finally:
        if received_signals:
            failure = f"signal:{signal.Signals(received_signals[0]).name}"
        if child is not None:
            try:
                stop_child(child)
            except (OSError, subprocess.SubprocessError) as exc:
                failure = f"{failure}; child_stop={exc!r}"
        if start is not None:
            try:
                stop_scope(record["scope_unit"])
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                failure = f"{failure}; scope_stop={exc!r}"
        record.update(
            status=(
                "interrupted"
                if received_signals
                else "finished"
                if failure is None and child is not None and child.returncode == 0
                else "failed"
            ),
            finished_at=timestamp(),
            elapsed_seconds=0 if start is None else time.monotonic() - start,
            admission_elapsed_seconds=record.get(
                "admission_elapsed_seconds", time.monotonic() - admission_start
            ),
            exit_code=None if child is None else child.returncode,
            failure=failure,
            log=str(log_path.relative_to(ROOT)),
        )
        write_record(record_path, record)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if lock is not None:
            lock.close()
    return 0 if record["status"] == "finished" else 1


def launch(args: argparse.Namespace) -> None:
    """Launch a transient service that remains alive after the shell exits."""
    check_launch_allowed(continuation=continuation_job(args))
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        raise ValueError("job names may contain only letters, digits, underscore and dash")
    unit = f"wnn-top1-{args.name}"
    if (JOBS / f"{args.name}.json").exists():
        raise ValueError("use a new job name; prior evidence is immutable")
    remaining = budget()["gpu_seconds_remaining"]
    allowance = min(args.timeout, remaining - 15) if args.kind == "gpu" else args.timeout
    if allowance <= 0:
        raise ValueError("GPU budget exhausted")
    family = GPU_SLICE if args.kind == "gpu" else "wnn-campaign.slice"
    threads = min(args.threads, 8) if args.kind == "gpu" else args.threads
    if args.kind == "gpu":
        subprocess.run(
            [
                "systemctl",
                "--user",
                "set-property",
                "--runtime",
                GPU_SLICE,
                "CPUQuota=800%",
                "MemoryMax=24G",
                "MemorySwapMax=0",
            ],
            check=True,
            timeout=5,
        )
    env = {
        "WNN_SLICE": family,
        "WNN_CPU_PCT": "75",
        "WNN_MEM_MAX": "24G" if args.kind == "gpu" else "2G",
        "WNN_GPU_FRACTION": "0.70",
        "WNN_LOADER_RAM_GB": "3",
        "WNN_LOADER_WORKERS": "3",
        "WNN_THREADS": str(threads),
        "WNN_CPU_QUOTA": f"{threads * 100}%",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMBA_NUM_THREADS": str(threads),
        "POLARS_MAX_THREADS": str(threads),
        "ARROW_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "WNN_RUNTIME_MAX": str(math.ceil(allowance + 5)),
        "WNN_SCOPE_UNIT": f"wnn-top1-{args.name}-scope",
    }
    command = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        f"--slice={family}",
        "--no-block",
        "--quiet",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "Type=notify",
        "-p",
        "NotifyAccess=main",
        "-p",
        f"TimeoutStartSec={math.ceil(args.admission_timeout + 8)}",
        "-p",
        f"RuntimeMaxSec={math.ceil(allowance + 8)}",
        "-p",
        "TimeoutStopSec=15s",
        "-p",
        "KillMode=control-group",
        "-p",
        f"CPUQuota={threads * 100}%",
        "-p",
        f"MemoryMax={env['WNN_MEM_MAX']}",
    ]
    command.extend(f"--setenv={key}={value}" for key, value in env.items())
    if "WNN_CAMPAIGN_VRAM_BYTES" in os.environ:
        command.append(f"--setenv=WNN_CAMPAIGN_VRAM_BYTES={VRAM_CAP}")
    command.extend(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--unit",
            unit,
            "--name",
            args.name,
            "--kind",
            args.kind,
            "--phase",
            args.phase,
            "--timeout",
            str(args.timeout),
            "--admission-timeout",
            str(args.admission_timeout),
            "--threads",
            str(threads),
            "--",
            *args.command,
        ]
    )
    check_launch_allowed(continuation=continuation_job(args))
    subprocess.run(command, check=True, timeout=10)
    print(json.dumps({"unit": unit, "name": args.name, "command": args.command}))


def main() -> None:
    """Parse launcher, monitor and read-only accounting commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("launch", "run"):
        part = sub.add_parser(action)
        part.add_argument("--name", required=True)
        part.add_argument("--kind", choices=("gpu", "cpu"), required=True)
        part.add_argument("--phase", default="controls")
        part.add_argument("--timeout", type=float, required=True)
        part.add_argument("--admission-timeout", type=float, default=ADMISSION_TIMEOUT)
        part.add_argument("--threads", type=int, choices=range(1, 14), default=12)
        if action == "run":
            part.add_argument("--unit", required=True)
        part.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.action == "status":
        print(json.dumps(budget(), indent=2))
        return
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if (
        not args.command
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.admission_timeout)
        or args.admission_timeout <= 0
    ):
        parser.error("a command and positive execution/admission timeouts are required")
    if args.action == "launch":
        launch(args)
    else:
        raise SystemExit(run_job(args))


if __name__ == "__main__":
    main()
