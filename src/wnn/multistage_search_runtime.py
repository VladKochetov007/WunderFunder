from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from wnn.multistage_search import atomic_write_json, file_lock
from wnn.multistage_search_budget import utc_now


def process_identity(pid: int) -> dict[str, int | str] | None:
    """Capture Linux process birth, PID, and process-group identity from procfs."""
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return {"pid": pid, "pgid": int(fields[2]), "birth": fields[19]}
    except ValueError:
        return None


def is_same_process(identity: Mapping[str, object]) -> bool:
    """Check that PID, kernel birth counter, and process group still match."""
    pid, expected_pgid, birth = identity.get("pid"), identity.get("pgid"), identity.get("birth")
    if type(pid) is not int or type(expected_pgid) is not int or not isinstance(birth, str):
        return False
    return process_identity(pid) == {"pid": pid, "pgid": expected_pgid, "birth": birth}


def terminate_owned_process(identity: Mapping[str, object], grace_seconds: float = 10.0) -> bool:
    """Stop only a process group whose recorded PID, birth, and PGID still match."""
    if not is_same_process(identity):
        return False
    pgid = identity["pgid"]
    if type(pgid) is not int:
        return False
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        if not is_same_process(identity):
            return True
        time.sleep(0.1)
    if is_same_process(identity):
        os.killpg(pgid, signal.SIGKILL)
    return True


@contextlib.contextmanager
def hold_gpu_slot(run_dir: Path, gpu_key: str) -> Iterator[None]:
    """Hold one admitted GPU slot for one worker without touching MPS or other jobs."""
    with file_lock(run_dir / "gpu_slots" / f"gpu-{gpu_key}.lock", blocking=False):
        yield


def launch_detached_worker(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    receipt_path: Path,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Start one independent one-shot stage worker and persist its exact process owner."""
    if os.name != "posix":
        raise RuntimeError("detached B300 workers must be launched on the POSIX receiver")
    command = list(argv)
    command_digest = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    start = time.time()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.with_suffix(".log").open("ab") as log_stream:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    identity = None
    for _ in range(50):
        identity = process_identity(proc.pid)
        if identity is not None or proc.poll() is not None:
            break
        time.sleep(0.02)
    if identity is None and proc.poll() is None:
        proc.kill()
        proc.wait()
        raise RuntimeError("could not establish detached worker PID/birth/PGID")
    record: dict[str, object] = {
        "schema": "wnn.multistage-worker-owner.v1",
        "pid": proc.pid,
        "pgid": proc.pid,
        "birth": None if identity is None else identity["birth"],
        "command": command,
        "command_sha256": command_digest,
        "started_at_utc": utc_now(),
        "started_at_epoch": start,
        "cwd": str(cwd.resolve()),
        "status": "running" if proc.poll() is None else "exited_before_identity_capture",
    }
    if metadata is not None:
        record.update(dict(metadata))
    try:
        atomic_write_json(receipt_path, record, replace=False)
    except (OSError, TypeError, ValueError):
        if identity is not None:
            terminate_owned_process(identity, grace_seconds=0.0)
        elif proc.poll() is None:
            proc.kill()
        proc.wait()
        raise
    if proc.poll() is not None:
        record["exit_code"] = proc.returncode
        atomic_write_json(receipt_path, record)
    return record


def current_process_owner() -> dict[str, int | str]:
    """Return this controller's PID/birth/PGID tuple for child ownership records."""
    identity = process_identity(os.getpid())
    if identity is None:
        raise RuntimeError("controller process birth identity is unavailable")
    return identity


def base_environment(gpu_value: str, source_root: Path | None = None) -> dict[str, str]:
    """Build bounded remote worker env while preserving strict trainer precision controls."""
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_value,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        }
    )
    if source_root is not None:
        source_path = (source_root.resolve() / "src").as_posix()
        inherited = env.get("PYTHONPATH", "")
        retained = [
            item
            for item in inherited.split(os.pathsep)
            if item and Path(item).resolve().as_posix() != source_path
        ]
        env["PYTHONPATH"] = os.pathsep.join([source_path, *retained])
    return env
