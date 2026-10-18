from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

GPU_UUIDS = (
    "GPU-daf615f5-ea55-860a-bfe1-41c36ab96c3e",
    "GPU-25ae492c-db98-e1ae-0f5e-c56365a422c7",
    "GPU-df6fe7d0-64e5-7af2-467d-c1699cb6fc67",
    "GPU-76951d88-add3-0d6b-5bee-c0390915ca47",
    "GPU-e6d65cb1-aba4-14f5-d4e5-bf24ef1ee3d8",
    "GPU-1fc4d921-dc78-6711-c2a8-ec0589b7f106",
    "GPU-2913ac69-eccb-63b0-1112-e09ea9598b57",
    "GPU-be85f55c-71a1-f389-eeed-55c63a0e3ce7",
)
ARMS = ("baseline", "moments", "dictionary", "combined")
CONTROLS = ("all", "moments_only", "dictionary_only", "none")
JOB_FIELDS = (
    "id",
    "pair_index",
    "fold",
    "seed",
    "arm",
    "control",
    "gpu_index",
    "gpu_uuid",
    "lane",
    "cpus",
)


@dataclass(frozen=True)
class ProcIdentity:
    """Kernel identity needed to distinguish a live process from a reused PID."""

    pid: int
    state: str
    parent_pid: int
    process_group: int
    session: int
    start_ticks: int


def _object(value: object, label: str) -> dict[str, object]:
    """Require a JSON object at an untrusted state boundary."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return value


def _original_cpu_pair(gpu: int, lane: int) -> list[int]:
    """Reproduce PR15 v1's frozen CPU assignment for its two existing lanes."""
    start = 16 + (gpu % 4) * 8 + lane * 4 + (gpu // 4)
    return [start, start + 2]


def _expected_job(pair_index: int, arm_index: int) -> dict[str, object]:
    """Build one queue record using the frozen v1 panel mapping."""
    fold, seed = pair_index % 4, 1001 + pair_index // 4
    position = fold * 4 + arm_index
    gpu, lane = (position + pair_index // 4) % 8, position // 8
    arm, control = ARMS[arm_index], CONTROLS[arm_index]
    return {
        "id": f"pr15_b300_v1_{arm}_f{fold}_s{seed}",
        "pair_index": pair_index,
        "fold": fold,
        "seed": seed,
        "arm": arm,
        "control": control,
        "gpu_index": gpu,
        "gpu_uuid": GPU_UUIDS[gpu],
        "lane": lane,
        "cpus": _original_cpu_pair(gpu, lane),
    }


def load_panel(path: Path) -> tuple[dict[str, object], ...]:
    """Verify the persisted v1 panel against its exact 120-cell identity."""
    payload = _object(json.loads(path.read_text(encoding="utf-8")), "panel")
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != 120:
        raise ValueError("source panel must contain exactly 120 jobs")
    jobs = tuple(_object(row, "panel job") for row in raw_jobs)
    expected = tuple(
        _expected_job(pair_index, arm_index)
        for pair_index in range(30)
        for arm_index in range(4)
    )
    actual_by_id = {str(job.get("id")): job for job in jobs}
    if len(actual_by_id) != 120:
        raise ValueError("source panel contains duplicate or missing job IDs")
    if tuple(actual_by_id[str(job["id"])] for job in expected) != expected:
        raise ValueError("source panel differs from the frozen v1 cell/GPU mapping")
    return jobs


def read_started_events(
    path: Path, jobs: Iterable[dict[str, object]]
) -> dict[str, tuple[int, dict[str, object]]]:
    """Read fsynced v1 starts strictly; an incomplete ledger blocks takeover."""
    expected = {str(job["id"]): job for job in jobs}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("v1 event journal has a partial final record")
    starts: dict[str, tuple[int, dict[str, object]]] = {}
    for line_number, line in enumerate(raw.splitlines(), 1):
        event = _object(json.loads(line), f"event line {line_number}")
        if event.get("event") != "fit_started":
            continue
        job = _object(event.get("job"), f"fit_started job on line {line_number}")
        job_id, pid = job.get("id"), event.get("pid")
        if not isinstance(job_id, str) or type(pid) is not int or pid <= 0:
            raise ValueError(f"invalid fit_started identity on line {line_number}")
        if job_id not in expected or job != expected[job_id]:
            raise ValueError(f"fit_started event does not match frozen panel: {job_id}")
        if job_id in starts:
            raise ValueError(f"duplicate fit_started event for {job_id}")
        starts[job_id] = (pid, job)
    return starts


def claimed_jobs(
    jobs: Iterable[dict[str, object]],
    starts: dict[str, tuple[int, dict[str, object]]],
    old_work_root: Path,
) -> tuple[set[str], set[str]]:
    """Exclude every started cell and every pre-existing work directory."""
    panel_ids = {str(job["id"]) for job in jobs}
    if not set(starts) <= panel_ids:
        raise ValueError("start ledger includes cells outside the original panel")
    directories: set[str] = set()
    for job_id in panel_ids:
        work = old_work_root / job_id
        if work.is_symlink():
            raise ValueError(f"worker path is a symlink; manual review required: {work}")
        if work.exists():
            if not work.is_dir():
                raise ValueError(f"worker path is not a directory: {work}")
            directories.add(job_id)
    return set(starts) | directories, directories - set(starts)


def pending_by_gpu(
    jobs: Iterable[dict[str, object]], claimed: set[str]
) -> dict[int, list[dict[str, object]]]:
    """Keep original cell-to-GPU placement and return only never-claimed cells."""
    pending = {gpu: [] for gpu in range(8)}
    for job in jobs:
        job_id, gpu = str(job["id"]), int(job["gpu_index"])
        if job_id not in claimed:
            pending[gpu].append(job)
    if any(len(queue) > 15 for queue in pending.values()):
        raise ValueError("pending queue count exceeds the frozen 15 cells per GPU")
    return pending


def density4_cpu_pair(gpu: int) -> tuple[int, int]:
    """Assign the new lane to its disjoint 48-63 CPU pair, NUMA-interleaved."""
    if not 0 <= gpu < 8:
        raise ValueError(f"invalid GPU index: {gpu}")
    first = 48 + (gpu % 4) * 4 + (gpu // 4)
    return first, first + 2


def slot_table(
    jobs: Iterable[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """Build 24 PR15 slots while retaining the v1 placements for lanes 0-1."""
    slot_cpus: dict[tuple[int, int], list[int]] = {}
    for job in jobs:
        key = (int(job["gpu_index"]), int(job["lane"]))
        cpus = job["cpus"]
        if not isinstance(cpus, list) or key in slot_cpus and slot_cpus[key] != cpus:
            raise ValueError(f"inconsistent original CPU slot: {key}")
        slot_cpus[key] = cpus
    slots: list[dict[str, object]] = []
    for gpu in range(8):
        for lane in range(3):
            cpus = slot_cpus[(gpu, lane)] if lane < 2 else list(density4_cpu_pair(gpu))
            slots.append(
                {"gpu_index": gpu, "gpu_uuid": GPU_UUIDS[gpu], "lane": lane, "cpus": cpus}
            )
    flattened = [cpu for slot in slots for cpu in slot["cpus"]]
    if len(slots) != 24 or len(flattened) != len(set(flattened)):
        raise ValueError("successor CPU slots are not disjoint")
    if set(flattened) != set(range(16, 64)):
        raise ValueError("successor CPU slot union must be exactly 16-63")
    return tuple(slots)


def campaign_deadline(
    launch_time_unix: float,
    wall_cap_seconds: int = 120 * 3600,
    parent_deadline_unix: float | None = None,
) -> float:
    """Return the inherited absolute deadline without resetting the v1 wall cap."""
    if launch_time_unix <= 0 or wall_cap_seconds <= 0:
        raise ValueError("launch time and wall cap must be positive")
    deadline = launch_time_unix + wall_cap_seconds
    return min(deadline, parent_deadline_unix) if parent_deadline_unix else deadline


def can_start_cell(
    deadline_unix: float,
    now_unix: float,
    forecast_seconds: float,
    per_fit_cap_seconds: float = 12 * 3600,
) -> bool:
    """Require the conservative fit projection and original 12-hour cap to fit."""
    if forecast_seconds <= 0 or per_fit_cap_seconds <= 0:
        raise ValueError("fit forecast and per-fit cap must be positive")
    return deadline_unix - now_unix >= max(forecast_seconds, per_fit_cap_seconds)


def parse_proc_stat(text: str, pid: int) -> ProcIdentity:
    """Parse Linux proc stat fields while tolerating spaces and ')' in comm."""
    right = text.rfind(")")
    left = text.find("(")
    if left < 0 or right <= left or not text[:left].strip().isdigit():
        raise ValueError("malformed /proc/PID/stat record")
    actual_pid = int(text[:left].strip())
    fields = text[right + 1 :].split()
    if actual_pid != pid or len(fields) <= 19 or len(fields[0]) != 1:
        raise ValueError("/proc/PID/stat PID or field count is invalid")
    return ProcIdentity(
        pid=pid,
        state=fields[0],
        parent_pid=int(fields[1]),
        process_group=int(fields[2]),
        session=int(fields[3]),
        start_ticks=int(fields[19]),
    )


def capture_pidfd(
    pid: int, proc_root: Path = Path("/proc")
) -> tuple[int, ProcIdentity]:
    """Open a stable process handle and reject a PID reused during inspection."""
    if not hasattr(os, "pidfd_open"):
        raise OSError("Linux pidfd_open is required for PR15 worker adoption")
    before = parse_proc_stat((proc_root / str(pid) / "stat").read_text(), pid)
    pidfd = os.pidfd_open(pid)
    try:
        after = parse_proc_stat((proc_root / str(pid) / "stat").read_text(), pid)
        if before.start_ticks != after.start_ticks:
            raise ProcessLookupError(f"PID {pid} changed identity during pidfd capture")
    except BaseException:
        os.close(pidfd)
        raise
    return pidfd, after


def sha256_bytes(payload: bytes) -> str:
    """Hash a captured command line without retaining its potentially sensitive text."""
    return hashlib.sha256(payload).hexdigest()
