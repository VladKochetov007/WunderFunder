"""Frozen R01-v4 packet, panel, journal, and progress validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

R01_PANEL_SHA256 = "471f465cc1d13fc6e5153a4c998390bb14e06823fed4c6721f929ce95fee47fb"
R01_MANIFEST_SHA256 = "95f82d442eefe425dde6574222b037ba1deba2d3ae36534aeec41e31e450874f"
R01_EXECUTE_SHA256 = "67a74b63991054020baed9ed15315d489dc1a3b94c25e78ab4a9158887df19d5"
R01_ROLLING_SHA256 = "818e5882e29548623019a3e97b8142ff154e3ee150c17145d8c67f7291c970fd"
R01_DECISION_SHA256 = "32a9f0aaee4a0a839468c618853ec441d65e5bce91860a5b202454e931b235d7"
R01_POLICY_SHA256 = "a320e92eed882f325922f7d20ed1e4b4aac5b0a3cc32c52031e81420f7a3aef6"
FIT_CAP_SECONDS = 12 * 3600
CAMPAIGN_CAP_SECONDS = 72 * 3600
OLD_UNIT = "tabdata-r01-execution-v4"
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


def sha256_file(path: Path) -> str:
    """Hash one regular non-symlink file without loading it all into memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read a regular JSON object and reject symbolic-link inputs."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def verify_frozen_packet(packet: Path) -> dict[str, Any]:
    """Verify the actual deployed R01 v4 packet and every manifest entry."""
    pins = {
        "EXECUTION_MANIFEST.json": R01_MANIFEST_SHA256,
        "PANEL.json": R01_PANEL_SHA256,
        "DECISION_CONTRACT.json": R01_DECISION_SHA256,
        "ROLLING_POLICY.json": R01_POLICY_SHA256,
        "execute.py": R01_EXECUTE_SHA256,
        "control/rolling.py": R01_ROLLING_SHA256,
    }
    for relative, expected in pins.items():
        if sha256_file(packet / relative) != expected:
            raise ValueError(f"R01-v4 frozen pin mismatch: {relative}")
    manifest = read_object(packet / "EXECUTION_MANIFEST.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("R01-v4 manifest has no file map")
    missing: list[str] = []
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("R01-v4 manifest file entries must be string hashes")
        path = packet / relative
        if path.is_symlink():
            raise ValueError(f"R01-v4 packet path is a symlink: {relative}")
        if not path.is_file():
            missing.append(relative)
        elif sha256_file(path) != expected:
            raise ValueError(f"R01-v4 manifest hash mismatch: {relative}")
    if missing:
        raise FileNotFoundError(f"R01-v4 packet is incomplete: {missing}")
    return manifest


def _flag_is(argv: list[Any], flag: str, value: str) -> bool:
    """Require an exact two-token flag/value pair in a frozen trainer argv."""
    return any(argv[index] == flag and argv[index + 1] == value for index in range(len(argv) - 1))


def validate_panel(panel: dict[str, Any], root: Path) -> tuple[dict[str, Any], ...]:
    """Require the exact paired 60-cell R01 panel and pinned device placement."""
    jobs = panel.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 60:
        raise ValueError("R01-v4 panel must contain exactly 60 jobs")
    pairs: dict[tuple[int, int], set[str]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    gpu_base: dict[int, tuple[str, tuple[int, ...]]] = {}
    for raw in jobs:
        if not isinstance(raw, dict):
            raise ValueError("R01 panel job must be an object")
        job = raw
        job_id = job.get("id")
        if not isinstance(job_id, str) or job_id in by_id:
            raise ValueError("R01 job IDs must be unique strings")
        if job.get("arm") not in {"wide", "deep"}:
            raise ValueError(f"unknown R01 arm: {job_id}")
        if type(job.get("fold")) is not int or job["fold"] not in range(4):
            raise ValueError(f"invalid fold: {job_id}")
        if type(job.get("seed")) is not int or not 1003 <= job["seed"] <= 1010:
            raise ValueError(f"invalid registered seed: {job_id}")
        gpu = job.get("gpu_index")
        if type(gpu) is not int or gpu not in range(8) or job.get("gpu_uuid") != GPU_UUIDS[gpu]:
            raise ValueError(f"R01 job GPU placement differs from the frozen host: {job_id}")
        run_dir = Path(str(job.get("run_dir", "")))
        argv, cpus = job.get("argv"), job.get("cpu_ids_initial")
        if not run_dir.is_absolute() or not isinstance(argv, list) or not argv:
            raise ValueError(f"R01 job lacks absolute run path or argv: {job_id}")
        if (
            not isinstance(cpus, list)
            or len(cpus) != 2
            or any(type(cpu) is not int for cpu in cpus)
        ):
            raise ValueError(f"R01 job must have two integer CPUs: {job_id}")
        if run_dir.parent.resolve() != (root / "runs").resolve() or run_dir.name != job_id:
            raise ValueError(f"R01 run path differs from the frozen root: {job_id}")
        expected_args = {
            "--arm": str(job["arm"]),
            "--fold": str(job["fold"]),
            "--seed": str(job["seed"]),
            "--run-name": job_id,
            "--mode": "fit",
        }
        if any(not _flag_is(argv, flag, value) for flag, value in expected_args.items()):
            raise ValueError(f"R01 frozen actor argv differs from panel identity: {job_id}")
        base = (str(job["gpu_uuid"]), tuple(cpus))
        if gpu in gpu_base and gpu_base[gpu] != base:
            raise ValueError(f"R01 jobs on GPU {gpu} do not share one frozen base CPU pair")
        gpu_base[gpu] = base
        pairs.setdefault((job["fold"], job["seed"]), set()).add(job["arm"])
        by_id[job_id] = job
    if len(pairs) != 30 or any(arms != {"wide", "deep"} for arms in pairs.values()):
        raise ValueError("R01 panel must contain 30 exact paired wide/deep cells")
    if set(gpu_base) != set(range(8)):
        raise ValueError("R01 panel does not use all eight pinned GPUs")
    return tuple(jobs)


def parse_journal(path: Path, jobs: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Parse all frozen events; any per-cell event claims that cell permanently."""
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("R01 event journal has a partial final record")
    known = {str(job["id"]): job for job in jobs}
    referenced: set[str] = set()
    starts: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError(f"R01 event line {line_number} is not an object")
        job_id = event.get("job_id")
        if job_id is not None:
            if not isinstance(job_id, str) or job_id not in known:
                raise ValueError(f"R01 event has an unknown job ID on line {line_number}")
            referenced.add(job_id)
        if event.get("event") == "fit_started":
            pid = event.get("pid")
            if type(pid) is not int or pid <= 0 or job_id in starts:
                raise ValueError(f"invalid or duplicate fit_started on line {line_number}")
            job = known[job_id]
            expected = (job["gpu_uuid"], job["fold"], job["seed"], job["arm"])
            observed = (
                event.get("gpu_uuid"),
                event.get("fold"),
                event.get("seed"),
                event.get("arm"),
            )
            if observed != expected:
                raise ValueError(f"fit_started metadata differs from panel: {job_id}")
            starts[job_id] = event
        if event.get("event") in {
            "fit_failed",
            "fit_admission_failed",
            "delivery_failed",
            "execution_failed",
        }:
            failures.append(event)
        events.append(event)
    return {
        "starts": starts,
        "referenced": referenced,
        "failures": failures,
        "events": events,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def progress_state(path: Path, jobs: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Recover per-cell progress and historical stop flags from the frozen controller."""
    if not path.is_file():
        return {"claimed": set(), "pending": None, "stop": False}
    progress = read_object(path)
    records, grades = progress.get("fits"), progress.get("grades")
    pending = progress.get("pending")
    if (
        not isinstance(records, list)
        or not isinstance(grades, list)
        or type(pending) is not int
        or not 0 <= pending <= len(jobs)
        or type(progress.get("new_admissions_stopped", False)) is not bool
    ):
        raise ValueError("R01 PROGRESS.json has malformed fields")
    known = {str(job["id"]) for job in jobs}
    claimed: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("R01 progress fit record must be an object")
        job_id = record.get("job_id")
        if job_id is not None:
            if not isinstance(job_id, str) or job_id not in known:
                raise ValueError(f"R01 PROGRESS.json references unknown cell: {job_id}")
            claimed.add(job_id)
    for grade in grades:
        if (
            not isinstance(grade, dict)
            or type(grade.get("fold")) is not int
            or grade["fold"] not in range(4)
        ):
            raise ValueError("R01 progress grade record is malformed")
    return {
        "claimed": claimed,
        "pending": pending,
        "stop": progress.get("new_admissions_stopped", False),
    }
