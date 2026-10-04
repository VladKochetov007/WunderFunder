from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from wnn.campaign_spec import (
    BLOCK_SECONDS,
    CPU_MAX,
    CPU_SMALL_RAM_MAX,
    NAME,
    RAM_MAX,
    SUPPORT_CPU_RESERVE,
    SUPPORT_RAM_RESERVE,
    ExperimentSpec,
    digest,
    file_hash,
    validate_local,
    validate_spec,
)

ACTIVE = {"reserved", "running"}
PREFLIGHT_AND_TEARDOWN_SECONDS = 930
RUNTIME_FILES = (
    "src/wnn/campaign.py",
    "src/wnn/campaign_spec.py",
    "src/wnn/campaign_local.py",
    "src/wnn/resources.py",
    "scripts/run_campaign.py",
)


class AdmissionBusy(RuntimeError):
    """Normal queue backpressure; it must not pause an active campaign."""


class JobQuarantined(AdmissionBusy):
    """A reviewed job-specific acknowledgement is required before another attempt."""


class BudgetExhausted(RuntimeError):
    """The controller must review and renew the block before this admission."""


def read_json(path: Path) -> dict[str, object]:
    """Read durable state; malformed evidence blocks execution."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    """Publish a complete, flushed record across worker/controller crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as output:
        json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def locked(state: Path, name: str = "controller.lock") -> Iterator[None]:
    """Serialize admission and ledger updates across independent CLI calls."""
    state.mkdir(parents=True, exist_ok=True)
    with (state / name).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def runtime_identity(root: Path) -> dict[str, str]:
    """Bind execution machinery separately from the scientific source manifest."""
    return {name: file_hash(root / name) for name in RUNTIME_FILES}


def ready(state: Path, root: Path) -> dict[str, object]:
    """Require this campaign's explicit readiness and unchanged execution code."""
    if (state / "PAUSE.json").exists():
        raise RuntimeError("campaign is paused; diagnose the saved incident")
    value = read_json(state / "READY.json")
    if value["root"] != str(root.resolve()):
        raise ValueError("readiness belongs to a different checkout")
    if value["runtime_sha256"] != runtime_identity(root):
        raise ValueError("runner changed after readiness; issue new reviewed readiness")
    if file_hash(Path(str(value["contract"]))) != value["contract_sha256"]:
        raise ValueError("authorized contract changed after readiness")
    return value


def attempts(state: Path) -> list[dict[str, object]]:
    """Read every attempt, preserving malformed records as admission failures."""
    return [read_json(path) for path in sorted((state / "attempts").glob("*.json"))]


def initialize(state: Path, root: Path, contract: Path, block_id: str) -> None:
    """Create a new campaign, never removing old or historical pause markers."""
    if not NAME.fullmatch(block_id):
        raise ValueError("invalid block ID")
    with locked(state):
        if (state / "READY.json").exists() or attempts(state) or (state / "PAUSE.json").exists():
            raise FileExistsError("campaign already exists; use a new state directory")
        atomic_json(
            state / "blocks" / f"{block_id}.json",
            {
                "id": block_id,
                "cap_seconds": BLOCK_SECONDS,
                "opened_unix": time.time(),
                "review": "initial authorized 24 GPU-job-hour block",
            },
        )
        atomic_json(
            state / "READY.json",
            {
                "root": str(root.resolve()),
                "contract": str(contract.resolve()),
                "contract_sha256": file_hash(contract),
                "runtime_sha256": runtime_identity(root),
                "block_id": block_id,
                "issued_unix": time.time(),
            },
        )


def budget(state: Path, block_id: str) -> dict[str, float]:
    """Reserve entire active-job timeouts and charge all ended GPU attempts."""
    block = read_json(state / "blocks" / f"{block_id}.json")
    used = reserved = 0.0
    for item in attempts(state):
        if item["block_id"] != block_id or item["kind"] != "gpu":
            continue
        if item["status"] in ACTIVE:
            reserved += float(str(item["reserved_seconds"]))
        else:
            used += float(str(item["charged_seconds"]))
    cap = float(str(block["cap_seconds"]))
    if cap != BLOCK_SECONDS:
        raise ValueError("unexpected block cap")
    return {
        "cap_seconds": cap,
        "used_seconds": used,
        "reserved_seconds": reserved,
        "remaining_seconds": max(0.0, cap - used - reserved),
    }


def open_block(state: Path, root: Path, block_id: str, review: str) -> None:
    """Renew one bounded block after controller review, without resetting history."""
    if not NAME.fullmatch(block_id) or not review.strip():
        raise ValueError("new block requires a safe ID and recorded review")
    with locked(state):
        value = ready(state, root)
        if any(item["status"] in ACTIVE for item in attempts(state)):
            raise RuntimeError("cannot renew a block while attempts are active")
        path = state / "blocks" / f"{block_id}.json"
        if path.exists():
            raise FileExistsError("block already exists")
        atomic_json(
            path,
            {
                "id": block_id,
                "cap_seconds": BLOCK_SECONDS,
                "opened_unix": time.time(),
                "review": review,
                "previous_block": value["block_id"],
            },
        )
        atomic_json(state / "READY.json", {**value, "block_id": block_id})


def pause_locked(state: Path, reason: str, attempt: str | None = None) -> None:
    """Publish an incident while the caller holds the shared admission lock."""
    event = {"reason": reason, "attempt": attempt, "time_unix": time.time(), "scope": "global"}
    current = state / "PAUSE.json"
    if current.exists():
        event["primary_incident"] = read_json(current)["id"]
    event["id"] = digest(event)
    atomic_json(state / "incidents" / f"{event['id']}.json", event)
    if not current.exists():
        atomic_json(current, event)


def quarantine_locked(
    state: Path, spec: ExperimentSpec, attempt: str, reason: str, category: str
) -> None:
    """Record a contained scientific failure without interrupting other jobs."""
    event = {
        "scope": "job",
        "job": spec["name"],
        "spec_sha256": digest(spec),
        "attempt": attempt,
        "reason": reason,
        "category": category,
        "time_unix": time.time(),
    }
    event["id"] = digest(event)
    atomic_json(state / "incidents" / f"{event['id']}.json", event)
    path = state / "quarantines" / f"{spec['name']}.json"
    if not path.exists():
        atomic_json(path, event)


def quarantines(state: Path) -> list[dict[str, object]]:
    """Expose outstanding scientific failures independently of global readiness."""
    return [read_json(path) for path in sorted((state / "quarantines").glob("*.json"))]


def acknowledge_job(state: Path, root: Path, job: str, incident_id: str, review: str) -> None:
    """Release one diagnosed job quarantine without refreshing runtime readiness."""
    if not NAME.fullmatch(job):
        raise ValueError("invalid job identity")
    with locked(state):
        ready(state, root)
        path = state / "quarantines" / f"{job}.json"
        event = read_json(path)
        if event["job"] != job or event["id"] != incident_id or not review.strip():
            raise ValueError("acknowledgement must bind the job incident and diagnosis")
        if any(
            item["status"] in ACTIVE and item["spec"]["name"] == job for item in attempts(state)
        ):
            raise RuntimeError("reconcile this job's active attempts before acknowledgement")
        atomic_json(
            state / "acknowledgements" / f"{incident_id}.json",
            {
                "scope": "job",
                "job": job,
                "incident_id": incident_id,
                "review": review,
                "time_unix": time.time(),
                "runtime_sha256": runtime_identity(root),
            },
        )
        path.unlink()


def pause(state: Path, reason: str, attempt: str | None = None) -> None:
    """Retain every incident and close admission until reviewed acknowledgement."""
    with locked(state):
        pause_locked(state, reason, attempt)


def reserve_attempt(
    state: Path, root: Path, spec: ExperimentSpec, attempt: str
) -> dict[str, object]:
    """Reserve one unique attempt; caller holds the admission lock."""
    if not NAME.fullmatch(attempt):
        raise ValueError("invalid attempt ID")
    value = ready(state, root)
    quarantine = state / "quarantines" / f"{spec['name']}.json"
    if quarantine.exists():
        event = read_json(quarantine)
        raise JobQuarantined(f"job {spec['name']} is quarantined by incident {event['id']}")
    path = state / "attempts" / f"{attempt}.json"
    if path.exists():
        raise FileExistsError("attempt IDs cannot be reused")
    running = [item for item in attempts(state) if item["status"] in ACTIVE]
    validate_local(spec)
    capacity = 1 if spec["kind"] == "gpu" else 2
    if sum(item["kind"] == spec["kind"] for item in running) >= capacity:
        raise AdmissionBusy("all local worker slots are occupied")
    allocated = [spec, *(validate_spec(item["spec"]) for item in running)]
    for item in allocated:
        validate_local(item)
    if (
        sum(
            item["kind"] == "cpu" and item["limits"]["ram_bytes"] > CPU_SMALL_RAM_MAX
            for item in allocated
        )
        > 1
    ):
        raise AdmissionBusy("the single larger-memory CPU helper slot is occupied")
    worker_cpus = sum(item["limits"]["cpus"] for item in allocated)
    worker_ram = sum(item["limits"]["ram_bytes"] for item in allocated)
    if worker_cpus + SUPPORT_CPU_RESERVE > CPU_MAX or worker_ram + SUPPORT_RAM_RESERVE > RAM_MAX:
        raise AdmissionBusy("worker reservations would consume the aggregate support reserve")
    block_id = str(value["block_id"])
    reservation = spec["limits"]["timeout_seconds"] + PREFLIGHT_AND_TEARDOWN_SECONDS
    if spec["kind"] == "gpu" and budget(state, block_id)["remaining_seconds"] < reservation:
        raise BudgetExhausted("job timeout exceeds unreserved block allowance")
    record: dict[str, object] = {
        "id": attempt,
        "spec": spec,
        "spec_sha256": digest(spec),
        "runtime_sha256": value["runtime_sha256"],
        "block_id": block_id,
        "kind": spec["kind"],
        "unit": f"wnn-c04-{attempt}.service",
        "status": "reserved",
        "timeout_seconds": spec["limits"]["timeout_seconds"],
        "reserved_seconds": reservation,
        "reserved_unix": time.time(),
        "elapsed_seconds": 0.0,
        "charged_seconds": 0.0,
        "admission_resources": {
            "worker_cpus": worker_cpus,
            "worker_ram_bytes": worker_ram,
            "support_cpus": SUPPORT_CPU_RESERVE,
            "support_ram_bytes": SUPPORT_RAM_RESERVE,
            "parent_cpu_max": CPU_MAX,
            "parent_ram_max": RAM_MAX,
        },
    }
    atomic_json(path, record)
    return record


def acknowledge(state: Path, root: Path, incident_id: str, review: str) -> None:
    """Acknowledge a diagnosed incident; active workers must first be reconciled."""
    with locked(state):
        event = read_json(state / "PAUSE.json")
        if event["id"] != incident_id or not review.strip():
            raise ValueError("acknowledgement must bind the current incident and diagnosis")
        if any(item["status"] in ACTIVE for item in attempts(state)):
            raise RuntimeError("reconcile all active attempts before acknowledgement")
        value = read_json(state / "READY.json")
        if value["root"] != str(root.resolve()):
            raise ValueError("wrong campaign checkout")
        atomic_json(
            state / "acknowledgements" / f"{incident_id}.json",
            {
                "incident_id": incident_id,
                "review": review,
                "time_unix": time.time(),
                "runtime_sha256": runtime_identity(root),
            },
        )
        value["runtime_sha256"] = runtime_identity(root)
        atomic_json(state / "READY.json", value)
        (state / "PAUSE.json").unlink()
