from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from bandit_resources import atomic_json, locked, read_json, sha256_file

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
LANES = {"local_gpu", "kaggle_gpu", "kaggle_tpu", "vast"}


def job_digest(job: dict) -> str:
    """Bind admission to the entire frozen job, including its launch command."""
    payload = json.dumps(job, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def project_path(root: Path, relative: str) -> Path:
    """Require queue artifacts to remain within the authorized project root."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("queue artifact escapes project root")
    return path


def load_jobs(queue_path: Path) -> list[dict]:
    """Validate a finite queue before considering any external action."""
    queue = read_json(queue_path)
    if queue.get("schema_version") != 1 or not isinstance(queue.get("jobs"), list):
        raise ValueError("expected queue schema_version 1 with jobs")
    jobs = queue["jobs"]
    seen = set()
    for job in jobs:
        attempt = job["attempt_id"]
        if not IDENTIFIER.fullmatch(attempt) or attempt in seen:
            raise ValueError("invalid or duplicate attempt identifier")
        if job["lane"] not in LANES:
            raise ValueError("unknown compute lane")
        if any(parent not in seen for parent in job.get("predecessors", [])):
            raise ValueError("predecessors must name earlier jobs in this queue")
        seen.add(attempt)
    return jobs


def verified_completion(root: Path, job: dict) -> bool:
    """Require bound terminal evidence and every expected artifact's bytes."""
    receipt_name = job.get("completion_receipt")
    if not receipt_name:
        return False
    receipt_path = project_path(root, receipt_name)
    if not receipt_path.exists():
        return False
    receipt = read_json(receipt_path)
    if receipt.get("status") != "complete":
        return False
    if receipt.get("attempt_id") != job["attempt_id"] or receipt.get("job_sha256") != job_digest(
        job
    ):
        raise ValueError("completion receipt does not bind the frozen job")
    artifacts = receipt.get("artifacts", [])
    expected = set(job.get("expected_artifacts", []))
    if not expected or {item["name"] for item in artifacts} != expected:
        raise ValueError("terminal receipt does not contain exact expected artifacts")
    if len(artifacts) != len(expected):
        raise ValueError("duplicate terminal artifact")
    for item in artifacts:
        artifact = project_path(root, item["path"])
        if not artifact.is_file() or sha256_file(artifact) != item["sha256"]:
            raise ValueError("terminal artifact missing or hash mismatch")
    return job["lane"] != "vast" or receipt.get("teardown_verified") is True


def valid_admission(root: Path, job: dict, now: datetime) -> list[str]:
    """Check fresh root authority; provider-specific guards still own admission."""
    path = project_path(root, job["admission_receipt"])
    receipt = read_json(path)
    if (
        receipt.get("owner") != "root"
        or receipt.get("authorized") is not True
        or receipt.get("attempt_id") != job["attempt_id"]
        or receipt.get("job_sha256") != job_digest(job)
    ):
        raise ValueError("missing root admission bound to this exact job")
    created = datetime.fromisoformat(receipt["created_utc"])
    expires = datetime.fromisoformat(receipt["expires_utc"])
    if created.tzinfo is None or expires.tzinfo is None:
        raise ValueError("admission timestamps require a timezone")
    if not created <= now < expires or (expires - created).total_seconds() > 120:
        raise ValueError("admission must be current and valid for at most 120 seconds")
    source_hashes = receipt.get("source_hashes", {})
    if not source_hashes:
        raise ValueError("admission has no frozen sources")
    for name, digest in source_hashes.items():
        if sha256_file(project_path(root, name)) != digest:
            raise ValueError("admitted source has changed")
    command = job.get("launch_command")
    if not isinstance(command, list) or not command:
        raise ValueError("job has no guarded launch command")
    if any(not isinstance(part, str) or "\0" in part for part in command):
        raise ValueError("launch command must be a literal argument list")
    return command


def append_event(runtime: Path, event: dict) -> None:
    """Preserve an append-only transition journal before returning control."""
    with (runtime / "journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def check_runtime(root: Path, runtime: Path) -> None:
    """Reject state paths that leave the project through a symlink."""
    for path in [
        runtime,
        runtime / "attempts",
        runtime / "controller.lock",
        runtime / "journal.jsonl",
    ]:
        project_path(root, str(path))
    for path in (runtime / "attempts").glob("*.json"):
        project_path(root, str(path))


def terminal_record_verified(root: Path, record: dict) -> bool:
    """Recheck terminal evidence before releasing even an old omitted attempt."""
    try:
        if record["status"] == "complete":
            job = record["frozen_job"]
            receipt = project_path(root, job["completion_receipt"])
            return (
                job["attempt_id"] == record["attempt_id"]
                and job_digest(job) == record["job_sha256"]
                and sha256_file(receipt) == record["completion_sha256"]
                and verified_completion(root, job)
            )
        if record["status"] == "failed_collected":
            path = project_path(root, record["reconciliation_path"])
            receipt = read_json(path)
            return (
                sha256_file(path) == record["reconciliation_sha256"]
                and receipt["attempt_id"] == record["attempt_id"]
                and receipt["job_sha256"] == record["job_sha256"]
                and receipt.get("owner") == "root"
                and receipt.get("provider_terminal_confirmed") is True
                and (record["lane"] != "vast" or receipt.get("teardown_verified") is True)
                and bool(receipt.get("evidence"))
                and all(
                    sha256_file(project_path(root, name)) == digest
                    for name, digest in receipt["evidence"].items()
                )
            )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return False


def reconcile_terminal(root: Path, runtime: Path, receipt_path: Path) -> None:
    """Release a failed lane only from root's hashed terminal/cleanup evidence."""
    check_runtime(root, runtime)
    receipt_path = project_path(root, str(receipt_path))
    receipt = read_json(receipt_path)
    attempt = receipt["attempt_id"]
    if not IDENTIFIER.fullmatch(attempt):
        raise ValueError("invalid attempt identifier")
    with locked(runtime):
        path = runtime / "attempts" / f"{attempt}.json"
        record = read_json(path)
        if (
            receipt.get("owner") != "root"
            or receipt.get("provider_terminal_confirmed") is not True
            or receipt.get("job_sha256") != record["job_sha256"]
            or not receipt.get("evidence")
        ):
            raise ValueError("root terminal reconciliation evidence required")
        if record["lane"] == "vast" and receipt.get("teardown_verified") is not True:
            raise ValueError("paid lane cannot release before verified owned teardown")
        for name, digest in receipt["evidence"].items():
            if sha256_file(project_path(root, name)) != digest:
                raise ValueError("terminal reconciliation evidence changed")
        record.update(
            status="failed_collected",
            reconciliation_path=str(receipt_path.relative_to(root.resolve())),
            reconciliation_sha256=sha256_file(receipt_path),
            observed_utc=datetime.now(UTC).isoformat(),
        )
        atomic_json(path, record)
        append_event(runtime, {**record, "event": "reconciled_terminal"})


def dispatch_once(root: Path, queue_path: Path, runtime: Path, execute: bool) -> list[dict]:
    """Admit at most one job per free lane; uncertain submissions never retry."""
    check_runtime(root, runtime)
    jobs = load_jobs(queue_path)
    if not execute:
        return inspect_queue(root, jobs, runtime)
    with locked(runtime):
        rows = inspect_queue(root, jobs, runtime)
        for job, row in zip(jobs, rows, strict=True):
            path = runtime / "attempts" / f"{job['attempt_id']}.json"
            if row["status"] == "complete" and path.exists():
                record = read_json(path)
                if record["status"] != "complete" or not terminal_record_verified(root, record):
                    record.update(
                        status="complete",
                        frozen_job=job,
                        completion_sha256=sha256_file(
                            project_path(root, job["completion_receipt"])
                        ),
                        observed_utc=datetime.now(UTC).isoformat(),
                    )
                    atomic_json(path, record)
                    append_event(runtime, {**record, "event": "artifacts_verified"})
        # Old attempts reserve their lane even if a later queue omits them.
        busy = set()
        for path in (runtime / "attempts").glob("*.json"):
            record = read_json(path)
            if not terminal_record_verified(root, record):
                busy.add(record["lane"])
        if (runtime / "STOP_ADMISSION.json").exists():
            return rows
        for job, row in zip(jobs, rows, strict=True):
            if row["status"] != "ready" or job["lane"] in busy:
                continue
            command = valid_admission(root, job, datetime.now(UTC))
            record = {
                "attempt_id": job["attempt_id"],
                "job_sha256": job_digest(job),
                "lane": job["lane"],
                "status": "launch_uncertain",
                "created_utc": datetime.now(UTC).isoformat(),
            }
            record_path = runtime / "attempts" / f"{job['attempt_id']}.json"
            atomic_json(record_path, record)
            append_event(runtime, {**record, "event": "submit_intent"})
            # The invoked adapter owns the durable worker and provider lifecycle.
            # A failed/timed-out CLI may have submitted; its attempt stays reserved.
            try:
                completed = subprocess.run(
                    command, cwd=root, capture_output=True, timeout=60, check=False
                )
                record["returncode"] = completed.returncode
                record["status"] = "submitted" if completed.returncode == 0 else "needs_review"
            except (OSError, subprocess.TimeoutExpired) as error:
                record["error_type"] = type(error).__name__
            record["observed_utc"] = datetime.now(UTC).isoformat()
            atomic_json(record_path, record)
            append_event(runtime, {**record, "event": "submit_returned"})
            busy.add(job["lane"])
        return inspect_queue(root, jobs, runtime)


def inspect_queue(root: Path, jobs: list[dict], runtime: Path) -> list[dict]:
    """Read durable states without allocating compute or trusting provider status."""
    check_runtime(root, runtime)
    complete = set()
    rows = []
    for job in jobs:
        status = "blocked"
        reason = job.get("blocked_reason", "recipe or fresh admission not ready")
        record_path = runtime / "attempts" / f"{job['attempt_id']}.json"
        try:
            if verified_completion(root, job):
                status, reason = "complete", "artifacts verified"
                complete.add(job["attempt_id"])
            elif record_path.exists():
                record = read_json(record_path)
                if record["job_sha256"] != job_digest(job):
                    raise ValueError("attempt was already bound to a different job")
                if record["status"] in {
                    "complete",
                    "failed_collected",
                } and not terminal_record_verified(root, record):
                    raise ValueError("persisted terminal status has no verified terminal evidence")
                status, reason = record["status"], "reconcile existing attempt"
            elif not set(job.get("predecessors", [])).issubset(complete):
                reason = "predecessor has no verified completion"
            elif job.get("prepared") is True:
                valid_admission(root, job, datetime.now(UTC))
                status, reason = "ready", "fresh root admission"
        except (OSError, ValueError, KeyError, TypeError) as error:
            status, reason = "needs_review", str(error)
        holds_lane = record_path.exists() and status not in {"complete", "failed_collected"}
        rows.append(
            {
                "attempt_id": job["attempt_id"],
                "lane": job["lane"],
                "status": status,
                "reason": reason,
                "holds_lane": holds_lane,
            }
        )
    return rows


def main() -> None:
    """Inspect or dispatch one finite cohort; never create a background retry loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reconcile-terminal", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    runtime = args.runtime.resolve()
    if not runtime.is_relative_to(root):
        raise ValueError("runtime must be inside the project root")
    if args.reconcile_terminal:
        if args.execute:
            raise ValueError("reconciliation and dispatch are separate operations")
        reconcile_terminal(root, runtime, args.reconcile_terminal.resolve(strict=True))
    print(json.dumps(dispatch_once(root, args.queue, runtime, args.execute), indent=2))


if __name__ == "__main__":
    main()
