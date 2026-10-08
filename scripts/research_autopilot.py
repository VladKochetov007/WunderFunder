"""Durably hand one root-reviewed local warm queue from v3 to v4."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import bandit_resources
import research_dispatch

LOG = logging.getLogger("research_autopilot")
SCHEMA_VERSION = 1
MAX_ADMISSION_SECONDS = 120
MAX_COLLECTION_ATTEMPTS = 3
STOP_FILE = "STOP.json"
ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
TERMINAL_WORKER_STATES = {"finished", "failed", "not_started"}
ACTIVE_WORKER_STATES = {"preflight", "reserved", "waiting", "running"}


class AutopilotError(RuntimeError):
    """An immutable queue or root authorization failed validation."""


def now_utc() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    """Read a JSON object, failing closed on malformed or missing evidence."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AutopilotError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise AutopilotError(f"{label} must be a JSON object")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write an fsynced JSON file by atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_event(path: Path, value: dict[str, Any]) -> None:
    """Append and fsync one service transition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_time(value: object, label: str) -> datetime:
    """Parse an ISO timestamp that includes a timezone."""
    if not isinstance(value, str):
        raise AutopilotError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AutopilotError(f"{label} is malformed") from error
    if parsed.tzinfo is None:
        raise AutopilotError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def project_path(root: Path, value: object, label: str, *, must_exist: bool = False) -> Path:
    """Resolve a project-relative path and prevent symlink escapes."""
    if not isinstance(value, str) or not value:
        raise AutopilotError(f"{label} must be a non-empty project-relative path")
    path = (root / value).resolve(strict=must_exist)
    if not path.is_relative_to(root.resolve(strict=True)):
        raise AutopilotError(f"{label} escapes the repository")
    return path


def verify_hash_map(root: Path, value: object, label: str) -> dict[str, str]:
    """Verify every pinned project file and return its normalized hash map."""
    if not isinstance(value, dict) or not value:
        raise AutopilotError(f"{label} must be a non-empty path-to-SHA256 map")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise AutopilotError(f"{label} contains an invalid path or digest")
        if sha256_file(project_path(root, name, label, must_exist=True)) != digest:
            raise AutopilotError(f"{label} file changed: {name}")
        result[name] = digest
    return result


def load_config(
    root: Path,
    queue_path: Path,
    policy_path: Path,
    review_path: Path,
    expected_review_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Verify the finite queue, root policy and code pins before any action."""
    root = root.resolve(strict=True)
    for path, label in ((queue_path, "queue"), (policy_path, "policy"), (review_path, "review")):
        if not path.resolve(strict=True).is_relative_to(root):
            raise AutopilotError(f"{label} must remain inside the repository")
    if (
        not SHA256.fullmatch(expected_review_sha256)
        or sha256_file(review_path) != expected_review_sha256
    ):
        raise AutopilotError("root review hash differs from the installed service pin")
    queue = read_json(queue_path, "finite queue")
    policy = read_json(policy_path, "autopilot policy")
    review = read_json(review_path, "root review receipt")
    queue_sha = sha256_file(queue_path)
    policy_sha = sha256_file(policy_path)
    if (
        queue.get("schema_version") != SCHEMA_VERSION
        or policy.get("schema_version") != SCHEMA_VERSION
        or review.get("schema_version") != SCHEMA_VERSION
    ):
        raise AutopilotError("unsupported autopilot schema")
    if policy.get("owner") != "root" or policy.get("authorized") is not True:
        raise AutopilotError("policy is not root-authorized")
    if review.get("owner") != "root" or review.get("authorized") is not True:
        raise AutopilotError("root review receipt is not authorized")
    if policy.get("queue_sha256") != queue_sha or review.get("queue_sha256") != queue_sha:
        raise AutopilotError("finite queue hash is not root-pinned")
    if review.get("policy_sha256") != policy_sha:
        raise AutopilotError("policy hash differs from root review")
    sources = verify_hash_map(root, review.get("source_hashes"), "review.source_hashes")
    required = {
        "scripts/research_autopilot.py": sha256_file(Path(__file__).resolve()),
        "scripts/research_dispatch.py": sha256_file(Path(research_dispatch.__file__).resolve()),
        "scripts/bandit_resources.py": sha256_file(Path(bandit_resources.__file__).resolve()),
        "scripts/warm_queue_worker.py": sha256_file(root / "scripts/warm_queue_worker.py"),
        "scripts/limited": sha256_file(root / "scripts/limited"),
    }
    for name, digest in required.items():
        if sources.get(name) != digest:
            raise AutopilotError(f"root review does not pin current code: {name}")
    expiry = parse_time(policy.get("expires_utc"), "policy.expires_utc")
    review_expiry = parse_time(review.get("expires_utc"), "review.expires_utc")
    if expiry != review_expiry:
        raise AutopilotError("policy and review expiration timestamps differ")
    poll_seconds = policy.get("poll_seconds")
    max_runtime = policy.get("max_runtime_seconds")
    min_disk = policy.get("min_free_disk_bytes")
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, int)
        or not 10 <= poll_seconds <= 600
    ):
        raise AutopilotError("poll_seconds must be in 10..600")
    if (
        isinstance(max_runtime, bool)
        or not isinstance(max_runtime, int)
        or not 0 < max_runtime <= 86_400
    ):
        raise AutopilotError("max_runtime_seconds must be in 1..86400")
    if isinstance(min_disk, bool) or not isinstance(min_disk, int) or min_disk < 0:
        raise AutopilotError("min_free_disk_bytes must be non-negative")
    jobs = research_dispatch.load_jobs(queue_path)
    if len(jobs) != 1:
        raise AutopilotError("this local successor release permits exactly one dispatcher job")
    job = jobs[0]
    if job.get("lane") != "local_gpu" or job.get("prepared") is not True:
        raise AutopilotError("the finite successor must be a prepared local GPU job")
    if job.get("attempt_id") != "warm-queue-local-v4-autopilot-20261008":
        raise AutopilotError("unexpected local v4 attempt id")
    if not isinstance(job.get("local_queue"), dict):
        raise AutopilotError("finite job has no local queue contract")
    local = job["local_queue"]
    allowed_manifest_name = local.get("allowed_spec_manifest")
    allowed_manifest_path = project_path(
        root, allowed_manifest_name, "local_queue.allowed_spec_manifest", must_exist=True
    )
    allowed_manifest = read_json(allowed_manifest_path, "allowed spec manifest")
    if allowed_manifest != local.get("pending_spec_hashes"):
        raise AutopilotError("worker allowlist differs from the reviewed pending spec hashes")
    contract_path = project_path(
        root,
        local.get("root_contract"),
        "local_queue.root_contract",
        must_exist=True,
    )
    if review.get("local_contract_sha256") != sha256_file(contract_path):
        raise AutopilotError("root contract hash differs from root review")
    job_sources = verify_hash_map(root, job.get("source_hashes"), "job.source_hashes")
    allowed_manifest_relative = str(allowed_manifest_path.relative_to(root))
    if allowed_manifest_relative not in job_sources:
        raise AutopilotError("job source pins do not include the worker allowlist")
    command = job.get("launch_command")
    positions = [
        index
        for index, part in enumerate(command if isinstance(command, list) else [])
        if part == "--allowed-spec-manifest"
    ]
    if (
        len(positions) != 1
        or positions[0] + 1 >= len(command)
        or command[positions[0] + 1] != str(allowed_manifest_path)
    ):
        raise AutopilotError("worker command does not pass the exact reviewed spec allowlist")
    return policy, review, jobs


def _local_config(root: Path, job: dict[str, Any]) -> tuple[dict[str, Any], Path, Path, Path]:
    """Resolve the queue, root contract and existing local attempt ledger."""
    local = job["local_queue"]
    queue_dir = project_path(root, local.get("queue_dir"), "local_queue.queue_dir", must_exist=True)
    state = project_path(
        root, local.get("bandit_state"), "local_queue.bandit_state", must_exist=True
    )
    contract = project_path(
        root, local.get("root_contract"), "local_queue.root_contract", must_exist=True
    )
    return local, queue_dir, state, contract


def _systemd_state(unit: str) -> tuple[str, str | None]:
    """Read a user unit's state without treating observer failure as inactivity."""
    try:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=LoadState,ActiveState,SubState,Result",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    state = values.get("ActiveState")
    if values.get("LoadState") == "not-found":
        return "inactive", values.get("Result")
    if result.returncode or state not in {
        "active",
        "activating",
        "deactivating",
        "inactive",
        "failed",
    }:
        return "unknown", values.get("Result")
    return state, values.get("Result")


def _live_queue_worker(queue_dir: Path) -> bool:
    """Detect the worker named in its heartbeat, guarding against PID reuse."""
    path = queue_dir / "worker_heartbeat.json"
    if not path.is_file():
        return False
    try:
        heartbeat = read_json(path, "warm queue heartbeat")
        pid = heartbeat.get("pid")
        if (
            heartbeat.get("host") != os.uname().nodename
            or isinstance(pid, bool)
            or not isinstance(pid, int)
        ):
            return False
        os.kill(pid, 0)
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        return b"warm_queue_worker.py" in command and str(queue_dir).encode() in command
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _pending_specs(queue_dir: Path, local: dict[str, Any]) -> tuple[list[str], str | None]:
    """Verify that pending files are the exact reviewed finite successor set."""
    expected = local.get("pending_spec_hashes")
    if not isinstance(expected, dict) or not expected:
        raise AutopilotError("pending_spec_hashes must be a non-empty map")
    pending_dir = queue_dir / "pending"
    if not pending_dir.is_dir():
        return [], "pending directory is missing"
    paths = sorted(pending_dir.glob("*.json"), key=lambda path: path.name)
    unknown = {path.stem for path in paths} - set(expected)
    if unknown:
        raise AutopilotError(f"unreviewed pending experiments appeared: {sorted(unknown)}")
    pending: list[str] = []
    for path in paths:
        digest = expected.get(path.stem)
        if (
            not isinstance(digest, str)
            or not SHA256.fullmatch(digest)
            or sha256_file(path) != digest
        ):
            raise AutopilotError(f"pending experiment changed after review: {path.stem}")
        pending.append(path.stem)
    if not pending:
        return [], "no useful pending jobs"
    return pending, None


def _required_prior_outputs(root: Path, queue_dir: Path, local: dict[str, Any]) -> dict[str, str]:
    """Bind a completed v3-produced checkpoint to the fresh v4 admission."""
    required = local.get("required_prior_outputs")
    if not isinstance(required, dict) or not required:
        raise AutopilotError("required_prior_outputs must bind the v3 student dependency")
    results_path = queue_dir / "results.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AutopilotError(
                        f"invalid warm worker result line {line_number}"
                    ) from error
                if isinstance(row, dict) and isinstance(row.get("run_id"), str):
                    latest[row["run_id"]] = row
    input_hashes: dict[str, str] = {}
    for run_id, spec in required.items():
        if not isinstance(run_id, str) or not isinstance(spec, dict):
            raise AutopilotError("required_prior_outputs contains an invalid run")
        row = latest.get(run_id)
        expected_spec = spec.get("spec_sha256")
        if (
            row is None
            or row.get("spec_sha256") != expected_spec
            or row.get("status") != "succeeded"
            or not (queue_dir / "done" / f"{run_id}.json").is_file()
        ):
            raise AutopilotError(f"required v3 dependency is not a verified success: {run_id}")
        recorded_outputs = row.get("expected_outputs")
        if not isinstance(recorded_outputs, list):
            raise AutopilotError(f"required v3 dependency outputs are malformed: {run_id}")
        by_path = {
            str(Path(output["path"]).resolve(strict=False)): output
            for output in recorded_outputs
            if isinstance(output, dict) and isinstance(output.get("path"), str)
        }
        paths = spec.get("paths")
        if not isinstance(paths, list) or not paths:
            raise AutopilotError(f"required v3 dependency paths are empty: {run_id}")
        for relative in paths:
            path = project_path(root, relative, f"required_prior_outputs.{run_id}", must_exist=True)
            recorded = by_path.get(str(path))
            if not isinstance(recorded, dict) or recorded.get("exists") is not True:
                raise AutopilotError(f"v3 result did not attest required artifact: {relative}")
            digest = sha256_file(path)
            if recorded.get("sha256") != digest:
                raise AutopilotError(f"v3 output changed after worker receipt: {relative}")
            input_hashes[str(path.relative_to(root))] = digest
    return input_hashes


def preflight(root: Path, policy: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Require v3 terminality, no live/running work, exact pending specs and headroom."""
    local, queue_dir, state, contract_path = _local_config(root, job)
    contract = read_json(contract_path, "local v4 root contract")
    resources = contract.get("resources")
    expected_resources = {
        "cpu_cores": 2,
        "worker_ram_bytes": 9_000_000_000,
        "observed_vram_ceiling_bytes": 5_000_000_000,
        "timeout_seconds": 21_600,
        "idle_exit_seconds": 300,
    }
    if contract.get("authorized") is not True:
        return {"state": "waiting_for_root_enable"}
    if contract.get("attempt_id") != job["attempt_id"] or resources != expected_resources:
        raise AutopilotError("root contract does not authorize this exact one-worker profile")
    if contract.get("pending_spec_hashes") != local.get("pending_spec_hashes"):
        raise AutopilotError("root contract does not bind the finite pending spec set")
    if contract.get("prior_attempt_id") != local.get("prior_attempt_id"):
        raise AutopilotError("root contract does not bind the live v3 attempt")

    prior_id = str(local["prior_attempt_id"])
    prior_unit = str(local["prior_unit"])
    prior_path = bandit_resources.attempt_path(state, prior_id)
    if not prior_path.is_file():
        return {"state": "waiting_for_v3_terminal_receipt"}
    prior_record = bandit_resources.read_json(prior_path)
    prior_state, _ = _systemd_state(prior_unit)
    if prior_state in {"active", "activating", "deactivating", "unknown"}:
        return {"state": "waiting_for_v3_service_drain", "prior_status": prior_record.get("status")}
    if prior_record.get("status") not in {"finished", "failed"}:
        return {
            "state": "waiting_for_v3_terminal_receipt",
            "prior_status": prior_record.get("status"),
        }
    if list((queue_dir / "running").glob("*.json")) or _live_queue_worker(queue_dir):
        return {"state": "waiting_for_v3_job_boundary"}
    pending, reason = _pending_specs(queue_dir, local)
    if reason:
        return {"state": "queue_exhausted", "reason": reason}
    dynamic_inputs = _required_prior_outputs(root, queue_dir, local)
    for run_id in pending:
        path = queue_dir / "pending" / f"{run_id}.json"
        relative = str(path.resolve(strict=True).relative_to(root))
        dynamic_inputs[relative] = sha256_file(path)
    dynamic_inputs[str(contract_path.relative_to(root))] = sha256_file(contract_path)
    successor_id = str(job["attempt_id"])
    if bandit_resources.attempt_path(state, successor_id).exists():
        return {"state": "successor_attempt_already_exists"}
    try:
        parent = bandit_resources.verify_parent(bandit_resources.parent_properties())
        bandit_resources.verify_parent_membership(parent)
        memory_bytes = bandit_resources.memory_current(parent)
        capacity = bandit_resources.verify_capacity(
            state,
            parent,
            cpus=2,
            ram_bytes=9_000_000_000,
            current_memory_bytes=memory_bytes,
        )
        vram_bytes = bandit_resources.gpu_memory_bytes()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        return {"state": "resources_unavailable", "error": type(error).__name__}
    if vram_bytes >= 5_000_000_000:
        return {"state": "gpu_headroom_low", "vram_bytes": vram_bytes}
    disk_bytes = shutil.disk_usage(root).free
    if disk_bytes < int(policy["min_free_disk_bytes"]):
        return {"state": "local_disk_headroom_low", "local_free_disk_bytes": disk_bytes}
    return {
        "state": "ready",
        "pending_attempts": pending,
        "input_hashes": dynamic_inputs,
        "local_free_disk_bytes": disk_bytes,
        "provider_quota_seconds": 21_600,
        "prior_worker_status": prior_record.get("status"),
        "resource_snapshot": {
            "parent_memory_current_bytes": memory_bytes,
            "observed_vram_bytes": vram_bytes,
            "capacity": capacity,
        },
    }


def _output_record(root: Path, value: object, run_id: str) -> dict[str, Any]:
    """Verify and hash one worker-reported artifact without materializing it."""
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise AutopilotError(f"{run_id} has a malformed output receipt")
    raw_path = Path(value["path"])
    path = raw_path.resolve(strict=False)
    if not path.is_relative_to(root):
        raise AutopilotError(f"{run_id} output escapes the repository")
    if not path.is_file():
        return {"path": str(path.relative_to(root)), "exists": False}
    digest = sha256_file(path)
    recorded = value.get("sha256")
    if recorded is not None and recorded != digest:
        raise AutopilotError(f"{run_id} output hash differs from worker receipt")
    return {
        "path": str(path.relative_to(root)),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def collect_local(
    root: Path,
    runtime: Path,
    job: dict[str, Any],
    worker_record: dict[str, Any],
) -> tuple[Path, Path]:
    """Write a durable SSD packet of queue results, predictions/checkpoints and cost."""
    local, queue_dir, _, _ = _local_config(root, job)
    expected_specs = local.get("all_spec_hashes")
    if not isinstance(expected_specs, dict) or not expected_specs:
        raise AutopilotError("all_spec_hashes must pin every local recipe in the queue")
    latest: dict[str, dict[str, Any]] = {}
    results_path = queue_dir / "results.jsonl"
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AutopilotError(
                        f"invalid warm worker result line {line_number}"
                    ) from error
                if isinstance(row, dict) and isinstance(row.get("run_id"), str):
                    latest[row["run_id"]] = row
    runs: list[dict[str, Any]] = []
    all_artifacts: dict[str, str] = {}
    for run_id, expected_hash in expected_specs.items():
        if (
            not isinstance(run_id, str)
            or not isinstance(expected_hash, str)
            or not SHA256.fullmatch(expected_hash)
        ):
            raise AutopilotError("all_spec_hashes contains an invalid entry")
        row = latest.get(run_id)
        if row is None or row.get("spec_sha256") != expected_hash:
            raise AutopilotError(f"no terminal worker result matches reviewed spec {run_id}")
        placement = (
            "done"
            if (queue_dir / "done" / f"{run_id}.json").is_file()
            else "failed"
            if (queue_dir / "failed" / f"{run_id}.json").is_file()
            else None
        )
        if placement is None:
            raise AutopilotError(f"reviewed queue item has no terminal placement: {run_id}")
        output_rows = row.get("expected_outputs", [])
        if not isinstance(output_rows, list):
            raise AutopilotError(f"{run_id} expected_outputs is malformed")
        outputs = [_output_record(root, output, run_id) for output in output_rows]
        for output in outputs:
            if output.get("exists") is True:
                all_artifacts[str(output["path"])] = str(output["sha256"])
        runs.append(
            {
                "run_id": run_id,
                "spec_sha256": expected_hash,
                "placement": placement,
                "status": row.get("status"),
                "steps": row.get("steps", []),
                "outputs": outputs,
            }
        )
    queue_artifacts = {}
    for name in ("results.jsonl", "worker_heartbeat.json"):
        path = queue_dir / name
        if path.is_file():
            relative = str(path.resolve(strict=True).relative_to(root))
            queue_artifacts[relative] = sha256_file(path)
    logs: dict[str, str] = {}
    log_dir = queue_dir / "logs"
    if log_dir.is_dir():
        ids = set(expected_specs)
        for path in sorted(log_dir.iterdir()):
            if path.is_file() and any(path.name.startswith(f"{run_id}.") for run_id in ids):
                logs[str(path.resolve(strict=True).relative_to(root))] = sha256_file(path)
    worker_attempt = str(job["attempt_id"])
    manifest = {
        "schema_version": 1,
        "attempt_id": worker_attempt,
        "collected_utc": now_utc().isoformat(),
        "worker_terminal_status": worker_record.get("status"),
        "worker_elapsed_seconds": worker_record.get("elapsed_seconds"),
        "resource_peaks": {
            "ram_bytes": worker_record.get("peak_ram_bytes"),
            "parent_ram_bytes": worker_record.get("peak_parent_ram_bytes"),
            "observed_vram_bytes": worker_record.get("peak_observed_vram_bytes"),
        },
        "reviewed_spec_hashes": expected_specs,
        "runs": runs,
        "queue_artifacts": queue_artifacts,
        "logs": logs,
        "model_and_prediction_artifacts": all_artifacts,
    }
    results_path_out = project_path(root, local.get("results_manifest"), "local results manifest")
    cost_path = project_path(root, local.get("cost_artifact"), "local cost artifact")
    atomic_json(results_path_out, manifest)
    elapsed = float(worker_record.get("elapsed_seconds", 0.0))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise AutopilotError("worker elapsed_seconds is invalid")
    atomic_json(
        cost_path,
        {
            "schema_version": 1,
            "attempt_id": worker_attempt,
            "actual_cost_usd": "0.00",
            "provider_quota_seconds": min(21_600, int(elapsed)),
            "elapsed_seconds": elapsed,
            "peak_ram_bytes": worker_record.get("peak_ram_bytes"),
            "peak_parent_ram_bytes": worker_record.get("peak_parent_ram_bytes"),
            "peak_observed_vram_bytes": worker_record.get("peak_observed_vram_bytes"),
            "observed_utc": now_utc().isoformat(),
        },
    )
    return results_path_out, cost_path


def collect_terminal(
    root: Path, runtime: Path, job: dict[str, Any], worker_record: dict[str, Any]
) -> None:
    """Persist terminal evidence before dispatcher marks the supervisor job complete."""
    if research_dispatch.verified_completion(root, job):
        return
    result_path, cost_path = collect_local(root, runtime, job, worker_record)
    receipt_path = project_path(root, job.get("completion_receipt"), "completion receipt")
    names = job.get("expected_artifacts")
    cost_name = job.get("cost_artifact_name")
    if not isinstance(names, list) or not isinstance(cost_name, str):
        raise AutopilotError("job expected_artifacts and cost_artifact_name are required")
    result_name = str(job["local_queue"]["results_artifact_name"])
    if set(names) != {result_name, cost_name}:
        raise AutopilotError("job must expect exactly the results manifest and cost receipt")
    artifacts = [
        {
            "name": result_name,
            "path": str(result_path.relative_to(root)),
            "sha256": sha256_file(result_path),
        },
        {
            "name": cost_name,
            "path": str(cost_path.relative_to(root)),
            "sha256": sha256_file(cost_path),
        },
    ]
    atomic_json(
        receipt_path,
        {
            "status": "complete",
            "owner": "root",
            "attempt_id": job["attempt_id"],
            "job_sha256": research_dispatch.job_digest(job),
            "finished_utc": now_utc().isoformat(),
            "worker_terminal_status": worker_record.get("status"),
            "science_statuses": {
                run["run_id"]: run["status"]
                for run in read_json(result_path, "collected local results")["runs"]
            },
            "artifacts": artifacts,
        },
    )


def _observe_worker(root: Path, state: Path, runtime: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Monitor the existing resource guardian; do not re-submit uncertain jobs."""
    attempt = str(job["attempt_id"])
    try:
        if research_dispatch.verified_completion(root, job):
            return {"state": "collected"}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {
            "state": "launch_uncertain",
            "reason": f"completion_receipt_invalid:{type(error).__name__}",
        }
    worker_record_path = bandit_resources.attempt_path(state, attempt)
    if not worker_record_path.is_file():
        return {"state": "launch_uncertain", "reason": "resource_guardian_record_missing"}
    record = bandit_resources.read_json(worker_record_path)
    status = record.get("status")
    if status in ACTIVE_WORKER_STATES:
        return {"state": "running", "phase": record.get("phase", status), "worker_record": record}
    if status in TERMINAL_WORKER_STATES:
        try:
            collect_terminal(root, runtime, job, record)
        except (AutopilotError, OSError, ValueError) as error:
            attempts_path = runtime / "collection_attempts" / f"{attempt}.json"
            prior = (
                read_json(attempts_path, "collection attempts") if attempts_path.is_file() else {}
            )
            attempts = int(prior.get("attempts", 0)) + 1
            status = "needs_review" if attempts >= MAX_COLLECTION_ATTEMPTS else "retry_wait"
            atomic_json(
                attempts_path,
                {
                    "attempt_id": attempt,
                    "attempts": attempts,
                    "status": status,
                    "last_error_type": type(error).__name__,
                    "observed_utc": now_utc().isoformat(),
                },
            )
            append_event(
                runtime / "events.jsonl",
                {
                    "event": "collection_failed",
                    "attempt_id": attempt,
                    "attempts": attempts,
                    "status": status,
                    "error_type": type(error).__name__,
                    "observed_utc": now_utc().isoformat(),
                },
            )
            return {
                "state": "collection_needs_review"
                if status == "needs_review"
                else "collection_retry",
                "attempts": attempts,
                "reason": type(error).__name__,
                "worker_record": record,
            }
        atomic_json(
            runtime / "collection_attempts" / f"{attempt}.json",
            {
                "attempt_id": attempt,
                "attempts": 0,
                "status": "collected",
                "observed_utc": now_utc().isoformat(),
            },
        )
        return {"state": "collected", "worker_record": record}
    return {"state": "launch_uncertain", "reason": f"unrecognized_guardian_status:{status}"}


def _write_admission(
    root: Path,
    queue_sha: str,
    review: dict[str, Any],
    review_sha: str,
    job: dict[str, Any],
    input_hashes: dict[str, str],
) -> None:
    """Mint a single-use, ≤120-second dispatcher receipt from root's finite review."""
    source_hashes = dict(review["source_hashes"])
    source_hashes.update(verify_hash_map(root, job.get("source_hashes"), "job.source_hashes"))
    source_hashes.update(verify_hash_map(root, input_hashes, "preflight.input_hashes"))
    receipt = {
        "schema_version": 1,
        "owner": "root",
        "authorized": True,
        "authority": "reviewed_finite_queue",
        "attempt_id": job["attempt_id"],
        "job_sha256": research_dispatch.job_digest(job),
        "queue_sha256": queue_sha,
        "review_sha256": review_sha,
        "created_utc": now_utc().isoformat(),
        "expires_utc": (now_utc() + timedelta(seconds=MAX_ADMISSION_SECONDS - 1)).isoformat(),
        "source_hashes": source_hashes,
    }
    atomic_json(project_path(root, job.get("admission_receipt"), "admission receipt"), receipt)


def _attempt_active(root: Path, runtime: Path, job: dict[str, Any]) -> bool:
    path = runtime / "attempts" / f"{job['attempt_id']}.json"
    if not path.is_file():
        return False
    record = research_dispatch.read_json(path)
    if record.get("status") in {"complete", "failed_collected"}:
        return not research_dispatch.terminal_record_verified(root, record)
    return True


def run_cycle(
    root: Path,
    queue_path: Path,
    policy_path: Path,
    review_path: Path,
    runtime: Path,
    expected_review_sha256: str,
    *,
    heartbeat: Callable[[str, str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Observe v3, admit at most one exact v4 worker, collect, then stop at drain."""
    root = root.resolve(strict=True)
    runtime = runtime.resolve()
    if not runtime.is_relative_to(root):
        raise AutopilotError("autopilot runtime must remain inside repository")
    runtime.mkdir(parents=True, exist_ok=True)
    policy, review, jobs = load_config(
        root,
        queue_path.resolve(strict=True),
        policy_path.resolve(strict=True),
        review_path.resolve(strict=True),
        expected_review_sha256,
    )
    job = jobs[0]
    queue_sha = sha256_file(queue_path)
    review_sha = sha256_file(review_path)
    now = now_utc()
    admission_open = (
        now < parse_time(policy["expires_utc"], "policy.expires_utc")
        and not (runtime / STOP_FILE).exists()
        and not (runtime / "STOP_ADMISSION.json").exists()
        and not (stop_requested is not None and stop_requested())
    )
    if not admission_open:
        request_stop(runtime, "root_stop_or_policy_expiry")
    state = project_path(
        root, job["local_queue"]["bandit_state"], "local bandit state", must_exist=True
    )
    existing_attempt = (runtime / "attempts" / f"{job['attempt_id']}.json").is_file()
    observed: dict[str, Any] | None = None
    preflight_result: dict[str, Any] = {"state": "existing_v4_attempt"}
    if existing_attempt:
        observed = _observe_worker(root, state, runtime, job)
        append_event(
            runtime / "events.jsonl",
            {
                "event": "worker_observed",
                "observed_utc": now.isoformat(),
                "result": {key: value for key, value in observed.items() if key != "worker_record"},
            },
        )
    else:
        # Expired preflight receipts are not carried across service cycles.
        admission_path = project_path(root, job["admission_receipt"], "admission receipt")
        if admission_path.is_file():
            old = read_json(admission_path, "old admission receipt")
            if (
                old.get("authority") != "reviewed_finite_queue"
                or old.get("attempt_id") != job["attempt_id"]
            ):
                raise AutopilotError(
                    "refusing to replace an admission minted by a different authority"
                )
            admission_path.unlink()
        if admission_open and shutil.disk_usage(root).free >= int(policy["min_free_disk_bytes"]):
            preflight_result = preflight(root, policy, job)
            atomic_json(
                runtime / "preflight.json",
                {
                    "attempt_id": job["attempt_id"],
                    "queue_sha256": queue_sha,
                    "observed_utc": now.isoformat(),
                    **preflight_result,
                },
            )
            append_event(
                runtime / "events.jsonl",
                {
                    "event": "preflight",
                    "observed_utc": now.isoformat(),
                    "state": preflight_result["state"],
                },
            )
            if preflight_result["state"] == "ready":
                _write_admission(
                    root,
                    queue_sha,
                    review,
                    review_sha,
                    job,
                    preflight_result["input_hashes"],
                )
        elif admission_open:
            preflight_result = {"state": "local_disk_headroom_low"}
            append_event(
                runtime / "events.jsonl",
                {
                    "event": "preflight",
                    "observed_utc": now.isoformat(),
                    "state": preflight_result["state"],
                },
            )
        else:
            preflight_result = {"state": "admission_closed"}
    if stop_requested is not None and stop_requested():
        request_stop(runtime, "service_signal_drain")
        admission_open = False
        admission_path = project_path(root, job["admission_receipt"], "admission receipt")
        if not existing_attempt and admission_path.exists():
            admission_path.unlink()
    # The frozen dispatcher handles uncertain launch receipts idempotently and
    # records completion only after every terminal artifact hash verifies.
    dispatch_rows = research_dispatch.dispatch_once(root, queue_path, runtime, execute=True)
    active = _attempt_active(root, runtime, job)
    pending, queue_reason = _pending_specs(_local_config(root, job)[1], job["local_queue"])
    local, queue_dir, _, _ = _local_config(root, job)
    prior_state, _ = _systemd_state(str(local["prior_unit"]))
    external_v3_active = prior_state in {
        "active",
        "activating",
        "deactivating",
        "unknown",
    } or _live_queue_worker(queue_dir)
    row_status = dispatch_rows[0]["status"]
    if observed is not None and observed["state"] in {
        "launch_uncertain",
        "collection_needs_review",
    }:
        should_exit, reason = True, observed["state"]
    elif row_status == "complete":
        should_exit, reason = True, "v4_terminal_artifacts_collected"
    elif not pending and not active and not external_v3_active:
        should_exit, reason = True, "finite_queue_exhausted"
    elif not admission_open and not active and not external_v3_active:
        should_exit, reason = True, "admission_closed_and_drained"
    elif active:
        should_exit, reason = False, "v4_active_or_uncertain"
    elif queue_reason is not None and not external_v3_active:
        should_exit, reason = True, "no_useful_pending_jobs"
    else:
        should_exit, reason = False, preflight_result.get("state", "waiting")
    result = {
        "status": "needs_review"
        if reason in {"launch_uncertain", "collection_needs_review"}
        else "draining"
        if not admission_open
        else "running",
        "reason": reason,
        "admission_open": admission_open,
        "queue_sha256": queue_sha,
        "review_sha256": review_sha,
        "worker_pending": pending,
        "external_v3_active": external_v3_active,
        "v4_attempt_status": row_status,
        "active_v4": active,
        "jobs": dispatch_rows,
        "should_exit": should_exit,
        "observed_utc": now_utc().isoformat(),
    }
    atomic_json(runtime / "state.json", result)
    if heartbeat is not None:
        heartbeat(job["attempt_id"] if active else "controller", reason)
    return result


def request_stop(runtime: Path, reason: str) -> None:
    """Persist graceful drain intent; the admitted worker remains untouched."""
    runtime.mkdir(parents=True, exist_ok=True)
    with research_dispatch.locked(runtime):
        _write_stop_markers(runtime, reason)


def _write_stop_markers(runtime: Path, reason: str) -> None:
    """Write the drain record and dispatcher barrier as one locked transition."""
    observed = now_utc().isoformat()
    path = runtime / STOP_FILE
    if not path.exists():
        atomic_json(path, {"reason": reason, "created_utc": observed})
        append_event(
            runtime / "events.jsonl",
            {"event": "stop_requested", "reason": reason, "observed_utc": observed},
        )
    barrier = runtime / "STOP_ADMISSION.json"
    if not barrier.exists():
        atomic_json(barrier, {"reason": reason, "observed_utc": observed})


def _service_signal_stop(runtime: Path) -> None:
    """Close the gate synchronously without reacquiring a lock the handler may interrupt.

    The service's signal handler runs on the dispatcher thread, so waiting for
    its flock here could deadlock. A signal delivered before dispatch_once
    reaches its gate writes STOP_ADMISSION before control returns to dispatch;
    an already-passed gate is an in-flight admission. CLI/root stops use
    request_stop and serialize the barrier with that same dispatcher lock.
    """
    runtime.mkdir(parents=True, exist_ok=True)
    _write_stop_markers(runtime, "service_signal_drain")


def wait_for_next_cycle(wakeup: threading.Event, seconds: int) -> None:
    """Wait at the configured poll rate, consuming a single early wakeup."""
    wakeup.wait(seconds)
    wakeup.clear()


def run_service(
    root: Path,
    queue_path: Path,
    policy_path: Path,
    review_path: Path,
    runtime: Path,
    expected_review_sha256: str,
    *,
    once: bool,
) -> int:
    """Run as a durable systemd user service; restart reattaches to receipts."""
    root = root.resolve(strict=True)
    runtime = runtime.resolve()
    if not runtime.is_relative_to(root):
        raise AutopilotError("runtime must remain inside repository")
    runtime.mkdir(parents=True, exist_ok=True)
    lock_file = (runtime / "autopilot.lock").open("a+")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AutopilotError("another autopilot service already owns this runtime") from error
        stop_event = threading.Event()
        wakeup = threading.Event()

        def drain_signal(_signum: int, _frame: object) -> None:
            stop_event.set()
            _service_signal_stop(runtime)
            wakeup.set()

        signal.signal(signal.SIGTERM, drain_signal)
        signal.signal(signal.SIGINT, drain_signal)
        policy, _, _ = load_config(
            root,
            queue_path.resolve(strict=True),
            policy_path.resolve(strict=True),
            review_path.resolve(strict=True),
            expected_review_sha256,
        )
        start_path = runtime / "service_started.json"
        if start_path.exists():
            start_record = read_json(start_path, "service start receipt")
            if start_record.get("queue_sha256") != sha256_file(queue_path) or start_record.get(
                "policy_sha256"
            ) != sha256_file(policy_path):
                raise AutopilotError("queue or policy changed during this service lifetime")
            started = parse_time(start_record.get("started_utc"), "service_started.started_utc")
        else:
            started = now_utc()
            atomic_json(
                start_path,
                {
                    "started_utc": started.isoformat(),
                    "queue_sha256": sha256_file(queue_path),
                    "policy_sha256": sha256_file(policy_path),
                    "review_sha256": expected_review_sha256,
                },
            )
        while True:
            if stop_event.is_set():
                request_stop(runtime, "service_signal_drain")
            elapsed = max(0.0, (now_utc() - started).total_seconds())
            if elapsed >= int(policy["max_runtime_seconds"]):
                request_stop(runtime, "maximum_runtime_reached")
            atomic_json(
                runtime / "heartbeat.json",
                {"pid": os.getpid(), "phase": "cycle", "last_update_utc": now_utc().isoformat()},
            )
            try:
                result = run_cycle(
                    root,
                    queue_path,
                    policy_path,
                    review_path,
                    runtime,
                    expected_review_sha256,
                    stop_requested=stop_event.is_set,
                )
            except AutopilotError as error:
                atomic_json(
                    runtime / "state.json",
                    {
                        "status": "needs_review",
                        "error": type(error).__name__,
                        "observed_utc": now_utc().isoformat(),
                    },
                )
                append_event(
                    runtime / "events.jsonl",
                    {
                        "event": "contract_error",
                        "error": type(error).__name__,
                        "observed_utc": now_utc().isoformat(),
                    },
                )
                LOG.exception("autopilot stopped on a reviewed-contract error")
                return 2
            if result["should_exit"]:
                return 2 if result["status"] == "needs_review" else 0
            if once:
                return 0
            policy = read_json(policy_path, "policy")
            atomic_json(
                runtime / "heartbeat.json",
                {
                    "pid": os.getpid(),
                    "phase": result["reason"],
                    "last_update_utc": now_utc().isoformat(),
                },
            )
            wait_for_next_cycle(wakeup, int(policy["poll_seconds"]))
    finally:
        lock_file.close()


def resume_service(
    root: Path,
    queue_path: Path,
    policy_path: Path,
    review_path: Path,
    runtime: Path,
    expected_review_sha256: str,
) -> None:
    """Resume only from a new root review that binds the exact stop receipt."""
    _, review, _ = load_config(
        root,
        queue_path.resolve(strict=True),
        policy_path.resolve(strict=True),
        review_path.resolve(strict=True),
        expected_review_sha256,
    )
    stop_path = runtime / STOP_FILE
    if not stop_path.is_file():
        raise AutopilotError("no stop receipt exists")
    stop_sha = sha256_file(stop_path)
    if review.get("resume_from_stop_sha256") != stop_sha:
        raise AutopilotError("new root review does not bind the previous stop receipt")
    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    stop_path.replace(runtime / f"STOP.closed.{stamp}.json")
    barrier = runtime / "STOP_ADMISSION.json"
    if barrier.exists():
        barrier.replace(runtime / f"STOP_ADMISSION.closed.{stamp}.json")
    started = runtime / "service_started.json"
    if started.exists():
        started.replace(runtime / f"service_started.closed.{stamp}.json")
    append_event(
        runtime / "events.jsonl",
        {
            "event": "resumed",
            "stop_sha256": stop_sha,
            "review_sha256": expected_review_sha256,
            "observed_utc": now_utc().isoformat(),
        },
    )


def parser() -> argparse.ArgumentParser:
    """Build the run, stop, resume and status CLI."""
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("run", "once", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--queue", type=Path, required=True)
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--review", type=Path, required=True)
        command.add_argument("--runtime", type=Path, required=True)
        command.add_argument("--expected-review-sha256", required=True)
    stop = commands.add_parser("stop")
    stop.add_argument("--runtime", type=Path, required=True)
    stop.add_argument("--reason", required=True)
    status = commands.add_parser("status")
    status.add_argument("--runtime", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    """Entry point for a root-installed persistent service or safe control CLI."""
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "stop":
        request_stop(args.runtime, args.reason)
        return 0
    if args.command == "status":
        print(json.dumps(read_json(args.runtime / "state.json", "autopilot state"), indent=2))
        return 0
    if args.command == "resume":
        resume_service(
            args.root,
            args.queue,
            args.policy,
            args.review,
            args.runtime,
            args.expected_review_sha256,
        )
        return 0
    return run_service(
        args.root,
        args.queue,
        args.policy,
        args.review,
        args.runtime,
        args.expected_review_sha256,
        once=args.command == "once",
    )


if __name__ == "__main__":
    sys.exit(main())
