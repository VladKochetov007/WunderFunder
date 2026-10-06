"""Wait for the frozen C0 exports, then invoke the immutable paired grader."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from bandit_resources import (
    atomic_json,
    locked,
    read_json,
    sha256_file,
    verify_controller_membership,
)

LOG = logging.getLogger(__name__)
ACTIVE_STATUSES = {"preflight", "launch_uncertain", "reserved", "waiting", "running"}
OBSERVER_RETRIES = 3
OBSERVER_RETRY_SECONDS = 0.25
HEARTBEAT_MAX_AGE_SECONDS = 180
SEQUENCE_ROWS = 20_000
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_EXPORT_IDS = (
    "e503-bd26-c0-s3-f0-valid-v1",
    "e503-bd26-c0-s4-f0-valid-v1",
)


class GateError(RuntimeError):
    """A frozen gate precondition failed and requires explicit review."""


def _utc_now() -> datetime:
    """Return an aware UTC timestamp for durable gate heartbeats."""
    return datetime.now(UTC)


def _timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return _utc_now().isoformat()


def _create_receipt(path: Path, initial: dict[str, object]) -> None:
    """Create the first receipt exclusively so a rerun cannot overwrite evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(initial, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish(path: Path, state: dict[str, object], **updates: object) -> None:
    """Durably update the gate receipt and refresh its heartbeat."""
    state.update(updates, heartbeat_utc=_timestamp())
    atomic_json(path, state)


def _read_attempt(path: Path) -> dict[str, object]:
    """Retry short atomic-write races, then surface unknown evidence to review."""
    last_error: str | None = None
    for retry in range(OBSERVER_RETRIES):
        try:
            record = read_json(path)
        except (OSError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if retry + 1 < OBSERVER_RETRIES:
                time.sleep(OBSERVER_RETRY_SECONDS)
            continue
        return record
    raise GateError(f"attempt record observation is unknown for {path.name}: {last_error}")


def _attempt_state(
    runtime: Path,
    expected: Mapping[str, object],
    now: datetime,
) -> dict[str, object]:
    """Validate one exact E-503 record without signaling its worker."""
    attempt = str(expected["attempt"])
    path = runtime / "attempts" / f"{attempt}.json"
    if not path.exists():
        return {"attempt": attempt, "state": "missing"}
    record = _read_attempt(path)
    for key in ("id", "command", "kind", "parent_slice"):
        expected_value = expected.get(key)
        if expected_value is not None and record.get(key) != expected_value:
            raise GateError(f"{attempt}: recorded {key} differs from frozen E-503")

    status = record.get("status")
    exit_status = record.get("exit_status")
    heartbeat = record.get("heartbeat_utc")
    summary = {
        "attempt": attempt,
        "state": status,
        "exit_status": exit_status,
        "heartbeat_utc": heartbeat,
    }
    if status == "finished" and exit_status == 0:
        return summary | {"state": "success"}
    if status in ACTIVE_STATUSES:
        try:
            observed = datetime.fromisoformat(str(heartbeat))
        except ValueError as error:
            raise GateError(f"{attempt}: active attempt has an invalid heartbeat") from error
        if observed.tzinfo is None:
            raise GateError(f"{attempt}: active attempt heartbeat lacks a timezone")
        age = (now - observed.astimezone(UTC)).total_seconds()
        if age < -60:
            raise GateError(f"{attempt}: active attempt heartbeat is in the future")
        if age > HEARTBEAT_MAX_AGE_SECONDS:
            raise GateError(f"{attempt}: stale worker heartbeat; no worker was stopped")
        return summary | {"state": "active"}
    raise GateError(f"{attempt}: terminal or unknown attempt state needs review: {status!r}")


def _inspect_attempts(
    runtime: Path,
    expected_attempts: list[dict[str, object]],
    now: datetime,
    admission_closed: bool,
) -> tuple[str, list[dict[str, object]]]:
    """Return waiting, ready, or drained after checking every exact export."""
    states = [_attempt_state(runtime, expected, now) for expected in expected_attempts]
    if all(item["state"] == "success" for item in states):
        return ("drained" if admission_closed else "ready"), states
    if admission_closed and not any(item["state"] == "active" for item in states):
        return "drained", states
    return "waiting", states


def _verify_frozen_sources(plan: Mapping[str, object]) -> None:
    """Require the evaluator worktree and all registered metadata to stay frozen."""
    exports = plan.get("exports")
    if (
        not isinstance(exports, list)
        or len(exports) != len(EXPECTED_EXPORT_IDS)
        or any(not isinstance(item, dict) for item in exports)
        or tuple(item["attempt"] for item in exports) != EXPECTED_EXPORT_IDS
    ):
        raise GateError("gate must wait for exactly the preregistered C0 validation exports")
    frozen = plan["frozen_sources"]
    if not isinstance(frozen, dict):
        raise GateError("gate plan lacks frozen source bindings")
    repository_root = Path(str(frozen["repository_root"])).resolve(strict=True)
    commit = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    if commit != frozen["commit"]:
        raise GateError("immutable evaluator worktree is at an unexpected commit")
    status = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
    if status:
        raise GateError("immutable evaluator worktree has uncommitted changes")

    for name in ("grader", "e502_preregistration", "e503_preregistration", "source_families"):
        binding = frozen[name]
        if not isinstance(binding, dict):
            raise GateError(f"gate plan lacks the {name} hash binding")
        path = repository_root / str(binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise GateError(f"frozen {name} file hash changed")

    e502 = read_json(repository_root / str(frozen["e502_preregistration"]["path"]))
    if e502.get("id") != "E-502" or e502.get("status") != "preregistered":
        raise GateError("frozen E-502 registration is not preregistered")

    attestation = frozen["seed3_attestation"]
    if not isinstance(attestation, dict):
        raise GateError("gate plan lacks the accepted seed-3 attestation")
    attestation_path = Path(str(attestation["path"]))
    if sha256_file(attestation_path) != attestation["sha256"]:
        raise GateError("seed-3 completion attestation hash changed")
    attestation_record = read_json(attestation_path)
    if attestation_record.get("decision") != "accept_as_completed_with_guard_incident":
        raise GateError("seed-3 completion attestation is not the reviewed decision")

    registration = read_json(repository_root / str(frozen["e503_preregistration"]["path"]))
    if registration.get("id") != "E-503" or registration.get("status") != "preregistered":
        raise GateError("frozen E-503 registration is not preregistered")
    registered = {item.get("attempt"): item for item in registration.get("experiments", [])}
    for expected in plan["exports"]:
        item = registered.get(expected["attempt"])
        if not isinstance(item, dict):
            raise GateError(f"E-503 does not contain {expected['attempt']}")
        for key in ("run", "split", "checkpoint_epoch", "command", "output"):
            if item.get(key) != expected.get(key):
                raise GateError(f"{expected['attempt']}: E-503 {key} differs from gate plan")


def _npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype, int]:
    """Read an NPY header only; prediction values remain unread."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise GateError(f"unsupported NPY format version {version}")
        if fortran_order:
            raise GateError(f"prediction cache must use C-order storage: {path.name}")
        return tuple(map(int, shape)), np.dtype(dtype), stream.tell()


def _cache_bindings(plan: Mapping[str, object]) -> list[dict[str, object]]:
    """Hash finished cache bytes and verify sidecar, checkpoint, split and row order."""
    artifact_root = Path(str(plan["artifact_root"])).resolve(strict=True)
    artifact_cache = artifact_root / "cache"
    target_path = artifact_cache / "valid_targets.npz"
    split_path = artifact_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    if not target_path.is_file() or not split_path.is_file():
        raise GateError("frozen valid-target or split cache is missing")
    with np.load(target_path, allow_pickle=False) as targets:
        row_groups = list(map(int, targets["row_group"]))
    split_sha256 = sha256_file(split_path)
    cache_root = artifact_cache.resolve(strict=True)
    bindings: list[dict[str, object]] = []

    for export in plan["exports"]:
        prediction_path = Path(str(export["output"]))
        if (
            not prediction_path.is_file()
            or prediction_path.with_name(prediction_path.name + ".partial").exists()
        ):
            raise GateError(f"{export['attempt']}: prediction cache is absent or partial")
        if prediction_path.resolve(strict=True).parent != cache_root:
            raise GateError(
                f"{export['attempt']}: prediction cache is outside frozen artifact cache"
            )
        sidecar_path = prediction_path.with_suffix(".json")
        if not sidecar_path.is_file():
            raise GateError(f"{export['attempt']}: prediction sidecar is missing")
        metadata = read_json(sidecar_path)
        run = str(export["run"])
        epoch = int(export["checkpoint_epoch"])
        checkpoint_path = artifact_root / "artifacts/runs" / run / f"model_ep{epoch}.pt"
        checkpoint_sha256 = sha256_file(checkpoint_path)
        prediction_sha256 = sha256_file(prediction_path)
        expected_metadata = {
            "run": run,
            "split": "valid",
            "epoch": epoch,
            "checkpoint_sha256": checkpoint_sha256,
            "prediction_sha256": prediction_sha256,
            "split_manifest_sha256": split_sha256,
            "row_groups": row_groups,
        }
        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                raise GateError(f"{export['attempt']}: prediction sidecar {key} differs")
        if not SHA256.fullmatch(str(metadata.get("prediction_sha256", ""))):
            raise GateError(f"{export['attempt']}: invalid prediction hash in sidecar")
        shape, dtype, header_bytes = _npy_header(prediction_path)
        expected_shape = (len(row_groups), SEQUENCE_ROWS, 2)
        expected_size = header_bytes + int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize
        if shape != expected_shape or dtype != np.dtype(np.float32):
            raise GateError(f"{export['attempt']}: prediction NPY header has wrong shape or dtype")
        if prediction_path.stat().st_size != expected_size:
            raise GateError(f"{export['attempt']}: prediction NPY byte length is incomplete")
        bindings.append(
            {
                "attempt": export["attempt"],
                "run": run,
                "checkpoint_sha256": checkpoint_sha256,
                "prediction_sha256": prediction_sha256,
                "sidecar_sha256": sha256_file(sidecar_path),
                "split_manifest_sha256": split_sha256,
                "row_groups_sha256": hashlib.sha256(
                    np.asarray(row_groups, dtype="<i8").tobytes()
                ).hexdigest(),
                "shape": list(shape),
                "dtype": str(dtype),
                "bytes": prediction_path.stat().st_size,
            }
        )
    return bindings


def _grader_command(plan: Mapping[str, object]) -> list[str]:
    """Build the one frozen C0 grading command from the immutable source path."""
    frozen = plan["frozen_sources"]
    repository_root = Path(str(frozen["repository_root"])).resolve(strict=True)
    return [
        sys.executable,
        str(repository_root / str(frozen["grader"]["path"])),
        "--artifact-root",
        str(Path(str(plan["artifact_root"]))),
        "--candidate-prefix",
        "bd26_c0",
        "--candidate-id",
        "C0",
        "--repository-root",
        str(repository_root),
        "--output-dir",
        str(Path(str(plan["output_dir"]))),
    ]


def _run_grader(
    plan: Mapping[str, object], runtime: Path, receipt: Path, state: dict[str, object]
) -> None:
    """Start the grader under a race-free admission-barrier check and monitor it."""
    output_dir = Path(str(plan["output_dir"]))
    if output_dir.exists():
        raise GateError("grader output already exists; refusing to overwrite or repeat")
    log_path = runtime / "logs" / "c0-eval-gate-v1.stderr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _grader_command(plan)
    with log_path.open("xb") as stderr_log, locked(runtime):
        if (runtime / "STOP_ADMISSION.json").exists():
            _publish(receipt, state, status="drained", reason="admission closed before grading")
            return
        if output_dir.exists():
            raise GateError("grader output appeared before launch; refusing to overwrite")
        _publish(receipt, state, status="grading", grader_command=command)
        process = subprocess.Popen(
            command,
            cwd=str(Path(str(plan["frozen_sources"]["repository_root"]))),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_log,
        )

    while process.poll() is None:
        _publish(receipt, state, status="grading", evaluator_pid=process.pid)
        time.sleep(float(plan.get("poll_seconds", 15)))
    exit_status = process.returncode
    if exit_status != 0:
        raise GateError(
            f"frozen evaluator exited with status {exit_status}; no retry was attempted"
        )

    report_files = {
        name: output_dir / name for name in ("report.json", "report.md", "components.json")
    }
    if not output_dir.is_dir() or any(not path.is_file() for path in report_files.values()):
        raise GateError("frozen evaluator exited successfully without all expected reports")
    report_hashes = {name: sha256_file(path) for name, path in report_files.items()}
    _publish(
        receipt,
        state,
        status="finished",
        evaluator_exit_status=exit_status,
        evaluator_stderr_sha256=sha256_file(log_path),
        report_sha256=report_hashes,
    )


def run_gate(
    plan_path: Path,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    """Wait for exact C0 exports, then run the pinned evaluator once."""
    plan_bytes = plan_path.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise GateError("unsupported evaluation-gate plan")
    receipt = Path(str(plan["receipt"]))
    runtime = Path(str(plan["runtime"]))
    state: dict[str, object] = {
        "plan": str(plan_path.resolve()),
        "plan_sha256": plan_sha256,
        "status": "starting",
        "evaluator_commit": plan["frozen_sources"]["commit"],
        "exports": [],
    }
    _create_receipt(receipt, state | {"heartbeat_utc": _timestamp()})

    try:
        if Path(str(plan["output_dir"])).exists():
            raise GateError("grader output already exists; refusing to overwrite or repeat")
        _verify_frozen_sources(plan)
        _publish(receipt, state, status="waiting_for_exports")
        while True:
            if plan_path.read_bytes() != plan_bytes:
                raise GateError("gate plan changed after admission")
            admission_closed = (runtime / "STOP_ADMISSION.json").exists()
            decision, attempt_states = _inspect_attempts(
                runtime, plan["exports"], now_fn(), admission_closed
            )
            _publish(receipt, state, status=decision, exports=attempt_states)
            if decision == "waiting":
                sleep_fn(float(plan.get("poll_seconds", 15)))
                continue
            if decision == "drained":
                _publish(
                    receipt,
                    state,
                    status="drained",
                    reason="admission closed before evaluation started",
                )
                return state
            break

        _publish(receipt, state, status="validating_exports")
        _verify_frozen_sources(plan)
        cache_bindings = _cache_bindings(plan)
        _publish(receipt, state, cache_bindings=cache_bindings)
        if plan_path.read_bytes() != plan_bytes:
            raise GateError("gate plan changed before evaluator launch")
        _verify_frozen_sources(plan)
        _run_grader(plan, runtime, receipt, state)
        return state
    except (GateError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        _publish(receipt, state, status="needs_review", error=f"{type(error).__name__}: {error}")
        raise


def main() -> int:
    """Run one registered, CPU-only C0 evaluator gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        verify_controller_membership()
        receipt = run_gate(args.plan.resolve(strict=True))
    except (GateError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        LOG.error("C0 evaluator gate needs review: %s", error)
        return 1
    LOG.info("C0 evaluator gate ended with receipt status %s", receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
