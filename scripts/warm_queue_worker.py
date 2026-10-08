"""Run queued experiments with staged data, managing process lifecycle and heartbeats."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

LOG = logging.getLogger(__name__)
HEARTBEAT_INTERVAL_SECONDS = 30
PROGRESS_SCHEMA_VERSION = 1
SPEC_SHA256_RE = re.compile(r"[a-f0-9]{64}\Z")


def _atomic_json_write(path: Path, obj: Any) -> None:
    """Write JSON atomically via temp file + replace."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, suffix=".tmp") as tmp:
        json.dump(obj, tmp)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _compute_sha256(data: bytes) -> str:
    """Return hex digest of SHA256(data)."""
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    """Stream a file's SHA256 in 8 MiB blocks without holding it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _validate_run_id(run_id: str) -> bool:
    """Check run_id matches ^[A-Za-z0-9_.-]{1,96}$."""
    return bool(re.match(r"^[A-Za-z0-9_.-]{1,96}$", run_id))


class QueueWorker:
    """Manages a single-run queue with heartbeat and crash recovery."""

    def __init__(
        self,
        queue_dir: Path,
        poll_seconds: int = 10,
        idle_exit_seconds: int | None = None,
        once: bool = False,
        max_runs: int | None = None,
        allowed_spec_manifest: Path | None = None,
    ) -> None:
        """Initialize worker with queue directory and exit conditions."""
        self.queue_dir = queue_dir.resolve()
        self.pending_dir = self.queue_dir / "pending"
        self.running_dir = self.queue_dir / "running"
        self.done_dir = self.queue_dir / "done"
        self.failed_dir = self.queue_dir / "failed"
        self.logs_dir = self.queue_dir / "logs"
        self.progress_dir = self.queue_dir / "progress"
        self.results_path = self.queue_dir / "results.jsonl"
        self.heartbeat_path = self.queue_dir / "worker_heartbeat.json"
        self.stop_path = self.queue_dir / "STOP"

        self.poll_seconds = poll_seconds
        self.idle_exit_seconds = idle_exit_seconds
        self.once = once
        self.max_runs = max_runs
        self.allowed_spec_hashes = (
            None
            if allowed_spec_manifest is None
            else self._load_allowed_spec_manifest(allowed_spec_manifest)
        )
        self.admission_hold_path = self.queue_dir / "admission_hold.json"
        self._last_admission_hold: tuple[str, str | None, str | None] | None = None

        self.pid = os.getpid()
        self.host = socket.gethostname()
        self.start_time = datetime.now(UTC).isoformat()
        self.runs_completed = 0
        self.current_run_id: str | None = None
        self.current_step: str | None = None
        self.last_idle_check = time.time()
        self.terminating = False
        self.current_proc: subprocess.Popen[bytes] | None = None

        self._ensure_dirs()
        self._recover_orphans()

    @staticmethod
    def _load_allowed_spec_manifest(path: Path) -> dict[str, str]:
        """Load a strict run-id-to-SHA256 admission allowlist."""
        try:
            value = json.loads(path.read_bytes())
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot load allowed spec manifest: {path}") from error
        if not isinstance(value, dict) or not value:
            raise ValueError("allowed spec manifest must be a non-empty run-id-to-SHA256 object")
        allowed: dict[str, str] = {}
        for run_id, digest in value.items():
            if not isinstance(run_id, str) or not _validate_run_id(run_id):
                raise ValueError("allowed spec manifest contains an invalid run_id")
            if not isinstance(digest, str) or not SPEC_SHA256_RE.fullmatch(digest):
                raise ValueError(f"allowed spec manifest has an invalid SHA256 for {run_id}")
            allowed[run_id] = digest
        return allowed

    def _hold_unreviewed_spec(
        self, spec_path: Path, expected_sha256: str | None, observed_sha256: str | None
    ) -> None:
        """Leave an unreviewed pending spec untouched and persist its admission block."""
        signature = (spec_path.name, expected_sha256, observed_sha256)
        if signature == self._last_admission_hold:
            return
        self._last_admission_hold = signature
        _atomic_json_write(
            self.admission_hold_path,
            {
                "status": "held_unreviewed",
                "run_id": spec_path.stem,
                "path": str(spec_path),
                "expected_sha256": expected_sha256,
                "observed_sha256": observed_sha256,
                "observed_utc": datetime.now(UTC).isoformat(),
            },
        )
        LOG.error(
            "Holding unreviewed queue spec %s (expected=%s observed=%s)",
            spec_path.name,
            expected_sha256,
            observed_sha256,
        )

    def _read_admitted_spec(self, spec_path: Path) -> bytes | None:
        """Return only bytes matching this worker's frozen per-run admission hash."""
        try:
            spec_bytes = spec_path.read_bytes()
        except OSError:
            self._hold_unreviewed_spec(spec_path, None, None)
            return None
        if self.allowed_spec_hashes is None:
            return spec_bytes
        expected = self.allowed_spec_hashes.get(spec_path.stem)
        observed = _compute_sha256(spec_bytes)
        if expected is None or observed != expected:
            self._hold_unreviewed_spec(spec_path, expected, observed)
            return None
        return spec_bytes

    def _clear_admission_hold(self) -> None:
        """Persist resolution when all current pending specs match the allowlist."""
        if not self.admission_hold_path.is_file():
            return
        prior = json.loads(self.admission_hold_path.read_bytes())
        if prior.get("status") == "cleared":
            return
        _atomic_json_write(
            self.admission_hold_path,
            {
                "status": "cleared",
                "prior_run_id": prior.get("run_id"),
                "prior_expected_sha256": prior.get("expected_sha256"),
                "prior_observed_sha256": prior.get("observed_sha256"),
                "observed_utc": datetime.now(UTC).isoformat(),
            },
        )
        self._last_admission_hold = None

    def _ensure_dirs(self) -> None:
        """Create queue directories if missing."""
        for d in [
            self.pending_dir,
            self.running_dir,
            self.done_dir,
            self.failed_dir,
            self.logs_dir,
            self.progress_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _recover_orphans(self) -> None:
        """Requeue checkpointed work; fail closed for non-resumable orphans."""
        for spec_path in sorted(self.running_dir.glob("*.json")):
            run_id = spec_path.stem
            spec: Any = None
            spec_sha256 = ""
            try:
                spec_bytes = spec_path.read_bytes()
                spec = json.loads(spec_bytes)
                spec_sha256 = _compute_sha256(spec_bytes)
                progress = self._load_progress(run_id, spec_sha256)
                if isinstance(spec, dict) and progress is not None:
                    try:
                        ordered_steps = self._ordered_steps(spec.get("steps"))
                    except ValueError:
                        ordered_steps = []
                    if ordered_steps:
                        ordered_names = [step["name"] for step, _ in ordered_steps]
                        completed_records = progress["completed_steps"]
                        completed_name_list = [
                            record.get("name")
                            for record in completed_records
                            if isinstance(record, dict)
                            and record.get("status") == "succeeded"
                            and isinstance(record.get("name"), str)
                        ]
                        completed_names = {name for name in completed_name_list}
                        completed_history_valid = len(completed_name_list) == len(
                            completed_records
                        ) and len(completed_name_list) == len(completed_names)
                        current_step = progress.get("current_step")
                        if current_step is None:
                            completed_prefix = ordered_names[: len(completed_names)]
                            can_resume = completed_history_valid and completed_names == set(
                                completed_prefix
                            )
                        else:
                            current_index = next(
                                (
                                    index
                                    for index, name in enumerate(ordered_names)
                                    if name == current_step
                                ),
                                None,
                            )
                            current = (
                                None if current_index is None else ordered_steps[current_index][0]
                            )
                            checkpoint = (
                                None if current is None else self._resume_checkpoint(current)
                            )
                            prior_steps_completed = (
                                current_index is not None
                                and completed_history_valid
                                and completed_names == set(ordered_names[:current_index])
                            )
                            can_resume = checkpoint is not None and prior_steps_completed
                            if can_resume:
                                progress["resume_current_step"] = True
                        pending_path = self.pending_dir / spec_path.name
                        if can_resume and pending_path.is_file():
                            if pending_path.read_bytes() == spec_bytes:
                                pending_path.unlink()
                            else:
                                can_resume = False
                        if can_resume and not pending_path.exists():
                            self._write_progress(run_id, progress)
                            spec_path.rename(pending_path)
                            self._write_result(
                                run_id=run_id,
                                spec_sha256=spec_sha256,
                                arm=spec.get("arm"),
                                hypothesis=spec.get("hypothesis"),
                                status="worker_restart_resumable",
                                steps=[],
                                expected_outputs=[],
                            )
                            LOG.info("Requeued checkpointed orphan %s", run_id)
                            continue
                failed_path = self.failed_dir / spec_path.name
                spec_path.rename(failed_path)
                self._write_result(
                    run_id=run_id,
                    spec_sha256=spec_sha256,
                    arm=spec.get("arm") if isinstance(spec, dict) else None,
                    hypothesis=spec.get("hypothesis") if isinstance(spec, dict) else None,
                    status="worker_restart_orphan",
                    steps=[],
                    expected_outputs=[],
                )
                LOG.info("Recovered orphan %s", run_id)
            except Exception as e:
                LOG.error("Failed to recover orphan %s: %s", run_id, e)

    def _progress_path(self, run_id: str) -> Path:
        return self.progress_dir / f"{run_id}.json"

    def _load_progress(self, run_id: str, spec_sha256: str) -> dict[str, Any] | None:
        """Load progress only when it belongs to these exact frozen spec bytes."""
        path = self._progress_path(run_id)
        if not path.is_file():
            return None
        try:
            progress = json.loads(path.read_bytes())
        except (OSError, ValueError):
            return None
        if (
            not isinstance(progress, dict)
            or progress.get("schema_version") != PROGRESS_SCHEMA_VERSION
            or progress.get("run_id") != run_id
            or progress.get("spec_sha256") != spec_sha256
            or not isinstance(progress.get("completed_steps"), list)
            or not (
                progress.get("current_step") is None
                or isinstance(progress.get("current_step"), str)
            )
            or not isinstance(progress.get("resume_current_step"), bool)
        ):
            return None
        return progress

    def _write_progress(self, run_id: str, progress: dict[str, Any]) -> None:
        """Persist step boundaries before and after child execution."""
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(self._progress_path(run_id), progress)

    @staticmethod
    def _resume_checkpoint(step_spec: dict[str, Any]) -> Path | None:
        """Return a declared, nonempty resume checkpoint for one step."""
        value = step_spec.get("resume_state_path")
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            cwd = step_spec.get("cwd")
            path = (Path(cwd) if isinstance(cwd, str) else Path.cwd()) / path
        try:
            return path if path.is_file() and path.stat().st_size > 0 else None
        except OSError:
            return None

    @staticmethod
    def _ordered_steps(steps_spec: object) -> list[tuple[dict[str, Any], list[str]]]:
        """Validate step names/dependencies and return a stable topological order.

        Missing ``depends_on`` retains the historical serial-list behavior. An
        explicit empty list declares an independent step.
        """
        if not isinstance(steps_spec, list) or not steps_spec:
            raise ValueError("no steps defined")
        steps: list[dict[str, Any]] = []
        names: list[str] = []
        for index, item in enumerate(steps_spec):
            if not isinstance(item, dict):
                raise ValueError(f"step {index} must be an object")
            name = item.get("name", "unnamed")
            if not isinstance(name, str) or not name:
                raise ValueError(f"step {index} has an invalid name")
            if name in names:
                raise ValueError(f"duplicate step name: {name}")
            steps.append(item)
            names.append(name)

        dependencies: dict[str, list[str]] = {}
        for index, step in enumerate(steps):
            name = names[index]
            if "depends_on" not in step:
                deps = names[index - 1 : index] if index else []
            else:
                raw_deps = step["depends_on"]
                if not isinstance(raw_deps, list) or any(
                    not isinstance(dep, str) for dep in raw_deps
                ):
                    raise ValueError(f"{name}: depends_on must be a list of step names")
                deps = list(raw_deps)
            if len(set(deps)) != len(deps):
                raise ValueError(f"{name}: depends_on contains duplicates")
            if any(dep not in names for dep in deps):
                raise ValueError(f"{name}: depends_on names an unknown step")
            if name in deps:
                raise ValueError(f"{name}: a step cannot depend on itself")
            dependencies[name] = deps

        remaining = set(names)
        ordered_names: list[str] = []
        while remaining:
            ready = [
                name
                for name in names
                if name in remaining and all(dep in ordered_names for dep in dependencies[name])
            ]
            if not ready:
                raise ValueError("step dependencies contain a cycle")
            ordered_names.append(ready[0])
            remaining.remove(ready[0])
        step_by_name = dict(zip(names, steps, strict=True))
        return [(step_by_name[name], dependencies[name]) for name in ordered_names]

    def _write_heartbeat(self) -> None:
        """Write worker heartbeat atomically."""
        heartbeat = {
            "pid": self.pid,
            "host": self.host,
            "current_run_id": self.current_run_id,
            "current_step": self.current_step,
            "runs_completed": self.runs_completed,
            "last_update_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json_write(self.heartbeat_path, heartbeat)

    def _write_result(
        self,
        run_id: str,
        spec_sha256: str,
        arm: str | None,
        hypothesis: str | None,
        status: str,
        steps: list[dict[str, Any]],
        expected_outputs: list[dict[str, Any]],
    ) -> None:
        """Append result line to results.jsonl with atomicity."""
        result = {
            "run_id": run_id,
            "spec_sha256": spec_sha256,
            "arm": arm,
            "hypothesis": hypothesis,
            "status": status,
            "steps": steps,
            "expected_outputs": expected_outputs,
            "worker_host": self.host,
            "worker_start_time": self.start_time,
        }
        line = json.dumps(result)
        with open(self.results_path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _select_next_run(self) -> Path | None:
        """Select run with lowest (priority, filename). Returns None if STOP present."""
        if self.stop_path.exists():
            return None
        specs = list(self.pending_dir.glob("*.json"))
        if not specs:
            if self.allowed_spec_hashes is not None:
                self._clear_admission_hold()
            return None
        if self.allowed_spec_hashes is not None:
            for spec_path in sorted(specs, key=lambda path: path.name):
                if self._read_admitted_spec(spec_path) is None:
                    return None
            self._clear_admission_hold()
        specs.sort(key=lambda p: (self._get_priority(p), p.name))
        return specs[0]

    def _get_priority(self, spec_path: Path) -> int:
        """Extract priority from spec, default 100."""
        try:
            spec = json.loads(spec_path.read_bytes())
            return int(spec.get("priority", 100))
        except Exception:
            return 100

    def _run_spec(self, spec_path: Path, spec_bytes: bytes | None = None) -> tuple[str, bool]:
        """Execute a run spec file. Returns (run_id, success)."""
        spec_bytes = spec_path.read_bytes() if spec_bytes is None else spec_bytes
        spec_sha256 = _compute_sha256(spec_bytes)
        try:
            spec = json.loads(spec_bytes)
            if not isinstance(spec, dict):
                raise ValueError("spec must be a JSON object")
        except ValueError as error:
            self._write_result(
                run_id=spec_path.stem,
                spec_sha256=spec_sha256,
                arm=None,
                hypothesis=None,
                status="invalid_spec",
                steps=[{"error": f"unreadable spec: {error}"}],
                expected_outputs=[],
            )
            return spec_path.stem, False
        run_id = spec.get("run_id", "")

        if not _validate_run_id(run_id):
            self._write_result(
                run_id=run_id,
                spec_sha256=spec_sha256,
                arm=spec.get("arm"),
                hypothesis=spec.get("hypothesis"),
                status="invalid_spec",
                steps=[],
                expected_outputs=[],
            )
            return run_id, False

        if run_id != spec_path.stem:
            self._write_result(
                run_id=run_id,
                spec_sha256=spec_sha256,
                arm=spec.get("arm"),
                hypothesis=spec.get("hypothesis"),
                status="invalid_spec",
                steps=[{"error": "run_id does not match filename"}],
                expected_outputs=[],
            )
            return run_id, False

        arm = spec.get("arm")
        hypothesis = spec.get("hypothesis")
        steps_spec = spec.get("steps", [])
        expected_outputs_spec = spec.get("expected_outputs", [])

        try:
            ordered_steps = self._ordered_steps(steps_spec)
        except ValueError as error:
            self._write_result(
                run_id=run_id,
                spec_sha256=spec_sha256,
                arm=arm,
                hypothesis=hypothesis,
                status="invalid_spec",
                steps=[{"error": str(error)}],
                expected_outputs=[],
            )
            return run_id, False
        if not isinstance(expected_outputs_spec, list) or any(
            not isinstance(path, str) for path in expected_outputs_spec
        ):
            self._write_result(
                run_id=run_id,
                spec_sha256=spec_sha256,
                arm=arm,
                hypothesis=hypothesis,
                status="invalid_spec",
                steps=[{"error": "expected_outputs must be a list of paths"}],
                expected_outputs=[],
            )
            return run_id, False

        progress = self._load_progress(run_id, spec_sha256) or {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "run_id": run_id,
            "spec_sha256": spec_sha256,
            "completed_steps": [],
            "current_step": None,
            "resume_current_step": False,
        }
        completed_by_name = {
            record["name"]: record
            for record in progress["completed_steps"]
            if isinstance(record, dict)
            and isinstance(record.get("name"), str)
            and record.get("status") == "succeeded"
        }

        self.current_run_id = run_id
        self._write_heartbeat()

        steps_records: list[dict[str, Any]] = []
        step_status: dict[str, str] = {}
        success = True
        worker_terminated = False

        for step_spec, dependencies in ordered_steps:
            step_name = step_spec.get("name", "unnamed")
            prior_record = completed_by_name.get(step_name)
            if prior_record is not None:
                steps_records.append(prior_record)
                step_status[step_name] = "succeeded"
                continue

            blocked_by = [dep for dep in dependencies if step_status.get(dep) != "succeeded"]
            if blocked_by:
                record = {
                    "name": step_name,
                    "status": "blocked",
                    "blocked_by": blocked_by,
                    "start_utc": datetime.now(UTC).isoformat(),
                    "end_utc": datetime.now(UTC).isoformat(),
                }
                steps_records.append(record)
                step_status[step_name] = "blocked"
                success = False
                continue

            resume_this_step = (
                progress.get("resume_current_step") is True
                and progress.get("current_step") == step_name
            )
            run_step_spec = step_spec
            if resume_this_step:
                checkpoint = self._resume_checkpoint(step_spec)
                if checkpoint is None:
                    step_record = {
                        "name": step_name,
                        "status": "failed",
                        "error": "declared resume checkpoint is missing or empty",
                        "start_utc": datetime.now(UTC).isoformat(),
                        "end_utc": datetime.now(UTC).isoformat(),
                    }
                    steps_records.append(step_record)
                    step_status[step_name] = "failed"
                    success = False
                    progress["resume_current_step"] = False
                    self._write_progress(run_id, progress)
                    continue
                try:
                    run_step_spec = self._step_with_resume_state(step_spec, checkpoint)
                except ValueError as error:
                    step_record = {
                        "name": step_name,
                        "status": "failed",
                        "error": str(error),
                        "start_utc": datetime.now(UTC).isoformat(),
                        "end_utc": datetime.now(UTC).isoformat(),
                    }
                    steps_records.append(step_record)
                    step_status[step_name] = "failed"
                    success = False
                    progress["resume_current_step"] = False
                    self._write_progress(run_id, progress)
                    continue

            self.current_step = step_name
            self._write_heartbeat()
            progress["current_step"] = step_name
            progress["resume_current_step"] = resume_this_step
            self._write_progress(run_id, progress)

            step_record = self._run_step(run_id, step_name, run_step_spec)
            steps_records.append(step_record)
            status = str(step_record.get("status"))
            step_status[step_name] = status
            if status == "succeeded":
                completed_by_name[step_name] = step_record
                progress["completed_steps"] = [
                    completed_by_name[step["name"]]
                    for step, _ in ordered_steps
                    if step["name"] in completed_by_name
                ]
                progress["current_step"] = None
                progress["resume_current_step"] = False
            else:
                success = False
                progress["current_step"] = step_name
                checkpoint = self._resume_checkpoint(step_spec)
                progress["resume_current_step"] = (
                    status == "worker_terminated" and checkpoint is not None
                )
                worker_terminated = status == "worker_terminated"
            self._write_progress(run_id, progress)
            self._write_heartbeat()
            if worker_terminated:
                break

        self.current_run_id = None
        self.current_step = None
        self._write_heartbeat()

        expected_outputs_records = self._record_expected_outputs(run_id, expected_outputs_spec)
        missing_outputs = [
            record["path"] for record in expected_outputs_records if not record.get("exists")
        ]
        if success and missing_outputs:
            success = False
            steps_records.append(
                {
                    "name": "expected_outputs",
                    "status": "failed",
                    "error": "missing expected artifacts: " + ", ".join(missing_outputs),
                    "start_utc": datetime.now(UTC).isoformat(),
                    "end_utc": datetime.now(UTC).isoformat(),
                }
            )

        self._write_result(
            run_id=run_id,
            spec_sha256=spec_sha256,
            arm=arm,
            hypothesis=hypothesis,
            status=(
                "succeeded" if success else "worker_terminated" if worker_terminated else "failed"
            ),
            steps=steps_records,
            expected_outputs=expected_outputs_records,
        )

        if success:
            with contextlib.suppress(FileNotFoundError):
                self._progress_path(run_id).unlink()

        return run_id, success

    @staticmethod
    def _step_with_resume_state(step_spec: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        """Return a step copy with the trainer's explicit resume flag appended."""
        argv = step_spec.get("argv")
        if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
            raise ValueError("resume-enabled step argv must be a list of strings")
        command = list(argv)
        checkpoint_value = str(checkpoint.resolve())
        if "--resume-state" in command:
            index = command.index("--resume-state")
            if index + 1 < len(command) and command[index + 1] == checkpoint_value:
                return {**step_spec, "argv": command}
            raise ValueError("resume-enabled step already declares a different --resume-state")
        if any(arg.startswith("--resume-state=") for arg in command):
            if f"--resume-state={checkpoint_value}" in command:
                return {**step_spec, "argv": command}
            raise ValueError("resume-enabled step already declares a different --resume-state")
        command.extend(["--resume-state", checkpoint_value])
        return {**step_spec, "argv": command}

    def _run_step(self, run_id: str, step_name: str, step_spec: dict[str, Any]) -> dict[str, Any]:
        """Execute a single step. Return step record."""
        argv = step_spec.get("argv", [])
        timeout_seconds = step_spec.get("timeout_seconds", 3600)
        env_overlay = step_spec.get("env", {})
        cwd = step_spec.get("cwd")

        log_path = self.logs_dir / f"{run_id}.{step_name}.log"

        step_record: dict[str, Any] = {
            "name": step_name,
            "start_utc": datetime.now(UTC).isoformat(),
        }

        try:
            env = os.environ.copy()
            env.update(env_overlay)

            start_time = time.time()
            with open(log_path, "a") as log_file:
                proc = subprocess.Popen(
                    argv,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=cwd,
                    start_new_session=True,
                )
            self.current_proc = proc

            try:
                deadline = time.monotonic() + timeout_seconds
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, timeout_seconds)
                    try:
                        proc.wait(timeout=min(HEARTBEAT_INTERVAL_SECONDS, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            raise
                        self._write_heartbeat()
                exit_code = proc.returncode
                status = (
                    "timeout" if exit_code is None else "succeeded" if exit_code == 0 else "failed"
                )
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        proc.wait()
                except ProcessLookupError:
                    pass
                status = "timeout"
                exit_code = "timeout"

            self.current_proc = None
            if self.terminating and status != "succeeded":
                status = "worker_terminated"
                exit_code = proc.returncode
            wall_seconds = time.time() - start_time

            try:
                usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                peak_rss_bytes = usage.ru_maxrss * 1024
            except Exception:
                peak_rss_bytes = 0

            step_record.update(
                {
                    "end_utc": datetime.now(UTC).isoformat(),
                    "wall_seconds": wall_seconds,
                    "exit_code": exit_code,
                    "status": status,
                    "peak_rss_bytes": peak_rss_bytes,
                }
            )

        except Exception as e:
            step_record.update(
                {
                    "end_utc": datetime.now(UTC).isoformat(),
                    "status": "failed",
                    "error": str(e),
                }
            )

        return step_record

    def _record_expected_outputs(
        self, run_id: str, expected_outputs_spec: list[str]
    ) -> list[dict[str, Any]]:
        """Record expected outputs: exist, size, and hash if <2GB."""
        records = []
        for path_str in expected_outputs_spec:
            path = Path(path_str)
            record: dict[str, Any] = {"path": path_str}

            try:
                if path.exists():
                    record["exists"] = True
                    size = path.stat().st_size
                    record["size"] = size
                    if size <= 2 * 1024**3:
                        record["sha256"] = _file_sha256(path)
                else:
                    record["exists"] = False
            except Exception as e:
                record["error"] = str(e)

            records.append(record)
        return records

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGTERM/SIGINT by terminating current step."""
        self.terminating = True
        proc = getattr(self, "current_proc", None)
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        if self.current_run_id:
            try:
                log_path = self.logs_dir / f"{self.current_run_id}.{self.current_step}.log"
                with open(log_path, "a") as f:
                    f.write(f"\n[Worker received signal {signum}, terminating]\n")
            except Exception:
                pass

    def run(self) -> int:
        """Main worker loop. Return exit code."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        try:
            while True:
                if self.terminating:
                    LOG.info("Worker terminated by signal")
                    return 130

                if self.max_runs is not None and self.runs_completed >= self.max_runs:
                    LOG.info("Reached max_runs=%d", self.max_runs)
                    return 0

                if self.stop_path.exists():
                    LOG.info("STOP file present; exiting before claiming a run")
                    self._write_heartbeat()
                    return 0

                spec_path = self._select_next_run()

                if spec_path is None:
                    if (
                        self.idle_exit_seconds is not None
                        and time.time() - self.last_idle_check > self.idle_exit_seconds
                    ):
                        LOG.info("Idle for %d seconds, exiting", self.idle_exit_seconds)
                        return 0
                    if self.once:
                        LOG.info("once mode with no pending runs")
                        return 0
                    time.sleep(self.poll_seconds)
                    continue

                self.last_idle_check = time.time()

                try:
                    running_path = self.running_dir / spec_path.name
                    spec_path.rename(running_path)
                except Exception as e:
                    LOG.error("Failed to claim run: %s", e)
                    time.sleep(1)
                    continue

                if self.allowed_spec_hashes is None:
                    run_id, success = self._run_spec(running_path)
                else:
                    admitted_bytes = self._read_admitted_spec(running_path)
                    if admitted_bytes is None:
                        try:
                            running_path.rename(self.pending_dir / running_path.name)
                        except OSError:
                            LOG.exception(
                                "Failed to return held spec to pending: %s", running_path.name
                            )
                            return 1
                        self._write_heartbeat()
                        time.sleep(self.poll_seconds)
                        continue
                    run_id, success = self._run_spec(running_path, spec_bytes=admitted_bytes)

                if success:
                    dest = self.done_dir / running_path.name
                elif self._can_resume_run(run_id):
                    dest = self.pending_dir / running_path.name
                else:
                    dest = self.failed_dir / running_path.name

                try:
                    running_path.rename(dest)
                except Exception as e:
                    LOG.error("Failed to finalize run %s: %s", run_id, e)

                self.runs_completed += 1
                self._write_heartbeat()

                if self.once:
                    return 0

        except KeyboardInterrupt:
            LOG.info("Interrupted by user")
            return 130
        except Exception as e:
            LOG.exception("Unexpected error: %s", e)
            return 1

    def _can_resume_run(self, run_id: str) -> bool:
        """Return true only for a worker-terminated step with a live checkpoint."""
        try:
            progress_path = self._progress_path(run_id)
            progress = json.loads(progress_path.read_bytes())
            if not isinstance(progress, dict) or progress.get("resume_current_step") is not True:
                return False
            current_step = progress.get("current_step")
            pending_path = self.pending_dir / f"{run_id}.json"
            spec_path = self.running_dir / f"{run_id}.json"
            active_spec = pending_path if pending_path.is_file() else spec_path
            spec = json.loads(active_spec.read_bytes())
            if (
                not isinstance(spec, dict)
                or progress.get("run_id") != run_id
                or progress.get("spec_sha256") != _compute_sha256(active_spec.read_bytes())
            ):
                return False
            step = next(
                (
                    item
                    for item in spec.get("steps", [])
                    if isinstance(item, dict) and item.get("name", "unnamed") == current_step
                ),
                None,
            )
            return isinstance(step, dict) and self._resume_checkpoint(step) is not None
        except (OSError, ValueError, TypeError):
            return False


def main() -> None:
    """Parse args and run queue worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True, help="Queue directory")
    parser.add_argument("--poll-seconds", type=int, default=10, help="Polling interval")
    parser.add_argument("--idle-exit-seconds", type=int, help="Exit after idle this long (seconds)")
    parser.add_argument("--once", action="store_true", help="Process at most one run")
    parser.add_argument("--max-runs", type=int, help="Exit after N runs")
    parser.add_argument(
        "--allowed-spec-manifest",
        type=Path,
        help="Optional exact JSON map of admitted run_id to spec SHA256",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    worker = QueueWorker(
        queue_dir=args.queue,
        poll_seconds=args.poll_seconds,
        idle_exit_seconds=args.idle_exit_seconds,
        once=args.once,
        max_runs=args.max_runs,
        allowed_spec_manifest=args.allowed_spec_manifest,
    )

    sys.exit(worker.run())


if __name__ == "__main__":
    main()
