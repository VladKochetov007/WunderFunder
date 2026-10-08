"""Run queued experiments with staged data, managing process lifecycle and heartbeats."""

from __future__ import annotations

import argparse
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


def _atomic_json_write(path: Path, obj: Any) -> None:
    """Write JSON atomically via temp file + replace."""
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(obj, tmp)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _compute_sha256(data: bytes) -> str:
    """Return hex digest of SHA256(data)."""
    return hashlib.sha256(data).hexdigest()


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
    ) -> None:
        """Initialize worker with queue directory and exit conditions."""
        self.queue_dir = queue_dir.resolve()
        self.pending_dir = self.queue_dir / "pending"
        self.running_dir = self.queue_dir / "running"
        self.done_dir = self.queue_dir / "done"
        self.failed_dir = self.queue_dir / "failed"
        self.logs_dir = self.queue_dir / "logs"
        self.results_path = self.queue_dir / "results.jsonl"
        self.heartbeat_path = self.queue_dir / "worker_heartbeat.json"
        self.stop_path = self.queue_dir / "STOP"

        self.poll_seconds = poll_seconds
        self.idle_exit_seconds = idle_exit_seconds
        self.once = once
        self.max_runs = max_runs

        self.pid = os.getpid()
        self.host = socket.gethostname()
        self.start_time = datetime.now(UTC).isoformat()
        self.runs_completed = 0
        self.current_run_id: str | None = None
        self.current_step: str | None = None
        self.last_idle_check = time.time()
        self.terminating = False

        self._ensure_dirs()
        self._recover_orphans()

    def _ensure_dirs(self) -> None:
        """Create queue directories if missing."""
        for d in [self.pending_dir, self.running_dir, self.done_dir, self.failed_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _recover_orphans(self) -> None:
        """Move any files in running/ to failed/ with restart status."""
        for spec_path in sorted(self.running_dir.glob("*.json")):
            run_id = spec_path.stem
            try:
                failed_path = self.failed_dir / spec_path.name
                spec_path.rename(failed_path)
                self._write_result(
                    run_id=run_id,
                    spec_sha256="",
                    arm=None,
                    hypothesis=None,
                    status="worker_restart_orphan",
                    steps=[],
                    expected_outputs=[],
                )
                LOG.info("Recovered orphan %s", run_id)
            except Exception as e:
                LOG.error("Failed to recover orphan %s: %s", run_id, e)

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
            return None
        specs.sort(key=lambda p: (self._get_priority(p), p.name))
        return specs[0]

    def _get_priority(self, spec_path: Path) -> int:
        """Extract priority from spec, default 100."""
        try:
            spec = json.loads(spec_path.read_bytes())
            return int(spec.get("priority", 100))
        except Exception:
            return 100

    def _run_spec(self, spec_path: Path) -> tuple[str, bool]:
        """Execute a run spec file. Returns (run_id, success)."""
        spec_bytes = spec_path.read_bytes()
        spec = json.loads(spec_bytes)
        run_id = spec.get("run_id", "")
        spec_sha256 = _compute_sha256(spec_bytes)

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

        if not steps_spec:
            self._write_result(
                run_id=run_id,
                spec_sha256=spec_sha256,
                arm=arm,
                hypothesis=hypothesis,
                status="invalid_spec",
                steps=[{"error": "no steps defined"}],
                expected_outputs=[],
            )
            return run_id, False

        self.current_run_id = run_id
        self._write_heartbeat()

        steps_records = []
        success = True

        for step_spec in steps_spec:
            step_name = step_spec.get("name", "unnamed")
            self.current_step = step_name
            self._write_heartbeat()

            step_record = self._run_step(run_id, step_name, step_spec)
            steps_records.append(step_record)

            if step_record.get("status") != "succeeded":
                success = False
                break

        self.current_run_id = None
        self.current_step = None
        self._write_heartbeat()

        expected_outputs_records = []
        if success:
            expected_outputs_records = self._record_expected_outputs(
                run_id, expected_outputs_spec
            )

        self._write_result(
            run_id=run_id,
            spec_sha256=spec_sha256,
            arm=arm,
            hypothesis=hypothesis,
            status="succeeded" if success else "failed",
            steps=steps_records,
            expected_outputs=expected_outputs_records,
        )

        return run_id, success

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

            try:
                proc.wait(timeout=timeout_seconds)
                exit_code = proc.returncode
                status = "timeout" if exit_code is None else "succeeded" if exit_code == 0 else "failed"
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
                    "status": "succeeded" if status == "succeeded" else "failed" if status == "failed" else "timeout",
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
                        record["sha256"] = _compute_sha256(path.read_bytes())
                else:
                    record["exists"] = False
            except Exception as e:
                record["error"] = str(e)

            records.append(record)
        return records

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGTERM/SIGINT by terminating current step."""
        self.terminating = True
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

                spec_path = self._select_next_run()

                if spec_path is None:
                    if self.idle_exit_seconds is not None and time.time() - self.last_idle_check > self.idle_exit_seconds:
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

                run_id, success = self._run_spec(running_path)

                if success:
                    dest = self.done_dir / running_path.name
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


def main() -> None:
    """Parse args and run queue worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True, help="Queue directory")
    parser.add_argument("--poll-seconds", type=int, default=10, help="Polling interval")
    parser.add_argument(
        "--idle-exit-seconds", type=int, help="Exit after idle this long (seconds)"
    )
    parser.add_argument("--once", action="store_true", help="Process at most one run")
    parser.add_argument("--max-runs", type=int, help="Exit after N runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    worker = QueueWorker(
        queue_dir=args.queue,
        poll_seconds=args.poll_seconds,
        idle_exit_seconds=args.idle_exit_seconds,
        once=args.once,
        max_runs=args.max_runs,
    )

    sys.exit(worker.run())


if __name__ == "__main__":
    main()
