#!/usr/bin/env python3
"""Submit one sealed DataSphere profile and keep ownership through collection."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

PROJECT_ID = "bt1dqf9jg0tuhggcrj4k"
EXPECTED_JOB_NAME = "computer-vision-homework-rank01-profile"
SUBMIT_TIMEOUT_SECONDS = 300
CLI_TIMEOUT_SECONDS = 60
DOWNLOAD_TIMEOUT_SECONDS = 600
CANCEL_AFTER_SECONDS = 3480
POLL_SECONDS = 30
TERMINAL_STATUSES = {"SUCCESS", "ERROR", "CANCELLED"}
VALID_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,200}\Z")


class ControllerError(RuntimeError):
    """A safe-to-report controller failure without provider output."""


class FailClosed(ControllerError):
    """Local state is ambiguous, so no provider mutation is safe."""


class Runner(Protocol):
    """Minimal subprocess interface for offline controller tests."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess[str]: ...


def run_quietly(
    command: Sequence[str], *, cwd: Path, env: dict[str, str], timeout: int
) -> subprocess.CompletedProcess[str]:
    """Run the provider CLI without preserving stdout, stderr, or credentials."""
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
        text=True,
    )


def utc_now() -> str:
    """Return a timezone-aware timestamp for controller receipts."""
    return datetime.now(timezone(timedelta(0))).isoformat()


def sha256_file(path: Path) -> str:
    """Hash a small job configuration in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_single_json(path: Path, label: str) -> dict[str, object]:
    """Load the CLI's object or one-row JSON response and validate its shape."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FailClosed(
            f"{label} is absent or invalid; manual reconciliation is required"
        ) from error
    if isinstance(value, list):
        if len(value) != 1:
            raise FailClosed(f"{label} has an unexpected row count")
        value = value[0]
    if not isinstance(value, dict):
        raise FailClosed(f"{label} is not a JSON object")
    return value


def validate_config(path: Path) -> str:
    """Ensure the configured job still names the sealed rank-1 g2.1 profile."""
    if path.is_symlink() or not path.is_file():
        raise FailClosed("private DataSphere config is missing or is a symlink")
    content = path.read_text(encoding="utf-8")
    name_lines = re.findall(r"^name:\s*(\S+)\s*$", content, flags=re.MULTILINE)
    gpu_lines = re.findall(r"^cloud-instance-type:\s*(\S+)\s*$", content, flags=re.MULTILINE)
    if name_lines != [EXPECTED_JOB_NAME] or gpu_lines != ["g2.1"]:
        raise FailClosed("config job name or GPU type differs from the sealed profile")
    return sha256_file(path)


def json_bytes(value: dict[str, object]) -> bytes:
    """Serialize a receipt deterministically for private atomic writes."""
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def atomic_write(path: Path, value: dict[str, object]) -> None:
    """Write a mode-0600 receipt atomically and sync its directory entry."""
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp.exists():
            temp.unlink()


class ProfileController:
    """Own one DataSphere job from submit through terminal output download."""

    def __init__(
        self,
        *,
        cli: Path,
        profile: str | None,
        job_dir: Path,
        config: Path,
        runner: Runner = run_quietly,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: int = POLL_SECONDS,
        cancel_after_seconds: int = CANCEL_AFTER_SECONDS,
    ) -> None:
        self.cli = cli.resolve()
        self.profile = profile
        self.job_dir = Path(os.path.abspath(job_dir))
        self.config = Path(os.path.abspath(config))
        self.runner = runner
        self.clock = clock
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self.cancel_after_seconds = cancel_after_seconds
        self.state_path = self.job_dir / "controller-state.json"
        self.execution_path = self.job_dir / "execution.json"
        self.status_path = self.job_dir / "provider-status.json"
        self.download_dir = self.job_dir / "downloaded-files"

    def run(self) -> int:
        """Run or resume the unique job while holding the controller lock."""
        self._prepare_private_dir()
        lock_path = self.job_dir / "controller.lock"
        fd = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ControllerError(
                    "another controller already owns this job directory"
                ) from error
            state = self._load_or_submit()
            if state["phase"] == "COMPLETE":
                return 0
            state = self._monitor(state)
            return self._download(state)
        finally:
            os.close(fd)

    def _prepare_private_dir(self) -> None:
        self.job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.job_dir.is_symlink() or not self.job_dir.is_dir():
            raise FailClosed("job directory is not a real directory")
        if hasattr(os, "getuid") and self.job_dir.stat().st_uid != os.getuid():
            raise FailClosed("job directory is owned by a different user")
        os.chmod(self.job_dir, 0o700)
        if not self.cli.is_file() or not os.access(self.cli, os.X_OK):
            raise FailClosed("configured DataSphere CLI is missing or not executable")
        if self.profile is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.profile):
            raise FailClosed("yc profile name contains unsupported characters")
        lock_path = self.job_dir / "controller.lock"
        for path in (
            self.state_path,
            self.execution_path,
            self.status_path,
            self.download_dir,
            lock_path,
        ):
            if path.is_symlink():
                raise FailClosed(f"controller artifact path is a symlink: {path.name}")
        if self.profile is None and not os.environ.get("YC_IAM_TOKEN"):
            raise FailClosed("provide a verified yc profile or a short-lived IAM token")
        if self.state_path.exists():
            os.chmod(self.state_path, 0o600)
        if lock_path.exists():
            os.chmod(lock_path, 0o600)

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        for key in ("YC_TOKEN", "YC_OAUTH_TOKEN"):
            env.pop(key, None)
        if self.profile is not None:
            env.pop("YC_IAM_TOKEN", None)
            env["YC_AUTH"] = "yc"
        return env

    def _base_command(self) -> list[str]:
        logs = self.job_dir / "provider-logs"
        logs.mkdir(mode=0o700, exist_ok=True)
        if logs.is_symlink():
            raise FailClosed("provider log directory is a symlink")
        os.chmod(logs, 0o700)
        command = [str(self.cli)]
        if self.profile is not None:
            command.extend(["--profile", self.profile])
        command.extend(["--log-level", "ERROR", "--log-dir", str(logs)])
        return command

    def _invoke(self, command: list[str], timeout: int) -> None:
        try:
            result = self.runner(
                command,
                cwd=self.job_dir,
                env=self._environment(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ControllerError("provider CLI call timed out") from error
        except OSError as error:
            raise ControllerError(
                f"provider CLI could not start ({type(error).__name__})"
            ) from error
        if result.returncode != 0:
            raise ControllerError(f"provider CLI returned exit status {result.returncode}")

    def _load_or_submit(self) -> dict[str, object]:
        if self.state_path.exists():
            state = read_single_json(self.state_path, "controller state")
            if state.get("project_id") != PROJECT_ID:
                raise FailClosed("saved state has a different project id")
            if state.get("job_name") != EXPECTED_JOB_NAME:
                raise FailClosed("saved state has a different job name")
            phase = state.get("phase")
            if phase == "AMBIGUOUS_SUBMISSION":
                raise FailClosed(
                    "submission identity is unknown; inspect manually without name-based cancel"
                )
            if phase not in {
                "SUBMITTING",
                "MONITORING",
                "PROVIDER_CANCELLING",
                "CANCEL_REQUESTED",
                "TERMINAL",
                "COMPLETE",
            }:
                raise FailClosed("saved controller phase is not recognized")
            if not isinstance(state.get("cancel_at_epoch"), (int, float)):
                raise FailClosed("saved state has no durable cancellation deadline")
            if phase == "SUBMITTING":
                self._recover_submission(state)
            elif not isinstance(state.get("job_id"), str) or not VALID_JOB_ID.fullmatch(
                str(state.get("job_id"))
            ):
                raise FailClosed("saved state has no exact job id")
            return state

        if self.execution_path.exists():
            raise FailClosed("execution output exists without controller state; inspect manually")
        config_hash = validate_config(self.config)
        now = self.clock()
        state: dict[str, object] = {
            "schema_version": 1,
            "phase": "SUBMITTING",
            "project_id": PROJECT_ID,
            "job_name": EXPECTED_JOB_NAME,
            "config_path": str(self.config),
            "config_sha256": config_hash,
            "submit_started_epoch": now,
            "submit_started_utc": utc_now(),
            "cancel_at_epoch": now + self.cancel_after_seconds,
            "cancel_after_seconds": self.cancel_after_seconds,
        }
        atomic_write(self.state_path, state)
        command = self._base_command() + [
            "project",
            "job",
            "execute",
            "-p",
            PROJECT_ID,
            "-c",
            str(self.config),
            "--async",
            "-o",
            str(self.execution_path),
        ]
        try:
            self._invoke(command, SUBMIT_TIMEOUT_SECONDS)
        except ControllerError as error:
            if not self.execution_path.exists():
                state.update(
                    phase="AMBIGUOUS_SUBMISSION",
                    submission_error=str(error),
                    updated_at_utc=utc_now(),
                )
                atomic_write(self.state_path, state)
                raise FailClosed(
                    "submit outcome is uncertain and has no exact job id; refusing a duplicate"
                ) from error
        self._recover_submission(state)
        return state

    def _recover_submission(self, state: dict[str, object]) -> None:
        try:
            response = read_single_json(self.execution_path, "execute response")
        except FailClosed as error:
            state.update(
                phase="AMBIGUOUS_SUBMISSION",
                submission_error=str(error),
                updated_at_utc=utc_now(),
            )
            atomic_write(self.state_path, state)
            raise
        job_id = response.get("job_id")
        operation_id = response.get("operation_id")
        if not isinstance(job_id, str) or not VALID_JOB_ID.fullmatch(job_id):
            state.update(
                phase="AMBIGUOUS_SUBMISSION",
                submission_error="execute response has no valid exact job id",
                updated_at_utc=utc_now(),
            )
            atomic_write(self.state_path, state)
            raise FailClosed("execute response has no valid exact job id")
        if not isinstance(operation_id, str) or not VALID_JOB_ID.fullmatch(operation_id):
            operation_id = None
        state.update(
            phase="MONITORING",
            job_id=job_id,
            operation_id=operation_id,
            submitted_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        atomic_write(self.state_path, state)

    def _read_provider_status(self, state: dict[str, object]) -> str:
        job_id = state["job_id"]
        command = self._base_command() + [
            "project",
            "job",
            "get",
            "--id",
            str(job_id),
            "--format",
            "json",
            "-o",
            str(self.status_path),
        ]
        self._invoke(command, CLI_TIMEOUT_SECONDS)
        response = read_single_json(self.status_path, "provider status")
        if response.get("id") != job_id or response.get("name") != EXPECTED_JOB_NAME:
            raise FailClosed("provider status does not match the persisted job identity")
        status = response.get("status")
        if not isinstance(status, str) or status not in {
            "CREATING",
            "PREPARING",
            "EXECUTING",
            "UPLOADING_OUTPUT",
            "CANCELLING",
            *TERMINAL_STATUSES,
        }:
            raise FailClosed("provider returned an unknown job status")
        return status

    def _cancel_exact_id(self, state: dict[str, object]) -> None:
        state.update(
            last_cancel_attempt_epoch=self.clock(),
            last_cancel_attempt_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        atomic_write(self.state_path, state)
        command = self._base_command() + [
            "project",
            "job",
            "cancel",
            "--id",
            str(state["job_id"]),
        ]
        self._invoke(command, CLI_TIMEOUT_SECONDS)
        state.update(
            cancel_requested=True,
            cancel_requested_at_utc=utc_now(),
            phase="CANCEL_REQUESTED",
            updated_at_utc=utc_now(),
        )
        atomic_write(self.state_path, state)

    def _monitor(self, state: dict[str, object]) -> dict[str, object]:
        if state.get("phase") == "TERMINAL":
            return state
        failures = 0
        while True:
            try:
                status = self._read_provider_status(state)
                failures = 0
            except FailClosed:
                raise
            except ControllerError as error:
                failures += 1
                state.update(
                    last_poll_error=str(error),
                    last_poll_error_at_utc=utc_now(),
                    consecutive_poll_errors=failures,
                )
                atomic_write(self.state_path, state)
                if self._cancel_due(state):
                    try:
                        self._cancel_exact_id(state)
                    except FailClosed:
                        raise
                    except ControllerError as cancel_error:
                        state.update(last_cancel_error=str(cancel_error), updated_at_utc=utc_now())
                        atomic_write(self.state_path, state)
                self.sleep(min(60, 5 * (2 ** min(failures - 1, 4))))
                continue

            state.update(
                provider_status=status,
                last_status_at_utc=utc_now(),
                consecutive_poll_errors=0,
            )
            if status in TERMINAL_STATUSES:
                state.update(phase="TERMINAL", terminal_at_utc=utc_now())
                atomic_write(self.state_path, state)
                return state
            if status == "CANCELLING":
                state.update(
                    provider_cancelling_seen=True,
                    phase=(
                        "CANCEL_REQUESTED"
                        if state.get("cancel_requested")
                        else "PROVIDER_CANCELLING"
                    ),
                )
                atomic_write(self.state_path, state)
            if self._cancel_due(state):
                try:
                    self._cancel_exact_id(state)
                except FailClosed:
                    raise
                except ControllerError as error:
                    state.update(last_cancel_error=str(error), updated_at_utc=utc_now())
                    atomic_write(self.state_path, state)
            self.sleep(self.poll_seconds)

    def _cancel_due(self, state: dict[str, object]) -> bool:
        if self.clock() < float(state["cancel_at_epoch"]):
            return False
        if not state.get("cancel_requested"):
            last_attempt = state.get("last_cancel_attempt_epoch")
            return last_attempt is None or self.clock() - float(last_attempt) >= 60
        return self.clock() - float(state.get("last_cancel_attempt_epoch", 0)) >= 60

    def _download(self, state: dict[str, object]) -> int:
        if state.get("phase") == "COMPLETE":
            return 0
        if (
            state.get("phase") != "TERMINAL"
            or state.get("provider_status") not in TERMINAL_STATUSES
        ):
            raise FailClosed("output collection was requested before terminal status")
        self.download_dir.mkdir(mode=0o700, exist_ok=True)
        if self.download_dir.is_symlink():
            raise FailClosed("download output directory is a symlink")
        os.chmod(self.download_dir, 0o700)
        command = self._base_command() + [
            "project",
            "job",
            "download-files",
            "--id",
            str(state["job_id"]),
            "--with-logs",
            "--with-diagnostics",
            "--output-dir",
            str(self.download_dir),
        ]
        try:
            self._invoke(command, DOWNLOAD_TIMEOUT_SECONDS)
        except ControllerError as error:
            state.update(download_error=str(error), download_error_at_utc=utc_now())
            atomic_write(self.state_path, state)
            raise ControllerError(
                "terminal outputs could not be downloaded; systemd may retry"
            ) from error
        state.update(phase="COMPLETE", downloaded_at_utc=utc_now())
        atomic_write(self.state_path, state)
        return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the one-shot service configuration without accepting credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cli", required=True, type=Path, help="Absolute DataSphere CLI executable"
    )
    parser.add_argument("--profile", help="Existing yc profile used for refreshable auth")
    parser.add_argument(
        "--job-dir", required=True, type=Path, help="Private, durable controller state directory"
    )
    parser.add_argument("--config", required=True, type=Path, help="Private sealed config.yaml")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the durable one-job controller and report only sanitized failures."""
    args = parse_args(argv)
    try:
        return ProfileController(
            cli=args.cli,
            profile=args.profile,
            job_dir=args.job_dir,
            config=args.config,
        ).run()
    except FailClosed as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 78
    except ControllerError as error:
        print(f"CONTROLLER_ERROR: {error}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
