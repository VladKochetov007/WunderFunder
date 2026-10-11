#!/usr/bin/env python3
"""Wake the active research Codex thread from a bounded cron entry."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CONFIG_SCHEMA = "wnn.research-monitor-cron.v1"
QUEUE_SCHEMA = "wnn.research-monitor-queued.v1"
EVENT_SCHEMA = "wnn.research-monitor-event.v1"
MAX_PROMPT_BYTES = 262_144
QUEUE_TIMEOUT_SECONDS = 90
COALESCE_SECONDS = 2 * 60 * 60
CHILD_ENV_KEYS = ("HOME", "PATH", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
QUEUED_MESSAGE_PATTERN = re.compile(r"\bQueued message ([\w.-]+) for thread\b", re.I)
QUEUE_ID_PATTERN = re.compile(r"\b(?:queue|message)[_-]?id\s*[:=]\s*([\w.-]+)", re.I)


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    """Validated configuration for one research-session wakeup."""

    thread_id: str
    cwd: Path
    prompt_path: Path
    codex_path: Path
    state_dir: Path


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


def utc_text(value: datetime) -> str:
    """Format an aware timestamp as ISO-8601 UTC with a Z suffix."""
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_utc(value: object, name: str) -> datetime:
    """Parse an ISO-8601 timestamp and reject values without a timezone."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def _configured_path(value: object, name: str, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    resolved = path.resolve(strict=must_exist)
    if name == "cwd" and (not resolved.is_dir()):
        raise ValueError("cwd must be an existing directory")
    if name in {"prompt_path", "codex_path"} and not resolved.is_file():
        raise ValueError(f"{name} must be an existing file")
    if name == "codex_path" and not os.access(resolved, os.X_OK):
        raise ValueError("codex_path must be executable")
    if name == "state_dir" and resolved.exists() and not resolved.is_dir():
        raise ValueError("state_dir must be a directory when it exists")
    return resolved


def load_config(path: Path) -> MonitorConfig:
    """Read and validate the small, credential-free cron configuration."""
    config_path = path.resolve(strict=True)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read UTF-8 JSON config: {type(error).__name__}") from error
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    expected_keys = {"schema", "thread_id", "cwd", "prompt_path", "codex_path", "state_dir"}
    if set(raw) != expected_keys:
        raise ValueError("config must contain exactly the documented fields")
    if raw["schema"] != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA}")
    thread_id = raw["thread_id"]
    if not isinstance(thread_id, str):
        raise ValueError("thread_id must be a UUID string")
    try:
        normalized_thread_id = str(uuid.UUID(thread_id))
    except ValueError as error:
        raise ValueError("thread_id must be a canonical UUID") from error
    if normalized_thread_id != thread_id:
        raise ValueError("thread_id must be a canonical lowercase UUID")
    return MonitorConfig(
        thread_id=thread_id,
        cwd=_configured_path(raw["cwd"], "cwd", must_exist=True),
        prompt_path=_configured_path(raw["prompt_path"], "prompt_path", must_exist=True),
        codex_path=_configured_path(raw["codex_path"], "codex_path", must_exist=True),
        state_dir=_configured_path(raw["state_dir"], "state_dir", must_exist=False),
    )


def read_prompt(path: Path) -> str:
    """Read the exact bounded UTF-8 message to submit to the Codex queue."""
    if path.stat().st_size > MAX_PROMPT_BYTES:
        raise ValueError(f"prompt exceeds {MAX_PROMPT_BYTES} bytes")
    prompt = path.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError(f"prompt exceeds {MAX_PROMPT_BYTES} bytes")
    return prompt


def _summary(value: str | bytes | None) -> dict[str, object]:
    if value is None:
        raw = b""
    elif isinstance(value, bytes):
        raw = value
    else:
        raw = value.encode("utf-8", errors="replace")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _optional_queue_id(stdout: str) -> str | None:
    try:
        decoded = json.loads(stdout)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        for key in ("queue_id", "message_id", "id"):
            value = decoded.get(key)
            if isinstance(value, str) and 0 < len(value) <= 256:
                return value
    match = QUEUED_MESSAGE_PATTERN.search(stdout[:4096])
    if match:
        return match.group(1)
    match = QUEUE_ID_PATTERN.search(stdout[:4096])
    return match.group(1) if match else None


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"), allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _append_event(state_dir: Path, event: dict[str, object]) -> None:
    event_path = state_dir / "events.jsonl"
    line = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(event_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def _event(
    config: MonitorConfig, status: str, at: datetime, **details: object
) -> dict[str, object]:
    return {
        "schema": EVENT_SCHEMA,
        "event_utc": utc_text(at),
        "status": status,
        "thread_id": config.thread_id,
        "cwd": str(config.cwd),
        "codex_path": str(config.codex_path),
        **details,
    }


def _read_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid receipt {path.name}: {type(error).__name__}") from error
    if not isinstance(value, dict):
        raise ValueError(f"receipt {path.name} must be a JSON object")
    return value


def _pending_wakeup_age(state_dir: Path) -> tuple[datetime, float] | None:
    queued = _read_json_object(state_dir / "LAST_QUEUED.json")
    if queued is None:
        return None
    if queued.get("schema") != QUEUE_SCHEMA:
        raise ValueError("LAST_QUEUED.json has an unexpected schema")
    queued_at = parse_utc(queued.get("accepted_utc"), "LAST_QUEUED.accepted_utc")
    completed = _read_json_object(state_dir / "LAST_COMPLETED.json")
    if completed is not None:
        completed_at = parse_utc(completed.get("completed_utc"), "LAST_COMPLETED.completed_utc")
        if completed_at > queued_at:
            return None
    age_seconds = (utc_now() - queued_at).total_seconds()
    return queued_at, age_seconds


def _child_environment() -> dict[str, str]:
    """Pass only the login/session variables Codex needs; omit other env values."""
    return {key: os.environ[key] for key in CHILD_ENV_KEYS if key in os.environ}


def _dry_run(config: MonitorConfig, prompt: str) -> int:
    metadata = {
        "status": "dry_run",
        "schema": CONFIG_SCHEMA,
        "thread_id": config.thread_id,
        "cwd": str(config.cwd),
        "prompt_path": str(config.prompt_path),
        "prompt_bytes": len(prompt.encode("utf-8")),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "codex_path": str(config.codex_path),
        "state_dir": str(config.state_dir),
        "timeout_seconds": QUEUE_TIMEOUT_SECONDS,
    }
    print(json.dumps(metadata, sort_keys=True))
    return 0


def run_wakeup(config: MonitorConfig, prompt: str) -> int:
    """Queue one bounded wakeup, coalescing recent unacknowledged requests."""
    config.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = config.state_dir / "wakeup.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            event = _event(config, "skipped_overlap", utc_now(), exit_code=None)
            _append_event(config.state_dir, event)
            print(json.dumps(event, sort_keys=True))
            return 0
        if (config.state_dir / "DISABLED").exists():
            event = _event(config, "skipped_disabled", utc_now(), exit_code=None)
            _append_event(config.state_dir, event)
            print(json.dumps(event, sort_keys=True))
            return 0
        try:
            pending = _pending_wakeup_age(config.state_dir)
        except ValueError as error:
            event = _event(
                config,
                "failed_receipt_validation",
                utc_now(),
                exit_code=None,
                failure=type(error).__name__,
            )
            _append_event(config.state_dir, event)
            print(json.dumps(event, sort_keys=True), file=sys.stderr)
            return 1
        if pending is not None:
            queued_at, age_seconds = pending
            if age_seconds < COALESCE_SECONDS:
                event = _event(
                    config,
                    "skipped_pending",
                    utc_now(),
                    exit_code=None,
                    last_queued_accepted_utc=utc_text(queued_at),
                    last_wake_age_seconds=round(age_seconds, 6),
                )
                _append_event(config.state_dir, event)
                print(json.dumps(event, sort_keys=True))
                return 0
        argv = [
            str(config.codex_path),
            "queue",
            "--thread",
            config.thread_id,
            "--message",
            prompt,
        ]
        accepted_at: datetime | None = None
        queue_id: str | None = None
        try:
            result = subprocess.run(
                argv,
                cwd=config.cwd,
                env=_child_environment(),
                timeout=QUEUE_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            exit_code = result.returncode
            stdout_summary = _summary(result.stdout)
            stderr_summary = _summary(result.stderr)
            if exit_code == 0:
                accepted_at = utc_now()
                queue_id = _optional_queue_id(result.stdout)
        except subprocess.TimeoutExpired as error:
            exit_code = 124
            stdout_summary = _summary(error.stdout)
            stderr_summary = _summary(error.stderr)
            failure = "timeout"
        except OSError as error:
            exit_code = 127
            stdout_summary = _summary(None)
            stderr_summary = _summary(None)
            failure = type(error).__name__
        else:
            failure = None if exit_code == 0 else "nonzero_exit"
        event_time = accepted_at if accepted_at is not None else utc_now()
        event = _event(
            config,
            "queued" if exit_code == 0 else "queue_failed",
            event_time,
            exit_code=exit_code,
            stdout_summary=stdout_summary,
            stderr_summary=stderr_summary,
            queue_id=queue_id,
            failure=failure,
        )
        if accepted_at is not None:
            receipt: dict[str, object] = {
                "schema": QUEUE_SCHEMA,
                "thread_id": config.thread_id,
                "accepted_utc": utc_text(accepted_at),
                "cwd": str(config.cwd),
                "codex_path": str(config.codex_path),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "exit_code": 0,
            }
            if queue_id is not None:
                receipt["queue_id"] = queue_id
            _atomic_json(config.state_dir / "LAST_QUEUED.json", receipt)
        _append_event(config.state_dir, event)
        print(json.dumps(event, sort_keys=True))
        return 0 if exit_code == 0 else exit_code
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        prompt = read_prompt(config.prompt_path)
    except (OSError, ValueError) as error:
        print(f"configuration error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        return _dry_run(config, prompt)
    try:
        return run_wakeup(config, prompt)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"wakeup failed: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
