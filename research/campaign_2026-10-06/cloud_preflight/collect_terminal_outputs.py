"""Collect terminal Kaggle outputs for the two root-pinned recovery kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from poll_provider import append_jsonl
    from poll_provider import collect as collect_provider_snapshot
except ImportError:  # pragma: no cover - package import for focused tests
    from .poll_provider import append_jsonl
    from .poll_provider import collect as collect_provider_snapshot

GIB = 1024**3
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name(
    "terminal_collection_targets_20261008_r1.json"
)
TERMINAL_STATES = {"COMPLETE", "COMPLETED", "ERROR", "FAILED", "CANCELED", "CANCELLED"}
MAX_ATTEMPTS = 3

# Configuration is deliberately restricted to these two already admitted jobs.
PINNED_TARGETS: dict[str, dict[str, Any]] = {
    "vladkochetov07/wnn-warm-cohort-smoke-20261008-v4": {
        "resource": "gpu",
        "metadata_id": "vladkochetov07/wnn-warm-cohort-smoke-20261008-v4",
        "root_admission": (
            "../bandit_continuation/controller_claude_20261008/"
            "KAGGLE_SMOKE4_ROOT_ADMISSION_R1.json"
        ),
        "root_admission_sha256": (
            "79adf2e9b11dc737459bb770124e4401586920a36c68a3384050fc1c99886a23"
        ),
        "artifact_dir": (
            "../bandit_continuation/kaggle_warm_cohort/"
            "smoke4_20261008/terminal_collect_r1"
        ),
        "terminal_receipt": (
            "../bandit_continuation/kaggle_warm_cohort/"
            "smoke4_20261008/KAGGLE_SMOKE4_TERMINAL_OPERATOR_RECEIPT_R1.json"
        ),
        "push_receipt": (
            "../bandit_continuation/kaggle_warm_cohort/"
            "smoke4_20261008/KAGGLE_SMOKE4_PROVIDER_PUSH_R1.json"
        ),
        "admission_bindings": {
            "kernel_script_sha256": (
                "5e75bdeade37964848e8e4f1023566e979ce29545edbd75ef64a59bb4204c538"
            ),
            "metadata_sha256": (
                "246c8415fd195827ef2432fabf6ac664858f27a6e3ee7281596abf42f043fae6"
            ),
            "packet_sha256": (
                "f7472dfd64f505de7e65a080691ae24917a4504c35e033f74084e88505e26179"
            ),
            "independent_review_sha256": (
                "62cf342c59b19af9c16bf5a00021ea7d6fb780dec384d32c8420eecb964d2469"
            ),
        },
    },
    "vladkochetov07/e463-tpu-real-data-reload-v7-20261008": {
        "resource": "tpu",
        "metadata_id": "vladkochetov07/e463-tpu-realdata-reload-v7-20261008",
        "root_admission": (
            "../bandit_continuation/controller_claude_20261008/"
            "KAGGLE_TPU_V7_ROOT_ADMISSION_R1.json"
        ),
        "root_admission_sha256": (
            "dbb32828d38d3b3f0f294a4f08b3d7fd189740e973493a212ba29a36dcdc4bd4"
        ),
        "artifact_dir": "tpu_v7_terminal_20261008_r1",
        "terminal_receipt": "E463_TPU_V7_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json",
        "push_receipt": "KAGGLE_TPU_V7_PROVIDER_PUSH_R1.json",
        "admission_bindings": {
            "code_sha256": (
                "d22947623bbe638280183ab019d4159ea37038b73840f3058e0137379bb4dad7"
            ),
            "metadata_sha256": (
                "23e67bce8659cbd98bcc9c985c73637e8e3b161e17f56f5cdb185e030714c4ef"
            ),
            "bundle_sha256": (
                "053638cf52a82f00b2da8fce82fcb63f21f7bfaccaf1b766b64e0ffd8ab1c528"
            ),
            "independent_review_sha256": (
                "26f4119a0a469d28588652bf5faca08a210ac472312af769606f64d41f999d2a"
            ),
        },
    },
}


def utc_now() -> str:
    """Return a stable UTC timestamp for a durable receipt."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large model outputs are never loaded at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace a JSON receipt without exposing a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def resolve_repo_path(repo_root: Path, config_dir: Path, relative_path: str) -> Path:
    """Resolve a config path while rejecting absolute paths and repo escapes."""
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"expected a repository-relative path: {relative_path}")
    resolved = (config_dir / candidate).resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        raise ValueError(f"path escapes repository root: {relative_path}")
    return resolved


def validate_config(
    config_path: Path,
    repo_root: Path,
    *,
    pinned_targets: dict[str, dict[str, Any]] = PINNED_TARGETS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the finite target list against root admissions and pinned hashes."""
    config_path = config_path.resolve()
    repo_root = repo_root.resolve()
    if not config_path.is_relative_to(repo_root):
        raise ValueError("collector config must live inside the repository")
    config = json.loads(config_path.read_text())
    if config.get("schema") != "wnn.kaggle-terminal-collector-targets.v1":
        raise ValueError("unsupported terminal collector config schema")
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("collector targets must be a non-empty list")
    kernels = [target.get("kernel") for target in targets]
    if len(kernels) != len(set(kernels)) or set(kernels) != set(pinned_targets):
        raise ValueError("collector must contain exactly the two pinned kernels")

    config_dir = config_path.parent
    for target in targets:
        kernel = target["kernel"]
        pinned = pinned_targets[kernel]
        for key, expected in pinned.items():
            if target.get(key) != expected:
                raise ValueError(f"{kernel}: {key} does not match the pinned value")

        admission_path = resolve_repo_path(
            repo_root, config_dir, target["root_admission"]
        )
        if sha256_file(admission_path) != target["root_admission_sha256"]:
            raise ValueError(f"{kernel}: root admission hash mismatch")
        admission = json.loads(admission_path.read_text())
        if admission.get("authorized") is not True:
            raise ValueError(f"{kernel}: root admission is not authorized")
        if admission.get("kernel_id") != target["metadata_id"]:
            raise ValueError(f"{kernel}: admission metadata ID mismatch")
        for key, expected in target["admission_bindings"].items():
            if admission.get(key) != expected:
                raise ValueError(f"{kernel}: admission binding mismatch for {key}")

        push_receipt_path = resolve_repo_path(
            repo_root, config_dir, target["push_receipt"]
        )
        push_receipt = json.loads(push_receipt_path.read_text())
        if push_receipt.get("canonical_provider_slug") != kernel:
            raise ValueError(f"{kernel}: provider push receipt slug mismatch")
        if push_receipt.get("root_admission_sha256") != target["root_admission_sha256"]:
            raise ValueError(f"{kernel}: provider push receipt admission mismatch")
        if push_receipt.get("push_count") != 1 or push_receipt.get("kernel_version") != 1:
            raise ValueError(f"{kernel}: expected exactly one accepted version")
        packet = push_receipt.get("packet", {})
        for key, expected in target["admission_bindings"].items():
            receipt_value = packet.get(key, push_receipt.get(key, admission.get(key)))
            if receipt_value != expected:
                raise ValueError(
                    f"{kernel}: provider receipt binding mismatch for {key}"
                )

        for path_field in ("artifact_dir", "terminal_receipt", "push_receipt"):
            resolve_repo_path(repo_root, config_dir, target[path_field])

    cli_path = Path(config.get("kaggle_cli", ""))
    if not cli_path.is_absolute() or not cli_path.is_file() or not os.access(cli_path, os.X_OK):
        raise ValueError("configured Kaggle CLI must be an executable absolute path")
    if cli_path.resolve() != (repo_root / ".venv/bin/kaggle").resolve():
        raise ValueError("collector must use the repository-pinned Kaggle CLI executable")
    if config.get("expected_cli_version") != "Kaggle CLI 2.2.4":
        raise ValueError("collector must pin the reviewed Kaggle CLI version")
    if int(config.get("minimum_free_bytes", 0)) < 2 * GIB:
        raise ValueError("minimum disk reserve must be at least 2 GiB")
    if int(config.get("maximum_download_bytes", 0)) < int(config["minimum_free_bytes"]):
        raise ValueError("maximum download size must be at least the disk reserve")
    if int(config["maximum_download_bytes"]) > 8 * GIB:
        raise ValueError("maximum_download_bytes may not exceed the reviewed 8 GiB cap")
    if not 1 <= int(config.get("max_attempts", 0)) <= MAX_ATTEMPTS:
        raise ValueError(f"max_attempts must be in [1, {MAX_ATTEMPTS}]")
    if not 60 <= int(config.get("max_runtime_seconds", 0)) <= 850:
        raise ValueError("max_runtime_seconds must be in [60, 850]")
    resolve_repo_path(repo_root, config_dir, config["poll_log"])
    return config, targets


def terminal_state(status: str) -> bool:
    """Return whether a Kaggle status string represents a terminal state."""
    return status.rsplit(".", maxsplit=1)[-1].upper() in TERMINAL_STATES


def file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Return a relative-path, byte-count, and SHA-256 inventory."""
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing to inventory a symlink: {path}")
        if path.is_file():
            inventory[path.relative_to(root).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return inventory


def directory_bytes(root: Path) -> int:
    """Count regular-file bytes beneath a directory without following symlinks."""
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            total += path.stat().st_size
    return total


def collection_is_complete(receipt_path: Path, artifact_dir: Path) -> bool:
    """Check that terminal outputs and logs listed by a prior receipt still exist."""
    try:
        receipt = json.loads(receipt_path.read_text())
        collection = receipt["terminal_collection"]
        files = collection["files"]
        root = artifact_dir.resolve()
        listed_files = [(artifact_dir / name, (root / name).resolve()) for name in files]
        return (
            terminal_state(receipt["terminal_status"])
            and collection["output_exit_code"] == 0
            and collection["log_exit_code"] == 0
            and bool(files)
            and all(
                not source.is_symlink()
                and resolved.is_relative_to(root)
                and source.is_file()
                for source, resolved in listed_files
            )
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def next_attempt_number(artifact_dir: Path) -> int:
    """Choose a new attempt number while preserving every earlier partial result."""
    numbers = []
    for path in artifact_dir.glob("attempt_*"):
        if path.is_dir() and path.name.removeprefix("attempt_").isdigit():
            numbers.append(int(path.name.removeprefix("attempt_")))
    return max(numbers, default=0) + 1


def run_bounded_command(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    monitor_dir: Path,
    min_free_bytes: int,
    max_bytes: int,
    deadline: float,
) -> dict[str, Any]:
    """Run one Kaggle read-only download, enforcing disk, byte, and wall bounds."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()
    reason: str | None = None
    process: subprocess.Popen[bytes] | None = None
    with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_stream,
                stderr=stderr_stream,
                shell=False,
            )
            while process.poll() is None:
                if shutil.disk_usage(monitor_dir).free < min_free_bytes:
                    reason = "minimum_free_disk_reserve_reached"
                elif directory_bytes(monitor_dir) > max_bytes:
                    reason = "download_byte_limit_reached"
                elif time.monotonic() >= deadline:
                    reason = "collector_runtime_limit_reached"
                if reason is not None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    break
                time.sleep(1)
            return_code = process.wait()
        except OSError as error:
            return {
                "exit_code": None,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "stop_reason": "process_start_error",
                "error_type": type(error).__name__,
            }
    final_size = directory_bytes(monitor_dir)
    final_free = shutil.disk_usage(monitor_dir).free
    if reason is None and final_size > max_bytes:
        reason = "download_byte_limit_reached_after_process_exit"
    if reason is None and final_free < min_free_bytes:
        reason = "minimum_free_disk_reserve_reached_after_process_exit"
    if reason is not None and return_code == 0:
        return_code = 125
    return {
        "exit_code": return_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stop_reason": reason,
        "bytes_in_monitor_dir": final_size,
        "free_bytes_after": final_free,
    }


def kaggle_cli_version(cli_path: str) -> dict[str, Any]:
    """Read the exact Kaggle CLI version without contacting the provider."""
    try:
        result = subprocess.run(
            [cli_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"exit_code": None, "version": "", "error_type": type(error).__name__}
    output = (result.stdout or result.stderr).strip()
    return {"exit_code": result.returncode, "version": output}


def _target_status(snapshot: dict[str, Any], kernel: str) -> dict[str, Any] | None:
    return next(
        (row for row in snapshot.get("status", []) if row.get("kernel") == kernel),
        None,
    )


def _terminal_receipt(
    *,
    target: dict[str, Any],
    status_row: dict[str, Any],
    attempt_dir: Path,
    artifact_dir: Path,
    result: dict[str, Any],
    cli_info: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create an auditable receipt, including partial attempts and all file hashes."""
    files = file_inventory(artifact_dir)
    output_count = sum(
        1
        for name in files
        if name.startswith(f"{attempt_dir.name}/outputs/")
    )
    output_code = result.get("output", {}).get("exit_code")
    log_code = result.get("logs", {}).get("exit_code")
    complete = output_code == 0 and log_code == 0
    return {
        "schema": "wnn.kaggle-terminal-collector-receipt.v1",
        "recorded_utc": utc_now(),
        "kernel": target["kernel"],
        "resource": target["resource"],
        "root_admission_sha256": target["root_admission_sha256"],
        "push_receipt_sha256": sha256_file(
            resolve_repo_path(
                Path(config["repo_root"]),
                Path(config["config_dir"]),
                target["push_receipt"],
            )
        ),
        "terminal_status": status_row["status"],
        "provider_failure_message": status_row.get("failure_message", ""),
        "terminal_collection": {
            "collection_complete": complete,
            "output_exit_code": output_code,
            "log_exit_code": log_code,
            "attempt_directory": attempt_dir.relative_to(artifact_dir).as_posix(),
            "provider_output_file_count": output_count,
            "empty_provider_outputs_note": (
                "A successful output command returned no provider artifacts; this alone "
                "does not show whether the kernel script executed."
                if complete and output_count == 0
                else None
            ),
            "minimum_free_bytes": int(config["minimum_free_bytes"]),
            "maximum_download_bytes": int(config["maximum_download_bytes"]),
            "cli_executable": config["kaggle_cli"],
            "cli_version": cli_info,
            "files": files,
        },
        "attempt_result": result,
    }


def collect_terminal_target(
    *,
    target: dict[str, Any],
    status_row: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
    config_dir: Path,
) -> dict[str, Any]:
    """Download one terminal kernel's logs and outputs into a new private attempt."""
    artifact_dir = resolve_repo_path(repo_root, config_dir, target["artifact_dir"])
    receipt_path = resolve_repo_path(repo_root, config_dir, target["terminal_receipt"])
    artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(artifact_dir, 0o700)
    if collection_is_complete(receipt_path, artifact_dir):
        return {"kernel": target["kernel"], "state": "already_collected"}

    existing_attempts = next_attempt_number(artifact_dir) - 1
    if existing_attempts >= int(config["max_attempts"]):
        return {
            "kernel": target["kernel"],
            "state": "retry_limit_reached",
            "attempts": existing_attempts,
        }

    attempt_dir = artifact_dir / f"attempt_{existing_attempts + 1:03d}"
    attempt_dir.mkdir(mode=0o700)
    os.chmod(attempt_dir, 0o700)
    snapshot_status_path = attempt_dir / "provider_status.json"
    atomic_write_json(
        snapshot_status_path,
        {
            "observed_utc": utc_now(),
            "kernel": target["kernel"],
            "status": status_row.get("status"),
            "failure_message": status_row.get("failure_message", ""),
        },
    )
    result: dict[str, Any] = {"logs": None, "output": None}
    cli_info = kaggle_cli_version(config["kaggle_cli"])
    cli_matches = (
        cli_info.get("exit_code") == 0
        and cli_info.get("version") == config["expected_cli_version"]
    )
    result["cli_version_match"] = cli_matches
    if not cli_matches:
        result["stop_reason"] = "pinned_kaggle_cli_version_unavailable"
    elif shutil.disk_usage(artifact_dir).free < int(config["minimum_free_bytes"]):
        result["stop_reason"] = "minimum_free_disk_reserve_not_met_before_download"
    else:
        deadline = time.monotonic() + int(config["max_runtime_seconds"])
        cli_path = config["kaggle_cli"]
        slug = target["kernel"]
        result["logs"] = run_bounded_command(
            [cli_path, "kernels", "logs", slug],
            stdout_path=attempt_dir / "provider_logs.txt",
            stderr_path=attempt_dir / "provider_logs.stderr.txt",
            monitor_dir=attempt_dir,
            min_free_bytes=int(config["minimum_free_bytes"]),
            max_bytes=int(config["maximum_download_bytes"]),
            deadline=deadline,
        )
        if result["logs"]["exit_code"] == 0 and time.monotonic() < deadline:
            if shutil.disk_usage(artifact_dir).free < int(config["minimum_free_bytes"]):
                result["output"] = {
                    "exit_code": None,
                    "stop_reason": "minimum_free_disk_reserve_not_met_before_output_download",
                }
            else:
                output_dir = attempt_dir / "outputs"
                output_dir.mkdir(mode=0o700)
                result["output"] = run_bounded_command(
                    [cli_path, "kernels", "output", "-p", str(output_dir), "-o", "-q", slug],
                    stdout_path=attempt_dir / "output_command.stdout.txt",
                    stderr_path=attempt_dir / "output_command.stderr.txt",
                    monitor_dir=output_dir,
                    min_free_bytes=int(config["minimum_free_bytes"]),
                    max_bytes=int(config["maximum_download_bytes"]),
                    deadline=deadline,
                )
        elif result["logs"]["exit_code"] != 0:
            result["output"] = {
                "exit_code": None,
                "stop_reason": "logs_download_failed; outputs preserved for retry",
            }
        else:
            result["output"] = {
                "exit_code": None,
                "stop_reason": "collector_runtime_limit_reached_before_output_download",
            }

    atomic_write_json(attempt_dir / "attempt_result.json", result)
    for path in artifact_dir.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            os.chmod(path, 0o700)
        elif path.is_file() and not path.is_symlink():
            os.chmod(path, 0o600)
    receipt = _terminal_receipt(
        target=target,
        status_row=status_row,
        attempt_dir=attempt_dir,
        artifact_dir=artifact_dir,
        result=result,
        cli_info=cli_info,
        config=config,
    )
    atomic_write_json(receipt_path, receipt)
    return {
        "kernel": target["kernel"],
        "state": (
            "terminal_outputs_and_logs_collected"
            if receipt["terminal_collection"]["collection_complete"]
            else "terminal_collection_partial"
        ),
        "attempt": attempt_dir.name,
        "provider_output_file_count": receipt["terminal_collection"][
            "provider_output_file_count"
        ],
        "receipt": str(receipt_path),
        "file_count": len(receipt["terminal_collection"]["files"]),
    }


def run_once(config_path: Path, repo_root: Path) -> dict[str, Any]:
    """Poll exactly two admitted kernels, then collect only terminal results."""
    config, targets = validate_config(config_path, repo_root)
    config["repo_root"] = str(repo_root.resolve())
    config["config_dir"] = str(config_path.resolve().parent)
    log_path = resolve_repo_path(
        repo_root.resolve(), config_path.resolve().parent, config["poll_log"]
    )
    try:
        snapshot = collect_provider_snapshot(config_path)
    except Exception as error:
        failure = {
            "observed_at_utc": utc_now(),
            "source": "terminal collector provider status/quota query failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        append_jsonl(log_path, failure)
        append_jsonl(
            log_path,
            {
                "collector_result": {
                    "observed_utc": failure["observed_at_utc"],
                    "state": "provider_poll_error",
                    "error_type": failure["error_type"],
                }
            },
        )
        return {"state": "provider_poll_error", "error_type": failure["error_type"]}
    append_jsonl(log_path, snapshot)
    outcomes = []
    for target in targets:
        row = _target_status(snapshot, target["kernel"])
        if row is None:
            outcomes.append({"kernel": target["kernel"], "state": "status_missing"})
        elif row.get("monitoring_state") == "stopped_after_terminal_evidence_downloaded":
            outcomes.append({"kernel": target["kernel"], "state": "already_collected"})
        elif not terminal_state(str(row.get("status", ""))):
            outcomes.append(
                {"kernel": target["kernel"], "state": "provider_not_terminal", "status": row.get("status")}
            )
        else:
            outcomes.append(
                collect_terminal_target(
                    target=target,
                    status_row=row,
                    config=config,
                    repo_root=repo_root.resolve(),
                    config_dir=config_path.resolve().parent,
                )
            )
    summary = {
        "observed_utc": snapshot.get("observed_at_utc"),
        "quota": snapshot.get("quota"),
        "targets": outcomes,
    }
    append_jsonl(log_path, {"collector_result": summary})
    return summary


def parse_args() -> argparse.Namespace:
    """Parse one-shot collector inputs; scheduling belongs to systemd."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    """Execute one finite poll/download pass and exit without waiting for the timer."""
    arguments = parse_args()
    try:
        result = run_once(arguments.config, arguments.repo_root)
    except Exception as error:  # keep a bounded durable failure instead of retrying a push
        print(
            json.dumps(
                {
                    "state": "collector_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 1 if result.get("state") == "provider_poll_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
