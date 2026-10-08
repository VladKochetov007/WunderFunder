"""Collect only the CPU source-stage's small, hash-bound terminal artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GIB = 1024**3
KERNEL = "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1"
TERMINAL_STATES = {"COMPLETE", "COMPLETED", "ERROR", "FAILED", "CANCELED", "CANCELLED"}
MAX_OUTPUT_BYTES = 64 * 1024**2
MIN_FREE_BYTES = 4 * GIB
COMMAND_TIMEOUT_SECONDS = 300
LISTING_TIMEOUT_SECONDS = 60
FILE_LISTING_PAGE_SIZE = 200
FILE_PATTERN = (
    r"(^|/)(boot_marker\.json|heartbeat\.json|final_status\.json|phase\.jsonl|"
    r"source_stage_receipt\.json|failure\.json|manifest\.json|"
    r"source_verification\.json|reader_validation\.json|reader_profile\.json|"
    r"cache_build\.log|cache_profile\.log|"
    r"wnn-parsed-cache-source-stage-20261009-v1\.log)$"
)
ALLOWED_OUTPUT_NAME = re.compile(FILE_PATTERN)
BASE = Path(__file__).resolve().parent
POLL_LOG = BASE / "provider_poll_log_active_20261009_r1.jsonl"
OUTPUT_ROOT = BASE / "cpu_source_stage_terminal_outputs_r1"
RECEIPT_PATH = BASE / "KAGGLE_CPU_CACHE_TERMINAL_OPERATOR_RECEIPT_R1.json"
KAGGLE = Path("/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/kaggle")


def utc_now() -> str:
    """Return an ISO timestamp with explicit UTC offset."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without reading it into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    """Persist a JSON receipt atomically and sync its containing directory."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def is_terminal_status(status: object) -> bool:
    """Recognize terminal Kaggle statuses without relying on full enum spelling."""
    if not isinstance(status, str):
        return False
    return status.rsplit(".", maxsplit=1)[-1].upper() in TERMINAL_STATES


def latest_terminal_observation(path: Path) -> dict[str, object] | None:
    """Return the latest successful terminal observation for this exact kernel."""
    if not path.is_file():
        return None
    latest: dict[str, object] | None = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row: dict[str, Any] = json.loads(line)  # JSON values are dynamically shaped.
            except json.JSONDecodeError:
                continue
            for item in row.get("status", []):
                if not isinstance(item, dict):
                    continue
                if (
                    item.get("kernel") == KERNEL
                    and item.get("monitoring_state") != "poll_error_retry_next_timer_tick"
                    and is_terminal_status(item.get("status"))
                ):
                    latest = {
                        "observed_at_utc": row.get("observed_at_utc"),
                        "status": item["status"],
                        "failure_message": item.get("failure_message", ""),
                    }
    return latest


def file_inventory(root: Path) -> tuple[dict[str, dict[str, object]], int]:
    """Hash allowed provider files and reject unexpected or symlinked outputs."""
    inventory: dict[str, dict[str, object]] = {}
    total_bytes = 0
    if not root.is_dir():
        return inventory, total_bytes
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing a symlinked provider output: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if (
            not ALLOWED_OUTPUT_NAME.search(relative)
            and path.name != f"{KERNEL.rsplit('/', 1)[1]}.log"
        ):
            raise ValueError(f"unexpected output escaped the small-file filter: {relative}")
        size_bytes = path.stat().st_size
        total_bytes += size_bytes
        inventory[relative] = {"size_bytes": size_bytes, "sha256": sha256_file(path)}
    return inventory, total_bytes


def receipt_files_match(files: object, root: Path) -> bool:
    """Verify a previously completed receipt before avoiding another download."""
    if not isinstance(files, dict) or not files:
        return False
    root = root.resolve()
    for relative, metadata in files.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            return False
        if not isinstance(metadata, dict):
            return False
        size_bytes = metadata.get("size_bytes")
        expected_sha256 = metadata.get("sha256")
        if (
            type(size_bytes) is not int
            or size_bytes < 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            return False
        path = root / relative
        try:
            if (
                path.is_symlink()
                or not path.resolve(strict=True).is_relative_to(root)
                or not path.is_file()
                or path.stat().st_size != size_bytes
                or sha256_file(path) != expected_sha256
            ):
                return False
        except OSError:
            return False
    return True


def already_collected() -> bool:
    """Return true only when the durable receipt and its recorded bytes match."""
    if not RECEIPT_PATH.is_file():
        return False
    try:
        receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(receipt, dict):
        return False
    collection = receipt.get("terminal_collection", {})
    return (
        receipt.get("terminal_status") is not None
        and collection.get("collection_complete") is True
        and collection.get("output_exit_code") == 0
        and collection.get("log_exit_code") == 0
        and receipt_files_match(collection.get("files"), OUTPUT_ROOT)
    )


def parse_selected_file_listing(csv_text: str) -> list[str]:
    """Select allowed basenames from Kaggle's metadata-only file listing."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("Kaggle file listing has no CSV header")
    fields = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in reader.fieldnames}
    name_field = next((fields[key] for key in ("name", "filename", "file") if key in fields), None)
    if name_field is None:
        raise ValueError(f"Kaggle file listing lacks a filename field: {reader.fieldnames!r}")
    selected: list[str] = []
    selected_names: set[str] = set()
    listing_rows = 0
    for row in reader:
        listing_rows += 1
        if listing_rows >= FILE_LISTING_PAGE_SIZE:
            raise ValueError(
                "Kaggle file listing reached its page-size limit; refusing a possibly truncated inventory"
            )
        name = str(row.get(name_field, ""))
        path_parts = Path(name).parts
        if Path(name).is_absolute() or ".." in path_parts or "\\" in name:
            raise ValueError(f"Kaggle file listing contains an unsafe path: {name!r}")
        if not name or not ALLOWED_OUTPUT_NAME.search(name):
            continue
        basename = Path(name).name
        if basename in selected_names:
            raise ValueError(
                f"Kaggle file listing contains a duplicate output basename: {basename!r}"
            )
        selected_names.add(basename)
        selected.append(basename)
    return selected


def preflight_selected_output_files() -> list[str]:
    """Read provider filenames before any output bytes are downloaded."""
    result = subprocess.run(
        [
            str(KAGGLE),
            "kernels",
            "files",
            KERNEL,
            "--csv",
            "--page-size",
            str(FILE_LISTING_PAGE_SIZE),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=LISTING_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle output inventory failed before download: {result.stderr[-2000:]}"
        )
    return parse_selected_file_listing(result.stdout)


def tree_bytes(root: Path) -> int:
    """Measure the attempt directory during download without reading file contents."""
    total = 0
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"refusing a symlinked collector path: {path}")
            if path.is_file():
                total += path.stat().st_size
    return total


def output_names_match(files: dict[str, object], selected_names: list[str]) -> bool:
    """Compare provider listing names with local output paths after CLI nesting."""
    observed_names = [Path(name).name for name in files]
    return len(observed_names) == len(set(observed_names)) and sorted(observed_names) == sorted(
        selected_names
    )


def run_capped_download(command: list[str], provider_dir: Path, attempt: Path) -> tuple[int, str]:
    """Run the output CLI while enforcing byte and free-space limits continuously."""
    cli_log = attempt / "collector_cli.log"
    with cli_log.open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + COMMAND_TIMEOUT_SECONDS
        failure = ""
        try:
            while process.poll() is None:
                if tree_bytes(attempt) > MAX_OUTPUT_BYTES:
                    failure = "selected output exceeded its live byte cap"
                elif shutil.disk_usage(provider_dir).free < MIN_FREE_BYTES:
                    failure = "free disk fell below the collector reserve during download"
                elif time.monotonic() >= deadline:
                    failure = "terminal output command exceeded its wall timeout"
                if failure:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    break
                time.sleep(0.1)
            if process.poll() is None:
                process.wait(timeout=5)
        finally:
            log.flush()
            os.fsync(log.fileno())
    tail = cli_log.read_text(encoding="utf-8", errors="replace")[-4000:]
    if failure:
        return 75, f"{failure}\n{tail}"[-4000:]
    return int(process.returncode), tail


def collect() -> int:
    """Collect one terminal attempt's selected files without fetching cache arrays."""
    if already_collected():
        logging.info("terminal files already collected and hash-verified")
        return 0
    observation = latest_terminal_observation(POLL_LOG)
    if observation is None:
        logging.info("waiting for a terminal status in the active provider poll log")
        return 0
    selected_files = preflight_selected_output_files()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    required_free_bytes = MIN_FREE_BYTES + MAX_OUTPUT_BYTES
    if shutil.disk_usage(OUTPUT_ROOT).free < required_free_bytes:
        raise RuntimeError(
            "refusing terminal collection without the 4 GiB reserve plus the full output cap"
        )
    attempt = OUTPUT_ROOT / "attempt_001"
    if attempt.exists():
        raise RuntimeError("attempt_001 already exists; preserve it and review before retry")
    provider_dir = attempt / "provider"
    provider_dir.mkdir(parents=True, mode=0o700)
    atomic_write_json(
        attempt / "collector_attempt.json",
        {
            "schema": "wnn.kaggle-cpu-cache-small-collector-attempt.v1",
            "started_at_utc": utc_now(),
            "kernel": KERNEL,
            "terminal_observation": observation,
            "file_pattern": FILE_PATTERN,
            "preflight_selected_files": selected_files,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            "minimum_free_bytes": MIN_FREE_BYTES,
        },
    )
    command = [
        str(KAGGLE),
        "kernels",
        "output",
        KERNEL,
        "--path",
        str(provider_dir),
        "--quiet",
        "--page-size",
        "200",
        "--file-pattern",
        FILE_PATTERN,
    ]
    process_exit_code, stderr_tail = run_capped_download(command, provider_dir, attempt)
    try:
        files, downloaded_bytes = file_inventory(provider_dir)
        names_match = output_names_match(files, selected_files)
        output_valid = downloaded_bytes <= MAX_OUTPUT_BYTES and names_match
        if not names_match:
            stderr_tail = (
                f"{stderr_tail}\nprovider output file inventory differed from metadata preflight"
            )[-4000:]
    except (OSError, ValueError) as error:
        files, downloaded_bytes, output_valid = {}, 0, False
        stderr_tail = f"{stderr_tail}\n{type(error).__name__}: {error}"[-4000:]
    collection_complete = process_exit_code == 0 and output_valid and bool(files)
    recorded_exit_code = process_exit_code if output_valid else 75
    receipt = {
        "schema": "wnn.kaggle-cpu-cache-terminal-operator-receipt.v1",
        "observed_at_utc": utc_now(),
        "terminal_status": observation["status"],
        "status_observed_at_utc": observation.get("observed_at_utc"),
        "status_failure_message": observation.get("failure_message", ""),
        "kernel": KERNEL,
        "kernel_version": 1,
        "terminal_collection": {
            "collection_complete": collection_complete,
            "provider_file_pattern": FILE_PATTERN,
            "cache_payload_downloaded": False,
            "cache_payload_bytes": 0,
            "maximum_download_bytes": MAX_OUTPUT_BYTES,
            "preflight_selected_files": selected_files,
            "downloaded_bytes": downloaded_bytes,
            "provider_output_exit_code": process_exit_code,
            "output_exit_code": recorded_exit_code,
            "log_exit_code": recorded_exit_code,
            "stderr_tail": stderr_tail,
            "files": {
                str(Path("attempt_001") / name): metadata for name, metadata in files.items()
            },
        },
    }
    atomic_write_json(RECEIPT_PATH, receipt)
    attempt_metadata = json.loads((attempt / "collector_attempt.json").read_text())
    attempt_metadata.update(
        {
            "finished_at_utc": receipt["observed_at_utc"],
            "provider_exit_code": process_exit_code,
            "downloaded_bytes": downloaded_bytes,
            "collector_cli_log": {
                "size_bytes": (attempt / "collector_cli.log").stat().st_size,
                "sha256": sha256_file(attempt / "collector_cli.log"),
            },
            "collection_complete": collection_complete,
        }
    )
    atomic_write_json(attempt / "collector_attempt.json", attempt_metadata)
    logging.info(
        "terminal collection state=%s files=%d bytes=%d",
        "complete" if collection_complete else "partial",
        len(files),
        downloaded_bytes,
    )
    return 0 if collection_complete else 1


def main() -> int:
    """Run the one-shot small-file collector under its caller's shared lock."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return collect()
    except (OSError, RuntimeError, ValueError) as error:
        logging.error("collector_error=%s:%s", type(error).__name__, error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
