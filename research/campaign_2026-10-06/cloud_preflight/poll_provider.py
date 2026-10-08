"""Capture exact Kaggle quota and pending kernel statuses without resubmitting."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetAcceleratorQuotaStatisticsRequest,
    ApiGetKernelSessionStatusRequest,
)

GPU_BASELINE_SECONDS = 37047.321
TPU_BASELINE_SECONDS = 134.623
# Retain the original baseline: 94.516s landed before the user resumed, plus
# 56,686s newly allocated in RESUMED_ALLOCATION_20261007.md, with 20% headroom.
GPU_BUDGET_SECONDS = 56780.516
# ADR-0010 admits independent TPU recovery with the original usage baseline kept.
TPU_BUDGET_SECONDS = 7200.0
TERMINAL_STATES = {"COMPLETE", "COMPLETED", "ERROR", "FAILED", "CANCELED", "CANCELLED"}
NO_PROVIDER_JOB_SCHEMA = "wnn-kaggle-no-provider-job-created-reconciliation-v1"
MAX_NO_JOB_RECEIPT_BYTES = 128 * 1024


def _seconds(value: Any) -> float | None:
    """Convert a duration-like object into seconds without rounding."""
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


def _status(client: Any, kernel: str) -> dict[str, Any]:
    """Read status for one exact kernel using a short HTTP timeout."""
    api = KaggleApi()
    owner, slug, version = api.parse_kernel_string(kernel)
    request = ApiGetKernelSessionStatusRequest()
    request.user_name = owner
    request.kernel_slug = slug
    if version is not None:
        request.version_label = version
    response = client.kernels.kernels_api_client.get_kernel_session_status(request)
    return {
        "kernel": kernel,
        "status": str(response.status),
        "failure_message": response.failure_message or "",
    }


def _terminal_evidence(target: dict[str, str], config_dir: Path) -> dict[str, Any] | None:
    """Return a completed receipt only after all named output files exist."""
    receipt_name = target.get("terminal_receipt")
    if not receipt_name:
        return None
    receipt_path = config_dir / receipt_name
    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    collection = receipt.get("terminal_collection", {})
    files = collection.get("files", {})
    artifact_dir = config_dir / target.get("artifact_dir", ".")
    outputs_downloaded = (
        collection.get("output_exit_code") == 0
        and collection.get("log_exit_code") == 0
        and bool(files)
        and all((artifact_dir / name).is_file() for name in files)
    )
    terminal_status = receipt.get("terminal_status")
    if not outputs_downloaded or not terminal_status:
        return None
    return {
        "receipt": receipt_name,
        "terminal_status": terminal_status,
        "outputs_downloaded": True,
    }


def _no_provider_job_evidence(
    target: dict[str, Any], config_dir: Path
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate an explicit receipt that a rejected push created no provider job.

    This is distinct from a terminal run: it has no provider status, outputs, or
    downloaded artifact set. Invalid evidence falls back to ordinary polling.
    """
    receipt_name = target.get("no_provider_job_receipt")
    if receipt_name is None:
        return None, None
    if (
        not isinstance(receipt_name, str)
        or "\\" in receipt_name
        or Path(receipt_name).name != receipt_name
    ):
        return None, "invalid_no_job_receipt_path"
    expected_sha256 = target.get("no_provider_job_receipt_sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        return None, "invalid_no_job_receipt_sha256"
    root = config_dir.resolve()
    receipt_path = config_dir / receipt_name
    if receipt_path.is_symlink():
        return None, "symlinked_no_job_receipt"
    try:
        resolved = receipt_path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            return None, "no_job_receipt_outside_config"
        if resolved.stat().st_size > 128 * 1024:
            return None, "no_job_receipt_too_large"
        raw = resolved.read_bytes()
    except OSError:
        return None, "no_job_receipt_unavailable"
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        return None, "no_job_receipt_sha256_mismatch"
    try:
        receipt = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "no_job_receipt_json_invalid"
    if not isinstance(receipt, dict):
        return None, "no_job_receipt_not_object"

    kernel = target.get("kernel")
    evidence = receipt.get("evidence")
    required_evidence = {"push_attempt", "read_only_absence", "hold_transfer"}
    if (
        receipt.get("schema") != "wnn-kaggle-no-provider-job-created-reconciliation-v1"
        or receipt.get("provider") != "Kaggle"
        or receipt.get("resolution") != "confirmed_not_created"
        or receipt.get("kernel_id") != kernel
        or receipt.get("provider_job_created") is not False
        or receipt.get("provider_version") is not None
        or receipt.get("terminal_job_status") is not None
        or receipt.get("outputs_downloaded") is not False
        or receipt.get("fit_authorized") is not False
        or type(receipt.get("provisional_reservation_seconds")) is not int
        or receipt.get("provisional_reservation_seconds") != 0
        or not isinstance(evidence, dict)
        or set(evidence) != required_evidence
        or receipt.get("root_one_push_authorization_sha256")
        != target.get("root_one_push_authorization_sha256")
        or receipt.get("hold_transferred_to_kernel") != target.get("hold_transferred_to_kernel")
        or target.get("fit_authorized") is not False
        or type(target.get("provisional_reservation_seconds")) is not int
        or target.get("provisional_reservation_seconds") != 0
        or target.get("status") != "push_rejected_http_400_readonly_not_found_hold_transferred"
    ):
        return None, "no_job_receipt_fields_mismatch"

    for binding in evidence.values():
        if not isinstance(binding, dict):
            return None, "invalid_no_job_evidence_binding"
        relative_name = binding.get("path")
        digest = binding.get("sha256")
        if (
            not isinstance(relative_name, str)
            or "\\" in relative_name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return None, "invalid_no_job_evidence_binding"
        relative = PurePosixPath(relative_name)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_name
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            return None, "invalid_no_job_evidence_path"

    absence = evidence["read_only_absence"]
    transfer = evidence["hold_transfer"]
    if (
        receipt.get("read_only_absence_observed_at_utc") is None
        or receipt.get("confirmed_at_utc") is None
        or not isinstance(absence.get("path"), str)
        or not isinstance(transfer.get("path"), str)
    ):
        return None, "no_job_receipt_fields_mismatch"
    try:
        campaign_root = root.parent
        configured_transfer = (config_dir / str(target.get("hold_transfer_receipt"))).resolve(
            strict=True
        )
        bound_transfer = campaign_root.joinpath(*PurePosixPath(transfer["path"]).parts).resolve(
            strict=True
        )
    except OSError:
        return None, "hold_transfer_path_unavailable"
    if configured_transfer != bound_transfer:
        return None, "hold_transfer_path_mismatch"

    return (
        {
            "receipt": receipt_name,
            "receipt_sha256": expected_sha256,
            "provider_job_created": False,
            "terminal_job_status": None,
            "outputs_downloaded": False,
            "hold_transferred_to_kernel": receipt["hold_transferred_to_kernel"],
            "provisional_reservation_seconds": 0,
        },
        None,
    )


def _bounded_client(api: KaggleApi) -> Any:
    """Build the authenticated SDK client with bounded HTTP connect/read timeouts."""
    client = api.build_kaggle_client()
    client.__enter__()
    http = client._http_client
    http._init_session()
    original_send = http._session.send

    def send_with_timeout(prepared_request: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", (5, 15))
        return original_send(prepared_request, **kwargs)

    http._session.send = send_with_timeout
    return client


def collect(targets_path: Path) -> dict[str, Any]:
    """Fetch one quota snapshot and statuses for targets without receipts."""
    targets_path = targets_path.resolve()
    config = json.loads(targets_path.read_text())
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("provider targets file must contain a non-empty targets list")

    api = KaggleApi()
    api.authenticate()
    client = _bounded_client(api)
    try:
        request = ApiGetAcceleratorQuotaStatisticsRequest()
        quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics(request)
        gpu_used = _seconds(quota.gpu_quota.time_used)
        tpu_used = _seconds(quota.tpu_quota.time_used)
        gpu_reserved = _seconds(quota.gpu_quota.time_reserved)
        tpu_reserved = _seconds(quota.tpu_quota.time_reserved)

        status_rows = []
        for target in targets:
            kernel = target.get("kernel")
            resource = target.get("resource")
            if not isinstance(kernel, str) or resource not in {"gpu", "tpu"}:
                raise ValueError(f"invalid provider target entry: {target!r}")
            no_job, no_job_error = _no_provider_job_evidence(target, targets_path.parent)
            if no_job is not None:
                status_rows.append(
                    {
                        "kernel": kernel,
                        "resource": resource,
                        "status": "NO_PROVIDER_JOB_CREATED",
                        "monitoring_state": "stopped_confirmed_no_provider_job",
                        "provider_job_created": False,
                        "no_provider_job_evidence": no_job,
                    }
                )
                continue
            evidence = _terminal_evidence(target, targets_path.parent)
            if evidence is not None:
                status_rows.append(
                    {
                        "kernel": kernel,
                        "resource": resource,
                        "status": evidence["terminal_status"],
                        "monitoring_state": "stopped_after_terminal_evidence_downloaded",
                        "terminal_evidence": evidence,
                    }
                )
                continue
            try:
                status_row = {
                    **_status(client, kernel),
                    "resource": resource,
                    "monitoring_state": "polling_until_terminal_evidence_downloaded",
                    "terminal_evidence_receipt": target.get("terminal_receipt"),
                }
                if no_job_error is not None:
                    status_row["no_provider_job_receipt_validation"] = "invalid"
                    status_row["no_provider_job_receipt_reason"] = no_job_error
                status_rows.append(status_row)
            except Exception as error:
                status_row = {
                    "kernel": kernel,
                    "resource": resource,
                    "monitoring_state": "poll_error_retry_next_timer_tick",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                if no_job_error is not None:
                    status_row["no_provider_job_receipt_validation"] = "invalid"
                    status_row["no_provider_job_receipt_reason"] = no_job_error
                status_rows.append(status_row)

        gpu_delta = None if gpu_used is None else gpu_used - GPU_BASELINE_SECONDS
        tpu_delta = None if tpu_used is None else tpu_used - TPU_BASELINE_SECONDS
        return {
            "observed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "cli_package_version": importlib.metadata.version("kaggle"),
            "source": "read-only Kaggle SDK quota/status calls; no resubmit or cancel",
            "quota_refresh_utc": (
                quota.quota_refresh_time.isoformat() if quota.quota_refresh_time else None
            ),
            "quota": {
                "gpu": {
                    "used_seconds": gpu_used,
                    "reserved_seconds": gpu_reserved,
                    "total_allowed_seconds": _seconds(quota.gpu_quota.total_time_allowed),
                    "campaign_baseline_used_seconds": GPU_BASELINE_SECONDS,
                    "campaign_used_seconds": gpu_delta,
                    "campaign_cap_seconds": GPU_BUDGET_SECONDS,
                    "campaign_headroom_seconds": (
                        None if gpu_delta is None else GPU_BUDGET_SECONDS - gpu_delta
                    ),
                },
                "tpu": {
                    "used_seconds": tpu_used,
                    "reserved_seconds": tpu_reserved,
                    "total_allowed_seconds": _seconds(quota.tpu_quota.total_time_allowed),
                    "campaign_baseline_used_seconds": TPU_BASELINE_SECONDS,
                    "campaign_used_seconds": tpu_delta,
                    "campaign_cap_seconds": TPU_BUDGET_SECONDS,
                    "campaign_headroom_seconds": (
                        None if tpu_delta is None else TPU_BUDGET_SECONDS - tpu_delta
                    ),
                },
            },
            "status": status_rows,
            "session_id": {
                "value": None,
                "reason": "Kaggle status JSON currently omits kernel_session_id; do not substitute versionNumber.",
            },
        }
    finally:
        client.__exit__(None, None, None)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append and fsync a complete JSON line so every poll is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with os.fdopen(descriptor, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def parse_args() -> argparse.Namespace:
    """Parse the bounded poll targets and durable log path."""
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=base / "provider_targets.json")
    parser.add_argument("--log", type=Path, default=base / "provider_poll_log.jsonl")
    return parser.parse_args()


def main() -> int:
    """Write one snapshot; capture failures durably and never resubmit work."""
    arguments = parse_args()
    try:
        row = collect(arguments.targets)
    except Exception as error:
        row = {
            "observed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": "read-only Kaggle API poll; snapshot incomplete",
            "targets_file": str(arguments.targets),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        append_jsonl(arguments.log, row)
        print(json.dumps(row, sort_keys=True), file=sys.stderr)
        return 1

    append_jsonl(arguments.log, row)
    print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
