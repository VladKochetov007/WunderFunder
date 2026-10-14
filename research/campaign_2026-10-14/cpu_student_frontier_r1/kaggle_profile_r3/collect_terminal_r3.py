from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import requests
from kaggle.api.kaggle_api_extended import (
    ApiListKernelSessionOutputRequest,
    KaggleApi,
)

KERNEL_ID = "vladkochetov07/wnn-qat-f0-s4-profile-r3-runner"
OWNER = "vladkochetov07"
KERNEL_SLUG = "wnn-qat-f0-s4-profile-r3-runner"
MAX_COLLECTION_BYTES = 100_000_000
PAGE_SIZE = 200
CHUNK_BYTES = 1024 * 1024
LOG_NAME = f"{KERNEL_SLUG}.log"


def utc_now() -> str:
    """Return a UTC timestamp for the operator receipt."""
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """Hash a collected file using bounded streaming reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> int:
    """Write deterministic JSON without replacing an earlier receipt."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {path}")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write(encoded)
    os.replace(temporary, path)
    return len(encoded)


def seconds_field(value: object) -> float | None:
    """Convert a typed duration/number without using lossy string formatting."""
    if isinstance(value, (int, float)):
        return float(value)
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        return float(total_seconds())
    return None


def quota_lane(quota: object) -> dict[str, float | None]:
    """Extract typed duration fields from one Kaggle accelerator quota."""
    source_fields = {
        "total_time_allowed": "total_time_allowed",
        "used": "time_used",
        "reserved": "time_reserved",
    }
    return {
        name: seconds_field(getattr(quota, source_name, None))
        for name, source_name in source_fields.items()
    }


def quota_snapshot(api: KaggleApi) -> dict[str, Any]:
    """Return GPU/TPU quotas in seconds from Kaggle's typed quota response."""
    response = api.quota_view()
    refresh_time = getattr(response, "quota_refresh_time", None)
    return {
        "observed_utc": utc_now(),
        "quota_refresh_time": (
            refresh_time.isoformat() if hasattr(refresh_time, "isoformat") else None
        ),
        "gpu": quota_lane(response.gpu_quota),
        "tpu": quota_lane(response.tpu_quota),
    }


def terminal_status(api: KaggleApi) -> tuple[str, str]:
    """Require the exact profile kernel to be terminal before collection."""
    status_response = api.kernels_status(KERNEL_ID)
    status = str(status_response.status)
    state = status.rsplit(".", maxsplit=1)[-1].upper()
    if state not in {"ERROR", "COMPLETE", "CANCELLED"}:
        raise RuntimeError(f"kernel is not terminal: {state}")
    failure_message = str(getattr(status_response, "failure_message", "") or "")
    return status, failure_message


def output_listing(api: KaggleApi) -> tuple[list[object], str]:
    """List all session outputs without downloading them or retaining signed URLs."""
    files: list[object] = []
    page_token: str | None = None
    session_log = ""
    pages = 0
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = OWNER
            request.kernel_slug = KERNEL_SLUG
            api._set_paging(request, PAGE_SIZE, page_token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            if pages == 0:
                session_log = response.log or ""
            files.extend(response.files or [])
            page_token = response.next_page_token or None
            pages += 1
            if not page_token:
                break
            if pages > 10 or len(files) > 2_000:
                raise RuntimeError("provider output pagination exceeded safety bound")
    return files, session_log


def relative_result_path(file_name: str) -> PurePosixPath:
    """Validate a provider path before using it beneath the collection root."""
    if "\\" in file_name or "\x00" in file_name:
        raise ValueError("provider output path contains a forbidden character")
    relative_path = PurePosixPath(file_name)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or not file_name.startswith("results/")
    ):
        raise ValueError("provider output path is outside the selected results tree")
    return relative_path


def download_result_file(
    item: Any,
    output_root: Path,
    remaining_bytes: int,
) -> dict[str, Any]:
    """Stream one declared result under the remaining collection-byte budget."""
    file_name = str(item.file_name)
    relative_path = relative_result_path(file_name)
    output_path = output_root.joinpath(*relative_path.parts)
    if not output_path.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("provider output path escaped the collection directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = str(item.url)
    if urlparse(url).scheme != "https":
        raise ValueError("provider returned a non-HTTPS output URL")

    partial_path = output_path.with_name(output_path.name + ".partial")
    digest = hashlib.sha256()
    file_bytes = 0
    with requests.get(url, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        content_length_text = response.headers.get("Content-Length")
        if content_length_text is not None:
            content_length = int(content_length_text)
            if content_length > remaining_bytes:
                raise ValueError("provider result exceeds remaining collection-byte budget")
        try:
            with partial_path.open("xb") as stream:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    if not chunk:
                        continue
                    if file_bytes + len(chunk) > remaining_bytes:
                        raise ValueError("streamed provider result exceeds collection-byte budget")
                    stream.write(chunk)
                    digest.update(chunk)
                    file_bytes += len(chunk)
        except BaseException:
            partial_path.unlink(missing_ok=True)
            raise

    if content_length_text is not None and file_bytes != int(content_length_text):
        partial_path.unlink(missing_ok=True)
        raise OSError("provider result length differs from Content-Length")
    os.replace(partial_path, output_path)
    return {
        "path": relative_path.as_posix(),
        "size_bytes": file_bytes,
        "sha256": digest.hexdigest(),
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Collect R3 results/logs under a strict byte cap and write terminal receipt."""
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    terminal_receipt = args.terminal_receipt.resolve()
    expected_output = (
        repo_root
        / "research/campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r3/collected/attempt_001"
    ).resolve()
    expected_receipt = (
        repo_root
        / "research/campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r3/TERMINAL_R3.json"
    ).resolve()
    if output_dir != expected_output or terminal_receipt != expected_receipt:
        raise ValueError("collection paths must match the approved R3 target")
    if not repo_root.is_dir():
        raise FileNotFoundError("repository root is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("approved collection directory is not empty")
    if terminal_receipt.exists():
        raise FileExistsError("terminal receipt already exists")
    free_bytes = os.statvfs(repo_root).f_bavail * os.statvfs(repo_root).f_frsize
    if free_bytes < 2_000_000_000:
        raise OSError("physical collection filesystem has less than 2 GB free")

    api = KaggleApi()
    api.authenticate()
    status, failure_message = terminal_status(api)
    quota_before = quota_snapshot(api)
    files, session_log = output_listing(api)
    selected_files = [
        item for item in files if str(getattr(item, "file_name", "")).startswith("results/")
    ]
    if not files:
        raise RuntimeError("provider session output inventory is empty")

    # Root may pre-create this approved collection directory. It must still be
    # empty immediately before writing; exclusive file creation below protects
    # each selected output against a concurrent collector.
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(output_dir.iterdir()):
        raise FileExistsError("approved collection directory became non-empty")
    log_bytes = session_log.encode("utf-8")
    if len(log_bytes) > MAX_COLLECTION_BYTES:
        raise ValueError("provider kernel log exceeds collection-byte budget")
    log_path = output_dir / LOG_NAME
    with log_path.open("xb") as stream:
        stream.write(log_bytes)
    downloaded_bytes = len(log_bytes)
    manifest_files: list[dict[str, Any]] = [
        {
            "path": LOG_NAME,
            "size_bytes": len(log_bytes),
            "sha256": hashlib.sha256(log_bytes).hexdigest(),
        }
    ]

    for item in selected_files:
        record = download_result_file(
            item,
            output_dir,
            MAX_COLLECTION_BYTES - downloaded_bytes,
        )
        downloaded_bytes += int(record["size_bytes"])
        manifest_files.append(record)

    manifest = {
        "schema": "qat-kaggle-f0-s4-r3-bounded-output-collection",
        "collected_utc": utc_now(),
        "kernel": KERNEL_ID,
        "terminal_status": status,
        "provider_failure_message": failure_message,
        "provider_output_file_count": len(files),
        "selected_results_file_count": len(selected_files),
        "source_bundle_files_not_downloaded": len(files) - len(selected_files),
        "selection": "results/** plus the provider session log; source bundle is pinned separately",
        "maximum_collection_bytes": MAX_COLLECTION_BYTES,
        "payload_bytes_before_manifest": downloaded_bytes,
        "files": manifest_files,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if downloaded_bytes + len(manifest_bytes) > MAX_COLLECTION_BYTES:
        raise ValueError("collection manifest would exceed the collection-byte budget")
    manifest_path = output_dir / "collection_manifest.json"
    with manifest_path.open("xb") as stream:
        stream.write(manifest_bytes)
    final_bytes = downloaded_bytes + len(manifest_bytes)

    quota_after = quota_snapshot(api)
    packet_dir = repo_root / "research/campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r3"
    push_receipt = packet_dir / "KERNEL_PUSH_R3.json"
    admission_receipt = packet_dir / "ROOT_ADMISSION_R3.json"
    terminal = {
        "schema": "qat-kaggle-f0-s4-profile-terminal-r3",
        "recorded_utc": utc_now(),
        "kernel": KERNEL_ID,
        "provider_status": status,
        "provider_failure_message": failure_message,
        "provider_version": 1,
        "provider_attempt": 1,
        "attempt_limit": 1,
        "automatic_retries": 0,
        "profile_only": True,
        "fit_authorized": False,
        "root_admission_sha256": sha256_file(admission_receipt),
        "push_receipt_sha256": sha256_file(push_receipt),
        "bootstrap_sha256": "5222029435057c0241afd8069550acee9651571cb25d4eef642d8099327f3d3f",
        "bundle_archive_sha256": "6cfc0b87762e9ad91be1c9528ee767cb5c021364f62620d8c5b63e04cf383a6a",
        "collection": {
            "path": str(output_dir),
            "maximum_bytes": MAX_COLLECTION_BYTES,
            "bytes_including_manifest": final_bytes,
            "provider_output_file_count": len(files),
            "selected_results_file_count": len(selected_files),
            "source_bundle_files_not_downloaded": len(files) - len(selected_files),
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "files": manifest_files,
        },
        "quota_before_collection": quota_before,
        "quota_after_collection": quota_after,
        "scientific_interpretation": "pending runner-receipt diagnosis; provider ERROR alone is not a profile result",
        "next_action": "preserve terminal outputs, diagnose first failed phase, and do not retry or push another version",
    }
    prior_abort = Path(__file__).with_name("COLLECTION_ATTEMPT_FAILURE_R3.json")
    if prior_abort.is_file():
        terminal["prior_collector_abort"] = {
            "path": prior_abort.name,
            "sha256": sha256_file(prior_abort),
            "payload_downloaded": False,
            "provider_mutation": False,
        }
    write_json(terminal_receipt, terminal)
    logging.info(
        "collected %d result files plus session log (%d bytes of %d maximum)",
        len(selected_files),
        final_bytes,
        MAX_COLLECTION_BYTES,
    )
    return terminal


def parse_args() -> argparse.Namespace:
    """Parse the two approved output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run one terminal collection without retrying provider operations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    collect(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
