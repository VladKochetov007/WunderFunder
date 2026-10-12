from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path("/home/vlad/development/wunder/wnn-agent-worktrees/f1-core-hft-review-20261012")
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = ROOT / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort/r6_bootstrap_guard_successor_20261012_r1"
KERNEL = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5-runner"
PUSH_RECEIPT = PACKET / "R6_KERNEL_VERSION2_PUSH_RECEIPT_R1.json"
ROOT_RELEASE = PACKET.parent / "ROOT_R6_VERSION2_DISPATCH_RELEASE_R1.json"
POLL_PATH = Path("/tmp/wnn_r6_kernel_v2_status_poll_20261012.jsonl")
COLLECTION = PACKET / "terminal_collection_v2"
INVENTORY_PATH = COLLECTION / "PROVIDER_INVENTORY_R1.json"

sys.path.insert(0, str(MAIN / "research/campaign_2026-10-06/cloud_preflight"))

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

import poll_provider


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for a byte string."""
    return hashlib.sha256(value).hexdigest()


def safe_name(value: str) -> bool:
    """Reject provider names that could escape an output directory."""
    path = PurePosixPath(value)
    return bool(
        value
        and not path.is_absolute()
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in path.parts)
        and not re.search(r"[\x00-\x1f]", value)
    )


def main() -> int:
    """Capture safe provider output filenames and response.log closure metadata."""
    if COLLECTION.exists() or INVENTORY_PATH.exists():
        print(json.dumps({"state": "refused_existing_collection_path"}))
        return 2
    release = json.loads(ROOT_RELEASE.read_text())
    push = json.loads(PUSH_RECEIPT.read_text())
    poll = json.loads(POLL_PATH.read_text().splitlines()[-1])
    if (
        release.get("authorized") is not True
        or release.get("kernel_id") != KERNEL
        or release.get("intended_kernel_version") != 2
        or release.get("archive_sha256") != push.get("archive_sha256")
        or release.get("metadata_sha256") != push.get("metadata_sha256")
        or push.get("provider_version") != 2
        or push.get("status") != "accepted_provider_queued"
        or not poll.get("status")
        or poll["status"][0].get("status") != "KernelWorkerStatus.ERROR"
    ):
        print(json.dumps({"state": "refused_identity_or_terminal_gate_mismatch"}))
        return 3

    api = KaggleApi()
    api.authenticate()
    client = poll_provider._bounded_client(api)
    owner, slug, _version = api.parse_kernel_string(KERNEL)
    files_by_name: dict[str, dict[str, Any]] = {}
    pages: list[dict[str, Any]] = []
    requested_tokens: set[str] = set()
    token: str | None = None
    first_log: dict[str, Any] | None = None
    try:
        for page_index in range(100):
            if token is not None and token in requested_tokens:
                print(json.dumps({"state": "refused_pagination_token_cycle"}))
                return 4
            if token is not None:
                requested_tokens.add(token)
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            response_log = (response.log or "").encode()
            response_log_info = {
                "size_bytes": len(response_log),
                "sha256": sha256_bytes(response_log),
            }
            if page_index == 0:
                first_log = response_log_info
            page_files: list[dict[str, Any]] = []
            for item in response.files or []:
                name = str(item.file_name)
                if not safe_name(name) or name in files_by_name:
                    print(json.dumps({"state": "refused_unsafe_or_duplicate_filename"}))
                    return 5
                record = {"name": name, "remote_size_bytes": None}
                files_by_name[name] = record
                page_files.append(record)
            next_token = response.next_page_token or None
            safe_page = {
                "schema": "wnn.kaggle.session_output_page.r1",
                "page_index": page_index + 1,
                "endpoint": "list_kernel_session_output",
                "kernel_slug": KERNEL,
                "version_label_sent": None,
                "request_page_token_sha256": sha256_bytes(token.encode()) if token else None,
                "page_size": 200,
                "files": page_files,
                "next_page_token_sha256": sha256_bytes(next_token.encode()) if next_token else None,
                "response_log_sha256": response_log_info["sha256"],
                "response_log_bytes": response_log_info["size_bytes"],
            }
            canonical = json.dumps(safe_page, sort_keys=True, separators=(",", ":")).encode()
            pages.append(
                {
                    "page_index": page_index + 1,
                    "row_count": len(page_files),
                    "safe_page_sha256": sha256_bytes(canonical),
                    "request_page_token_sha256": safe_page["request_page_token_sha256"],
                    "next_page_token_sha256": safe_page["next_page_token_sha256"],
                    "response_log": response_log_info,
                }
            )
            if next_token is None:
                break
            if next_token == token:
                print(json.dumps({"state": "refused_pagination_token_cycle"}))
                return 4
            token = next_token
        else:
            print(json.dumps({"state": "refused_page_limit"}))
            return 6
    finally:
        client.__exit__(None, None, None)

    if not pages:
        print(json.dumps({"state": "refused_empty_inventory"}))
        return 7
    status = poll["status"][0]
    quota = poll.get("quota", {}).get("gpu", {})
    body: dict[str, Any] = {
        "schema": "wnn.kaggle.r6_v2_readonly_provider_inventory.r1",
        "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "kernel_id": KERNEL,
        "accepted_kernel_version": 2,
        "kernel_version_pinned_by_endpoint": False,
        "version_binding": {
            "source": "root R6 release plus accepted provider push receipt",
            "root_release_sha256": sha256_bytes(ROOT_RELEASE.read_bytes()),
            "push_receipt_sha256": sha256_bytes(PUSH_RECEIPT.read_bytes()),
            "provider_version": push.get("provider_version"),
        },
        "provider_status": {
            "observed_utc": poll.get("observed_at_utc"),
            "status": status.get("status"),
            "failure_message": status.get("failure_message", ""),
            "poll_snapshot_sha256": sha256_bytes(POLL_PATH.read_bytes()),
        },
        "quota_snapshot": {
            "observed_utc": poll.get("observed_at_utc"),
            "gpu_used_seconds": quota.get("used_seconds"),
            "gpu_reserved_seconds": quota.get("reserved_seconds"),
            "campaign_headroom_gpu_seconds": quota.get("campaign_headroom_seconds"),
            "provider_refresh_utc": poll.get("quota_refresh_utc"),
        },
        "endpoint": "list_kernel_session_output",
        "remote_size_or_checksum_available": False,
        "response_log": first_log,
        "file_count": len(files_by_name),
        "files": [files_by_name[name] for name in sorted(files_by_name)],
        "pages": pages,
        "inventory_complete": True,
        "signed_urls_persisted": False,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["inventory_sha256"] = sha256_bytes(canonical)
    COLLECTION.mkdir(mode=0o700)
    os.chmod(COLLECTION, 0o700)
    with INVENTORY_PATH.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.chmod(INVENTORY_PATH, 0o600)
    print(
        json.dumps(
            {
                "state": "inventory_complete",
                "kernel": KERNEL,
                "status": status.get("status"),
                "file_count": body["file_count"],
                "response_log": first_log,
                "inventory_sha256": body["inventory_sha256"],
                "path": str(INVENTORY_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

