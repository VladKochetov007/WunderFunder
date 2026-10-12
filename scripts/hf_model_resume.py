from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from hf_model_inventory import REPO_ID
from hf_model_upload import assert_private, collect, remote_matches, write_receipt
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError


def main() -> None:
    """Resume existing disk payloads with fewer streamed commits and quota backoff."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    note = args.inventory.parent
    inventory = json.loads(args.inventory.read_text())
    if inventory["repo_id"] != REPO_ID or inventory["required_private"] is not True:
        raise ValueError("Inventory must bind the approved private model repository")
    api = HfApi()
    assert_private(api, anonymous_check=True)
    previous = note / "UPLOAD_RECEIPT_20261012.json"
    frozen = note / "UPLOAD_RECEIPT_ATTEMPT1_20261012.json"
    if not frozen.exists():
        frozen.write_bytes(previous.read_bytes())
    receipt = {
        "repo_id": REPO_ID,
        "repo_type": "model",
        "private": True,
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "basis": "Attempt 1 stopped at the observed 128 commits/hour limit. Reuse committed payloads; transfer only existing physical-disk files; at most 100 files per commit with streamed LFS bodies.",
        "prior_attempt": str(frozen),
        "batches": [],
    }
    output = note / "RESUME_UPLOAD_RECEIPT_20261012.json"
    write_receipt(output, receipt)
    stage = Path(inventory["stage"])
    entries = []
    for lane in ["best", "priority", "archive"]:
        entries.extend(collect(stage, lane, inventory))
    for name in ["worktree-stage", "additional-stage"]:
        directory = stage.parent / name
        if not directory.is_dir():
            raise ValueError(f"Missing existing supplement stage: {directory}")
        entries.extend(collect(directory, ".", {"files": []}))
    info = api.model_info(REPO_ID, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    pending = []
    for entry in entries:
        item = remote.get(entry["hub_path"])
        matches = item is not None and item.size == entry["bytes"]
        if matches:
            matches = (
                item.lfs.sha256 == entry["sha256"]
                if item.lfs
                else item.blob_id == entry["git_blob_sha1"]
            )
        if not matches:
            pending.append(entry)
    receipt.update(
        {
            "expected_files": len(entries),
            "expected_bytes": sum(e["bytes"] for e in entries),
            "already_verified_files": len(entries) - len(pending),
            "pending_files": len(pending),
            "pending_bytes": sum(e["bytes"] for e in pending),
        }
    )
    write_receipt(output, receipt)
    commits = api.list_repo_commits(REPO_ID, repo_type="model")
    ours = [c.created_at.timestamp() for c in commits if c.title != "initial commit"]
    not_before = min(ours) + 3660 if ours else 0
    groups = []
    current, size = [], 0
    for entry in pending:
        if current and (len(current) >= 100 or size + entry["bytes"] > 1_000_000_000):
            groups.append(current)
            current, size = [], 0
        current.append(entry)
        size += entry["bytes"]
    if current:
        groups.append(current)
    logging.info(
        "Resume: %s existing files verified; %s pending files / %.3f GB in %s larger commits",
        len(entries) - len(pending),
        len(pending),
        receipt["pending_bytes"] / 1e9,
        len(groups),
    )
    for number, group in enumerate(groups, 1):
        assert_private(api)
        event = {
            "batch": number,
            "files": len(group),
            "bytes": sum(e["bytes"] for e in group),
            "status": "uploading_existing_files",
            "started_utc": datetime.now(UTC).isoformat(),
        }
        receipt["batches"].append(event)
        write_receipt(output, receipt)
        operations = [
            CommitOperationAdd(path_in_repo=e["hub_path"], path_or_fileobj=e["stage_path"])
            for e in group
        ]
        api.preupload_lfs_files(
            REPO_ID, operations, repo_type="model", num_threads=5, free_memory=True
        )
        event["status"] = "binary_uploaded_pending_commit"
        write_receipt(output, receipt)
        while True:
            delay = not_before - time.time()
            while delay > 0:
                event["status"] = "waiting_for_commit_quota"
                event["not_before_utc"] = datetime.fromtimestamp(not_before, UTC).isoformat()
                write_receipt(output, receipt)
                logging.info("Commit quota wait: %.0f seconds remaining", delay)
                time.sleep(min(delay, 45))
                delay = not_before - time.time()
            assert_private(api)
            try:
                commit = api.create_commit(
                    REPO_ID,
                    repo_type="model",
                    operations=operations,
                    commit_message=f"Existing local models resume batch {number:03d}",
                    num_threads=5,
                )
                break
            except HfHubHTTPError as error:
                response = error.response
                if response is None or response.status_code != 429:
                    event["status"] = "failed"
                    event["error_type"] = type(error).__name__
                    receipt["status"] = "incomplete"
                    write_receipt(output, receipt)
                    raise RuntimeError("HF transfer failed; see sanitized resume receipt") from None
                wait = response.headers.get("Retry-After", "3600")
                seconds = int(wait) if wait.isdigit() else 3600
                not_before = time.time() + max(seconds, 60) + 10
                event.setdefault("rate_limit_events", []).append(
                    {
                        "utc": datetime.now(UTC).isoformat(),
                        "status_code": 429,
                        "retry_after_seconds": seconds,
                        "next_commit_not_before": datetime.fromtimestamp(
                            not_before, UTC
                        ).isoformat(),
                    }
                )
                write_receipt(output, receipt)
                logging.warning(
                    "Commit rate limit remains active; preserved upload and honoring %.0f-second backoff",
                    seconds,
                )
        assert_private(api)
        if not remote_matches(api, group):
            raise ValueError("Committed payloads failed hash/size verification")
        event.update(
            {
                "status": "uploaded_hash_verified",
                "revision": commit.oid,
                "private_verified": True,
                "completed_utc": datetime.now(UTC).isoformat(),
            }
        )
        write_receipt(output, receipt)
        logging.info(
            "Verified resume batch %s/%s: %.1f MB", number, len(groups), event["bytes"] / 1e6
        )
    receipt.update(
        {
            "status": "complete",
            "completed_utc": datetime.now(UTC).isoformat(),
            "final_revision": assert_private(api, anonymous_check=True),
        }
    )
    write_receipt(output, receipt)
    logging.info("All existing disk payloads uploaded and verified; private")


if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        main()
    except (RuntimeError, ValueError, HfHubHTTPError):
        subprocess.run(
            [
                "notify-send",
                "--app-name=Codex",
                "--urgency=critical",
                "--expire-time=0",
                "HF model upload needs attention",
                "The existing-weight transfer stopped; uploaded models remain preserved. Codex is checking the failure.",
            ],
            check=False,
        )
        raise
