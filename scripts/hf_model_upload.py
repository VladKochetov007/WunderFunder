from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from hf_model_inventory import REPO_ID
from huggingface_hub import HfApi


def assert_private(api: HfApi, anonymous_check: bool = False) -> str:
    """Refuse any destination that is not the exact approved private model repo."""
    info = api.model_info(REPO_ID)
    if info.private is not True:
        raise ValueError("Destination is not private; no payload upload is allowed")
    if anonymous_check and HfApi(token=False).repo_exists(REPO_ID, repo_type="model"):
        raise ValueError("Anonymous access unexpectedly exposes the model repo")
    return str(info.sha)


def identities(path: Path) -> tuple[int, str, str]:
    """Compute size, SHA-256 and Git identity without unpickling a payload."""
    size = path.stat().st_size
    sha = hashlib.sha256()
    git_sha = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            sha.update(chunk)
            git_sha.update(chunk)
    return size, sha.hexdigest(), git_sha.hexdigest()


def remote_matches(api: HfApi, entries: list[dict[str, object]]) -> bool:
    """Check committed size and server-reported SHA-256 or Git blob identity."""
    paths = [str(entry["hub_path"]) for entry in entries]
    remote = {item.path: item for item in api.get_paths_info(REPO_ID, paths, repo_type="model")}
    for entry in entries:
        item = remote.get(str(entry["hub_path"]))
        if item is None or item.size != entry["bytes"]:
            return False
        if item.lfs is not None:
            if item.lfs.sha256 != entry["sha256"]:
                return False
        elif item.blob_id != entry["git_blob_sha1"]:
            return False
    return True


def write_receipt(path: Path, receipt: dict[str, object]) -> None:
    """Atomically preserve observed transfer state, including incomplete batches."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)


def batches(entries: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    """Bound each transfer to 100 files and about 128 MB of payload."""
    result: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    total = 0
    for entry in entries:
        size = int(entry["bytes"])
        if current and (len(current) >= 100 or total + size > 128_000_000):
            result.append(current)
            current, total = [], 0
        current.append(entry)
        total += size
    if current:
        result.append(current)
    return result


def collect(stage: Path, lane: str, inventory: dict[str, object]) -> list[dict[str, object]]:
    """Use the frozen inventory and hash only newly generated model cards."""
    by_hub_path = {str(e["hub_path"]): e for e in inventory["files"] if "duplicate_of" not in e}
    entries = []
    for path in sorted((stage / lane).rglob("*")):
        if not path.is_file():
            continue
        hub_path = path.relative_to(stage / lane).as_posix()
        if hub_path in by_hub_path:
            entry = dict(by_hub_path[hub_path])
            if (
                path.stat().st_size != entry["bytes"]
                or path.stat().st_mtime_ns != entry["mtime_ns"]
            ):
                raise ValueError(f"Inventoried source changed: {entry['local_path']}")
        else:
            size, sha, git_sha = identities(path)
            entry = {
                "hub_path": hub_path,
                "bytes": size,
                "sha256": sha,
                "git_blob_sha1": git_sha,
                "kind": "archive_documentation",
            }
        entry["stage_path"] = str(path)
        entries.append(entry)
    return entries


def transfer(
    api: HfApi, batch: list[dict[str, object]], directory: Path, log: Path, label: str
) -> str:
    """Upload an explicit hardlinked batch with the authenticated HF CLI."""
    assert_private(api)
    directory.mkdir(parents=True, exist_ok=True)
    for entry in batch:
        target = directory / str(entry["hub_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(Path(str(entry["stage_path"])), target)
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_DISABLE_XET": "1",
            "HF_XET_HIGH_PERFORMANCE": "0",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
    with log.open("w") as handle:
        result = subprocess.run(
            [
                "hf",
                "upload",
                REPO_ID,
                str(directory),
                ".",
                "--repo-type",
                "model",
                "--private",
                "--commit-message",
                label,
            ],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"HF upload failed with exit {result.returncode}; preserved log: {log}")
    revision = assert_private(api)
    if not remote_matches(api, batch):
        raise ValueError(f"Remote hash/size verification failed for {label}")
    shutil.rmtree(directory)
    return revision


def main() -> None:
    """Deliver all staged models in priority order and verify every committed file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    if inventory["repo_id"] != REPO_ID or inventory["required_private"] is not True:
        raise ValueError("Inventory does not bind the approved private model repository")
    stage = Path(inventory["stage"])
    api = HfApi()
    initial_revision = assert_private(api, anonymous_check=True)
    receipt_path = args.inventory.parent / "UPLOAD_RECEIPT_20261012.json"
    receipt: dict[str, object] = {
        "schema": "wnn.hf-private-model-upload.v1",
        "repo_id": REPO_ID,
        "repo_type": "model",
        "private": True,
        "started_utc": datetime.now(UTC).isoformat(),
        "initial_revision": initial_revision,
        "status": "running",
        "batches": [],
    }
    write_receipt(receipt_path, receipt)
    operations = stage.parent / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    all_entries: list[dict[str, object]] = []
    for lane in ["best", "priority", "archive"]:
        entries = collect(stage, lane, inventory)
        all_entries.extend(entries)
        for index, batch in enumerate(batches(entries), 1):
            label = f"{lane} models batch {index:03d}"
            event = {
                "lane": lane,
                "batch": index,
                "files": len(batch),
                "bytes": sum(int(e["bytes"]) for e in batch),
                "status": "started",
                "utc": datetime.now(UTC).isoformat(),
            }
            receipt["batches"].append(event)
            write_receipt(receipt_path, receipt)
            if remote_matches(api, batch):
                revision = assert_private(api)
                event["status"] = "already_present_verified"
            else:
                logging.info(
                    "Uploading %s: %s files / %.1f MB",
                    label,
                    len(batch),
                    event["bytes"] / 1_000_000,
                )
                try:
                    revision = transfer(
                        api,
                        batch,
                        operations / f"{lane}-{index:03d}",
                        operations / f"{lane}-{index:03d}.log",
                        label,
                    )
                except (RuntimeError, ValueError):
                    event["status"] = "failed"
                    receipt["status"] = "incomplete"
                    write_receipt(receipt_path, receipt)
                    raise
                event["status"] = "uploaded_hash_verified"
            event["revision"] = revision
            event["private_verified"] = True
            write_receipt(receipt_path, receipt)
            logging.info("Verified %s (%s)", label, event["status"])
        receipt[f"{lane}_complete"] = True
        write_receipt(receipt_path, receipt)
        logging.info("Lane %s complete", lane)
    receipt.update(
        {
            "status": "complete",
            "completed_utc": datetime.now(UTC).isoformat(),
            "final_revision": assert_private(api, anonymous_check=True),
            "unique_files_verified": len(all_entries),
            "bytes_verified": sum(int(e["bytes"]) for e in all_entries),
            "verification": "Every staged file matched remote size and LFS SHA-256 or Git blob identity; private checked before and after every batch; anonymous repository visibility absent at start and finish.",
        }
    )
    write_receipt(receipt_path, receipt)
    logging.info(
        "Complete: %s files / %.3f GB; private",
        len(all_entries),
        receipt["bytes_verified"] / 1_000_000_000,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
