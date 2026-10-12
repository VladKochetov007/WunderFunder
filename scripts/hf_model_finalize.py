from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from hf_model_inventory import REPO_ID
from hf_model_upload import assert_private, batches, collect, transfer, write_receipt
from huggingface_hub import HfApi


def main() -> None:
    """Upload existing supplemental files and verify the complete private archive."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    note = args.inventory.parent
    inventory = json.loads(args.inventory.read_text())
    resumed = json.loads((note / "RESUME_UPLOAD_RECEIPT_20261012.json").read_text())
    if resumed["status"] != "complete":
        raise ValueError("Main transfer must complete before finalization")
    if inventory["repo_id"] != REPO_ID or inventory["required_private"] is not True:
        raise ValueError("Inventory must bind the approved private model repository")
    stage = Path(inventory["stage"])
    api = HfApi()
    assert_private(api, anonymous_check=True)
    entries = []
    for lane in ["best", "priority", "archive"]:
        entries.extend(collect(stage, lane, inventory))
    supplements = []
    for name in ["worktree-stage", "additional-stage"]:
        directory = stage.parent / name
        if not directory.is_dir():
            raise ValueError(f"Missing existing supplement stage: {directory}")
        supplements.extend(collect(directory, ".", {"files": []}))
    receipt_path = note / "SUPPLEMENT_UPLOAD_RECEIPT_20261012.json"
    receipt = {
        "repo_id": REPO_ID,
        "private": True,
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": "Existing standalone saved weights and own exported worktree packages only; no training or reconstruction.",
        "files": [{k: v for k, v in e.items() if k != "stage_path"} for e in supplements],
        "batches": [],
    }
    operations = stage.parent / "operations"
    write_receipt(receipt_path, receipt)
    for index, group in enumerate(batches(supplements), 1):
        revision = transfer(
            api,
            group,
            operations / f"supplement-final-{index:03d}",
            operations / f"supplement-final-{index:03d}.log",
            f"Existing supplemental models batch {index:03d}",
        )
        receipt["batches"].append(
            {
                "batch": index,
                "files": len(group),
                "bytes": sum(e["bytes"] for e in group),
                "revision": revision,
                "status": "uploaded_hash_verified",
                "private": True,
            }
        )
        write_receipt(receipt_path, receipt)
        logging.info("Verified supplemental batch %s", index)
    receipt.update(status="complete", completed_utc=datetime.now(UTC).isoformat())
    write_receipt(receipt_path, receipt)
    entries.extend(supplements)
    info = api.model_info(REPO_ID, files_metadata=True)
    if info.private is not True:
        raise ValueError("Final destination is not private")
    remote = {item.rfilename: item for item in info.siblings}
    failures = []
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
            failures.append(entry["hub_path"])
    anonymous_visible = HfApi(token=False).repo_exists(REPO_ID, repo_type="model")
    final = {
        "repo_id": REPO_ID,
        "repo_type": "model",
        "revision": info.sha,
        "private": info.private,
        "anonymous_visible": anonymous_visible,
        "verified_utc": datetime.now(UTC).isoformat(),
        "expected_files": len(entries),
        "expected_bytes": sum(e["bytes"] for e in entries),
        "main_files": resumed["expected_files"],
        "supplement_files": len(supplements),
        "integrity_failures": failures,
        "status": "complete" if not failures and not anonymous_visible else "failed",
        "method": "One authenticated files_metadata snapshot: size plus LFS SHA256 or Git blob SHA1 for every staged file; anonymous repository existence check.",
        "scope": "Existing own Challenge 2 model inventory and audited standalone/worktree supplements. No model training, inference or tensor reconstruction.",
    }
    write_receipt(note / "FINAL_PRIVACY_AND_INTEGRITY_20261012.json", final)
    if failures or anonymous_visible:
        raise ValueError("Final privacy/integrity verification failed; see receipt")
    manifest_path = note / "WORKTREE_UPLOAD_MANIFEST_20261012.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="uploaded_verified", verified_revision=info.sha)
    write_receipt(manifest_path, manifest)
    logging.info(
        "Complete: %s files / %.3f GB verified, strictly private",
        len(entries),
        final["expected_bytes"] / 1e9,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
