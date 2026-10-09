"""Restage the V5R2 private source dataset and outer kernel after root authorization (no rebuild)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PACKET = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKET))

import build_fit_packet as builder  # noqa: E402


def main() -> None:
    upload_root = PACKET / "source_dataset_upload"
    if upload_root.exists():
        shutil.rmtree(upload_root)
    upload_root.mkdir(parents=True)
    payload_files = [
        "capacity_worker.py", "capacity_fit_worker.py", "capacity_arm_worker.py",
        "train_entry.py", "verified_digest_handoff.py", "kernel.py", "kernel-metadata.json",
        "README.md", "GRU512_768_MUON_FIT_RECIPE_V5R2.json", "ROOT_FIT_SOURCE_REVIEW.json",
        "ROOT_FIT_AUTHORIZATION.json", "source/source_archive.tar", "source/source_archive_manifest.json",
        "fold0.json", "portable_inputs.json", "prior_input_identity_v2.json",
        "producer_evidence/v4/manifest.json",
        *[f"producer_evidence/v4/{name}" for name in builder.EVIDENCE_FILES],
    ]
    descriptors = []
    for relative in sorted(set(payload_files)):
        source = PACKET / relative
        digest = builder.sha256_file(source)
        destination = upload_root / f"{digest}.wnn"
        shutil.copyfile(source, destination)
        descriptors.append({"path": relative, "size": source.stat().st_size, "sha256": digest,
                            "filename": destination.name})
    manifest = {"version": 1, "dataset": builder.SOURCE_DATASET, "private": True, "files": descriptors}
    manifest_path = upload_root / "assets-manifest.json"
    builder.write_json(manifest_path, manifest)
    builder.write_json(upload_root / "dataset-metadata.json", {
        "id": builder.SOURCE_DATASET,
        "title": builder.SOURCE_DATASET.split("/", 1)[1],
        "isPrivate": True,
        "licenses": [{"name": "other"}],
    })
    outer = builder._outer_kernel_source(builder.sha256_file(manifest_path), [row["path"] for row in descriptors])
    outer_root = PACKET / "kernel_upload"
    outer_root.mkdir(exist_ok=True)
    (outer_root / "kernel.py").write_text(outer, encoding="utf-8")
    shutil.copyfile(PACKET / "kernel-metadata.json", outer_root / "kernel-metadata.json")
    receipt = {
        "schema": "wnn-gru-capacity-fit-v5r2-authorized-restage-v1",
        "packet_id": builder.PACKET_ID,
        "source_dataset": builder.SOURCE_DATASET,
        "source_dataset_manifest_sha256": builder.sha256_file(manifest_path),
        "source_dataset_files": len(descriptors),
        "outer_driver_sha256": builder.sha256_file(outer_root / "kernel.py"),
        "outer_metadata_sha256": builder.sha256_file(outer_root / "kernel-metadata.json"),
        "authorization_sha256": builder.sha256_file(PACKET / "ROOT_FIT_AUTHORIZATION.json"),
        "fit_started": False,
    }
    builder.write_json(PACKET / "V5R2_AUTHORIZED_RESTAGE_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
