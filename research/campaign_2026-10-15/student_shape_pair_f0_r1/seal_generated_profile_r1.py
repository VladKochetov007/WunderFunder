from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r1"
BUNDLE = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r1"
SOURCE_BINDING = PACKET / "SOURCE_BINDING_R1.json"
OUTPUT = PACKET / "GENERATED_KERNEL_SEAL_R1.json"
OLD_DEADLINE = "KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200"
NEW_DEADLINE = "KERNEL_PROCESS_WALL_SECONDS_MAX = 1800"
EXPECTED_KERNEL_ID = (
    "vladkochetov07/wnn-f0-student-shape-t4-profile-20261015-r1-runner"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    binding = read_json(SOURCE_BINDING)
    manifest_path = BUNDLE / "bundle-manifest.json"
    dataset_metadata_path = BUNDLE / "dataset-metadata.json"
    archive_path = BUNDLE / "bundle.wnn"
    kernel_path = BUNDLE / "kernel/kernel.py"
    kernel_metadata_path = BUNDLE / "kernel/kernel-metadata.json"
    manifest = read_json(manifest_path)
    dataset_metadata = read_json(dataset_metadata_path)
    kernel_metadata = read_json(kernel_metadata_path)

    expected_dataset = binding["kernel_dataset"]
    if manifest["dataset"] != expected_dataset or not manifest["private"]:
        raise ValueError("Generated private source-bundle identity differs from its binding")
    if dataset_metadata["id"] != expected_dataset or not dataset_metadata["isPrivate"]:
        raise ValueError("Generated dataset metadata has the wrong ID or privacy setting")
    if kernel_metadata["id"] != EXPECTED_KERNEL_ID or not kernel_metadata["is_private"]:
        raise ValueError("Generated kernel ID or privacy setting differs")
    if not kernel_metadata["enable_gpu"] or kernel_metadata["enable_tpu"]:
        raise ValueError("Generated kernel must enable GPU only")
    if kernel_metadata["machine_shape"] != "NvidiaTeslaT4":
        raise ValueError("Generated kernel does not target the frozen T4 profile")
    if manifest["archive_sha256"] != sha256_file(archive_path):
        raise ValueError("Generated archive bytes differ from the bundle manifest")
    if manifest["archive_bytes"] != archive_path.stat().st_size:
        raise ValueError("Generated archive size differs from the bundle manifest")
    if set(kernel_metadata["dataset_sources"]) != {
        expected_dataset,
        "vladkochetov07/wnn-c20261004-fold0-assets",
        "vladkochetov07/wnn-clean-t3-f0-h384-cp2-20261012-r9",
        "vladkochetov07/wnn-clean-t3-f0-treatment-assets-20261012-r22",
        "vladkochetov07/wnn-f0-h256l2-cp5-norm-20261015-r1",
        "vladkochetov07/wunder-funder",
    }:
        raise ValueError("Generated kernel mounted dataset list differs from the reviewed inputs")

    before = kernel_path.read_text()
    if before.count(OLD_DEADLINE) != 1:
        raise ValueError("Expected exactly one generic kernel deadline literal")
    before_sha256 = sha256_file(kernel_path)
    patched = before.replace(OLD_DEADLINE, NEW_DEADLINE, 1)
    if patched.count(NEW_DEADLINE) != 1 or OLD_DEADLINE in patched:
        raise ValueError("Kernel deadline replacement did not produce one exact 1,800s cap")
    kernel_path.write_text(patched)
    if sha256_file(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Kernel patch unexpectedly altered the source archive")

    receipt = {
        "schema": "wnn.student-shape-pair-generated-kernel-seal.r1",
        "status": "offline_bundle_and_kernel_pinned_no_provider_action",
        "provider_action_performed": False,
        "private_dataset_created": False,
        "kernel_pushed": False,
        "source_code_commit": binding["source_tree_commit"],
        "source_binding_sha256": sha256_file(SOURCE_BINDING),
        "recipe_contract_sha256": binding["recipe_contract_sha256"],
        "profile_jobs": binding["profile_jobs"],
        "bundle": {
            "dataset": manifest["dataset"],
            "private": manifest["private"],
            "archive_path": str(archive_path.relative_to(ROOT)),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": sha256_file(archive_path),
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "expanded_bytes": manifest["expanded_bytes"],
            "external_mount_count": len(manifest["external_mounts"]),
        },
        "dataset_metadata": {
            "path": str(dataset_metadata_path.relative_to(ROOT)),
            "sha256": sha256_file(dataset_metadata_path),
            "id": dataset_metadata["id"],
            "private": dataset_metadata["isPrivate"],
        },
        "kernel": {
            "path": str(kernel_path.relative_to(ROOT)),
            "kernel_id": kernel_metadata["id"],
            "machine_shape": kernel_metadata["machine_shape"],
            "enabled_gpu": kernel_metadata["enable_gpu"],
            "enabled_tpu": kernel_metadata["enable_tpu"],
            "kernel_metadata_path": str(kernel_metadata_path.relative_to(ROOT)),
            "kernel_metadata_sha256": sha256_file(kernel_metadata_path),
            "unpatched_sha256": before_sha256,
            "patched_sha256": sha256_file(kernel_path),
            "deadline_before": OLD_DEADLINE,
            "deadline_after": NEW_DEADLINE,
            "deadline_seconds": 1800,
            "patch_occurrences": 1,
        },
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
