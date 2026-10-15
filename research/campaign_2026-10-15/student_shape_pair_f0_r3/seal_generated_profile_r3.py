from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

# Parsed JSON uses heterogeneous nested values defined by the frozen packet schemas.

ROOT = Path(__file__).resolve().parents[3]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r3"
BUNDLE_R2 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r2"
BUNDLE = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r3"
SOURCE_BINDING = PACKET / "SOURCE_BINDING_R3.json"
INPUT_BINDING = PACKET / "INPUT_BINDING_R3.json"
MOUNT_SUCCESSOR = PACKET / "KERNEL_MOUNT_SUCCESSOR_R3.json"
BUILD_RECEIPT = PACKET / "OFFLINE_BUNDLE_BUILD_R3.json"
R9_MANIFEST = MAIN / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_f0_h390_r9_export_from_owned_checkpoints_20261012_r2/"
    "R9_CHECKPOINT_INPUT_MANIFEST_R2.json"
)
OLD_DEADLINE = "KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200"
NEW_DEADLINE = "KERNEL_PROCESS_WALL_SECONDS_MAX = 1800"


def sha256_file(path: Path) -> str:
    """Hash one generated artifact with bounded memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a small, local packet manifest."""
    return json.loads(path.read_text())


def external_map(binding: dict[str, Any]) -> dict[str, object]:
    """Return the mount map expected by the generated Kaggle kernel."""
    return {
        row["logical_path"]: (
            row["filename"]
            if row["dataset"] == "vladkochetov07/wunder-funder"
            else {"dataset": row["dataset"], "filename": row["filename"]}
        )
        for row in binding["external_mounts"]
    }


def kernel_external_map(kernel_path: Path) -> dict[str, object]:
    """Extract the kernel's literal mount map without importing remote code."""
    tree = ast.parse(kernel_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_EXTERNAL_MOUNTS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError("generated kernel mount constant is not a mapping")
            return value
    raise ValueError("generated kernel has no EXPECTED_EXTERNAL_MOUNTS constant")


def main() -> None:
    """Verify and seal the offline R3 bundle with its 1,800-second deadline."""
    source_binding = read_json(SOURCE_BINDING)
    input_binding = read_json(INPUT_BINDING)
    mount_successor = read_json(MOUNT_SUCCESSOR)
    build_receipt = read_json(BUILD_RECEIPT)
    r9_manifest = read_json(R9_MANIFEST)
    manifest_path = BUNDLE / "bundle-manifest.json"
    dataset_metadata_path = BUNDLE / "dataset-metadata.json"
    archive_path = BUNDLE / "bundle.wnn"
    kernel_path = BUNDLE / "kernel/kernel.py"
    kernel_metadata_path = BUNDLE / "kernel/kernel-metadata.json"
    manifest = read_json(manifest_path)
    dataset_metadata = read_json(dataset_metadata_path)
    kernel_metadata = read_json(kernel_metadata_path)

    expected_dataset = source_binding["kernel_dataset"]
    expected_kernel_id = source_binding["kernel_id"]
    if manifest["dataset"] != expected_dataset or manifest["private"] is not True:
        raise ValueError(
            "Generated private source-bundle identity differs from its binding"
        )
    if (
        dataset_metadata["id"] != expected_dataset
        or dataset_metadata["isPrivate"] is not True
    ):
        raise ValueError(
            "Generated dataset metadata has the wrong ID or privacy setting"
        )
    if (
        kernel_metadata["id"] != expected_kernel_id
        or kernel_metadata["is_private"] is not True
    ):
        raise ValueError("Generated kernel ID or privacy setting differs")
    if not kernel_metadata["enable_gpu"] or kernel_metadata["enable_tpu"]:
        raise ValueError("Generated kernel must enable GPU only")
    if kernel_metadata["machine_shape"] != "NvidiaTeslaT4":
        raise ValueError("Generated kernel does not target the frozen T4 profile")
    if manifest["archive_sha256"] != sha256_file(archive_path):
        raise ValueError("Generated archive bytes differ from the bundle manifest")
    if manifest["archive_bytes"] != archive_path.stat().st_size:
        raise ValueError("Generated archive size differs from the bundle manifest")
    r2_archive = BUNDLE_R2 / "bundle.wnn"
    if (
        build_receipt["archive_byte_identical_to_r2"] is not True
        or build_receipt["r3_source_manifest_sha256"] != sha256_file(manifest_path)
        or build_receipt["r3_kernel_metadata_sha256"]
        != sha256_file(kernel_metadata_path)
        or sha256_file(archive_path) != sha256_file(r2_archive)
        or sha256_file(archive_path) != mount_successor["r3_archive_sha256"]
        or sha256_file(r2_archive) != mount_successor["r2_archive_sha256"]
    ):
        raise ValueError("R3 transport archive is not byte-identical to sealed R2")
    if source_binding["bundle_successor_receipt_sha256"] != sha256_file(BUILD_RECEIPT):
        raise ValueError("R3 bundle successor receipt differs from source binding")
    if source_binding["kernel_mount_successor_receipt_sha256"] != sha256_file(
        MOUNT_SUCCESSOR
    ):
        raise ValueError("R3 kernel mount receipt differs from source binding")

    expected_mounts = external_map(input_binding)
    if manifest["external_mounts"] != expected_mounts:
        raise ValueError(
            "Built archive's external mounts differ from the R3 input binding"
        )
    if kernel_external_map(kernel_path) != expected_mounts:
        raise ValueError("Generated kernel will not mount the exact R3 flat filenames")

    expected_r9 = {item["path"]: item for item in r9_manifest["files"]}
    teacher_mounts = [
        item
        for item in input_binding["external_mounts"]
        if item["dataset"] == source_binding["teacher_cp2_asset_dataset"]
    ]
    if len(teacher_mounts) != 6:
        raise ValueError("R3 source binding must contain exactly six R9 teacher files")
    for item in teacher_mounts:
        remote = expected_r9.get(item["filename"])
        if (
            remote is None
            or remote["sha256"] != item["sha256"]
            or remote["bytes"] != item["size"]
        ):
            raise ValueError(
                f"R3 teacher mount differs from the pinned R9 manifest: {item['filename']}"
            )
    if len({item["sha256"] for item in teacher_mounts}) != 6:
        raise ValueError("R3 teacher file identities are not unique")

    dataset_sources = set(kernel_metadata["dataset_sources"])
    expected_datasets = set(source_binding["asset_datasets"]) | {
        expected_dataset,
        "vladkochetov07/wunder-funder",
    }
    if dataset_sources != expected_datasets:
        raise ValueError(
            "Generated kernel mounted dataset list differs from the reviewed inputs"
        )

    before = kernel_path.read_text()
    if before.count(OLD_DEADLINE) == 1:
        before_sha256 = sha256_file(kernel_path)
        patched = before.replace(OLD_DEADLINE, NEW_DEADLINE, 1)
        if patched.count(NEW_DEADLINE) != 1 or OLD_DEADLINE in patched:
            raise ValueError(
                "Kernel deadline replacement did not produce one exact 1,800s cap"
            )
        kernel_path.write_text(patched)
    elif (
        before.count(NEW_DEADLINE) == 1
        and OLD_DEADLINE not in before
        and sha256_file(kernel_path) == mount_successor["r3_kernel_sha256"]
        and sha256_file(BUNDLE_R2 / "kernel/kernel.py")
        == mount_successor["r2_kernel_sha256"]
    ):
        before_sha256 = mount_successor["r2_kernel_sha256"]
        patched = before
    else:
        raise ValueError(
            "Generated kernel is neither unpatched nor equal to the prior seal"
        )
    if sha256_file(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Kernel patch unexpectedly altered the source archive")

    receipt = {
        "schema": "wnn.student-shape-pair-generated-kernel-seal.r3",
        "status": "offline_bundle_and_kernel_pinned_no_provider_action",
        "provider_action_performed": False,
        "private_dataset_created": False,
        "kernel_pushed": False,
        "source_code_commit": source_binding["source_tree_commit"],
        "source_binding_sha256": sha256_file(SOURCE_BINDING),
        "input_binding_sha256": sha256_file(INPUT_BINDING),
        "recipe_contract_sha256": source_binding["recipe_contract_sha256"],
        "bundle_successor_receipt_path": str(BUILD_RECEIPT.relative_to(ROOT)),
        "bundle_successor_receipt_sha256": sha256_file(BUILD_RECEIPT),
        "kernel_mount_successor_receipt_path": str(MOUNT_SUCCESSOR.relative_to(ROOT)),
        "kernel_mount_successor_receipt_sha256": sha256_file(MOUNT_SUCCESSOR),
        "profile_jobs": source_binding["profile_jobs"],
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
        "teacher_r9_asset": {
            "dataset": source_binding["teacher_cp2_asset_dataset"],
            "version": 1,
            "manifest_path": str(R9_MANIFEST),
            "manifest_sha256": sha256_file(R9_MANIFEST),
            "mounts": [
                {
                    "logical_path": item["logical_path"],
                    "filename": item["filename"],
                    "sha256": item["sha256"],
                    "size": item["size"],
                }
                for item in teacher_mounts
            ],
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
            "r2_parent_sha256": before_sha256,
            "patched_sha256": sha256_file(kernel_path),
            "deadline_before": NEW_DEADLINE,
            "deadline_after": NEW_DEADLINE,
            "deadline_seconds": 1800,
            "patch_occurrences": 0,
            "mount_only_successor": True,
            "bootstrap_mounts_match_binding": True,
        },
    }
    output = PACKET / "GENERATED_KERNEL_SEAL_R3_SEALED.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "archive_sha256": receipt["bundle"]["archive_sha256"],
                "kernel_sha256": receipt["kernel"]["patched_sha256"],
                "manifest_sha256": receipt["bundle"]["manifest_sha256"],
                "teacher_mount_count": len(teacher_mounts),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
