from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PACKET_R2 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r2"
PACKET_R3 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r3"
BUNDLE_R2 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r2"
BUNDLE_R3 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r3"
SOURCE_BINDING = PACKET_R3 / "SOURCE_BINDING_R3.json"
INPUT_BINDING = PACKET_R3 / "INPUT_BINDING_R3.json"
BUILD_RECEIPT = PACKET_R3 / "OFFLINE_BUNDLE_BUILD_R3.json"
MOUNT_RECEIPT = PACKET_R3 / "KERNEL_MOUNT_SUCCESSOR_R3.json"


def sha256_file(path: Path) -> str:
    """Hash an artifact with bounded memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, max_bytes: int = 4_000_000) -> dict[str, Any]:
    """Read one bounded packet JSON file."""
    if path.stat().st_size > max_bytes:
        raise ValueError(f"refusing oversized JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Create a new deterministic JSON receipt without overwriting evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def external_map(binding: dict[str, Any]) -> dict[str, object]:
    """Convert the input binding to the kernel bundle's exact mount format."""
    return {
        row["logical_path"]: (
            row["filename"]
            if row["dataset"] == "vladkochetov07/wunder-funder"
            else {"dataset": row["dataset"], "filename": row["filename"]}
        )
        for row in binding["external_mounts"]
    }


def kernel_mount_map(path: Path) -> dict[str, object]:
    """Read the literal external mount map without importing the kernel."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_EXTERNAL_MOUNTS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return value
    raise ValueError("kernel has no literal EXPECTED_EXTERNAL_MOUNTS map")


def without_mount_assignment(source: str) -> str:
    """Mask the mount-map assignment so all other kernel source can be compared."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_EXTERNAL_MOUNTS"
            for target in node.targets
        ):
            lines[node.lineno - 1 : node.end_lineno] = [
                "EXPECTED_EXTERNAL_MOUNTS = <BOUND_MAP>\n"
            ]
            return "".join(lines)
    raise ValueError("kernel has no literal EXPECTED_EXTERNAL_MOUNTS assignment")


def main() -> None:
    """Reuse the sealed R2 archive and change only the two norm mount names."""
    if BUILD_RECEIPT.exists() or MOUNT_RECEIPT.exists():
        raise FileExistsError("R3 bundle receipts already exist")

    source_binding = read_json(SOURCE_BINDING)
    input_binding = read_json(INPUT_BINDING)
    r2_seal = read_json(PACKET_R2 / "GENERATED_KERNEL_SEAL_R2.json")
    r2_manifest_path = BUNDLE_R2 / "bundle-manifest.json"
    r3_manifest_path = BUNDLE_R3 / "bundle-manifest.json"
    r2_kernel_path = BUNDLE_R2 / "kernel/kernel.py"
    r3_kernel_path = BUNDLE_R3 / "kernel/kernel.py"

    if sha256_file(BUNDLE_R2 / "bundle.wnn") != r2_seal["bundle"]["archive_sha256"]:
        raise ValueError("R2 archive no longer matches its frozen kernel seal")
    if sha256_file(r2_manifest_path) != r2_seal["bundle"]["manifest_sha256"]:
        raise ValueError("R2 manifest no longer matches its frozen kernel seal")
    if sha256_file(r2_kernel_path) != r2_seal["kernel"]["patched_sha256"]:
        raise ValueError("R2 kernel no longer matches its frozen kernel seal")
    if (
        sha256_file(BUNDLE_R2 / "kernel/kernel-metadata.json")
        != r2_seal["kernel"]["kernel_metadata_sha256"]
    ):
        raise ValueError("R2 kernel metadata no longer matches its frozen seal")

    r2_manifest = read_json(r2_manifest_path)
    old_expected = r2_manifest["external_mounts"]
    new_expected = external_map(input_binding)
    r2_binding = read_json(PACKET_R2 / "INPUT_BINDING_R2.json")
    if old_expected != external_map(r2_binding):
        raise ValueError("sealed R2 bundle and R2 input binding disagree")
    if set(old_expected) != set(new_expected):
        raise ValueError("R3 mount correction changed logical paths")

    norm_dataset = source_binding["normalization_mount_correction"]["dataset"]
    changed_paths: list[str] = []
    for row in input_binding["external_mounts"]:
        previous = next(
            item
            for item in r2_binding["external_mounts"]
            if item["logical_path"] == row["logical_path"]
        )
        if row["dataset"] != previous["dataset"]:
            raise ValueError("R3 mount correction changed a dataset identity")
        if row["dataset"] == norm_dataset and row["filename"] != previous["filename"]:
            changed_paths.append(row["logical_path"])
            if (row["sha256"], row["size"]) != (
                previous["sha256"],
                previous["size"],
            ):
                raise ValueError("R3 normalization filename change altered payload")
        elif row != previous:
            raise ValueError("R3 changed a field outside two norm filenames")
    if len(changed_paths) != 2:
        raise ValueError("expected exactly two normalization filename changes")

    new_manifest = copy.deepcopy(r2_manifest)
    new_manifest["external_mounts"] = new_expected
    old_kernel = r2_kernel_path.read_text(encoding="utf-8")
    new_kernel = old_kernel
    for logical_path in changed_paths:
        previous = next(
            item
            for item in r2_binding["external_mounts"]
            if item["logical_path"] == logical_path
        )
        successor = next(
            item
            for item in input_binding["external_mounts"]
            if item["logical_path"] == logical_path
        )
        old_name = previous["filename"]
        new_name = successor["filename"]
        if new_kernel.count(old_name) != 1:
            raise ValueError(
                f"expected one kernel literal for {logical_path}, got "
                f"{new_kernel.count(old_name)}"
            )
        new_kernel = new_kernel.replace(old_name, new_name, 1)
    ast.parse(new_kernel)
    if BUNDLE_R3.exists():
        if sha256_file(BUNDLE_R3 / "bundle.wnn") != sha256_file(
            BUNDLE_R2 / "bundle.wnn"
        ):
            raise ValueError("partial R3 stage archive differs from frozen R2")
        for name in ("dataset-metadata.json", "kernel/kernel-metadata.json"):
            if sha256_file(BUNDLE_R3 / name) != sha256_file(BUNDLE_R2 / name):
                raise ValueError(f"partial R3 stage changed inherited {name}")
        if read_json(r3_manifest_path) != new_manifest:
            raise ValueError("partial R3 manifest is not the exact expected successor")
        if r3_kernel_path.read_text(encoding="utf-8") != new_kernel:
            raise ValueError("partial R3 kernel is not the exact expected successor")
    else:
        shutil.copytree(BUNDLE_R2, BUNDLE_R3)
        r3_manifest_path.write_text(
            json.dumps(new_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        r3_kernel_path.write_text(new_kernel, encoding="utf-8")
    if kernel_mount_map(r3_kernel_path) != new_expected:
        raise ValueError("R3 kernel literal map differs from the new input binding")
    if kernel_mount_map(r2_kernel_path) != old_expected:
        raise ValueError("R2 kernel literal map differs from the sealed R2 manifest")
    if new_kernel.count("KERNEL_PROCESS_WALL_SECONDS_MAX = 1800") != 1:
        raise ValueError("R3 kernel deadline is not the inherited 1,800-second cap")

    prior_manifest = copy.deepcopy(new_manifest)
    prior_manifest["external_mounts"] = old_expected
    if prior_manifest != r2_manifest:
        raise ValueError("R3 source manifest differs beyond external mount filenames")
    if without_mount_assignment(old_kernel) != without_mount_assignment(new_kernel):
        raise ValueError("R3 kernel differs beyond the two external filename literals")
    old_kernel_map = kernel_mount_map(r2_kernel_path)
    new_kernel_map = kernel_mount_map(r3_kernel_path)
    changed_kernel_paths = [
        key for key in old_kernel_map if old_kernel_map[key] != new_kernel_map[key]
    ]
    if changed_kernel_paths != changed_paths:
        raise ValueError("R3 kernel map differs outside the two norm filename fields")

    kernel_receipt = {
        "schema": "wnn.student-shape-pair-kernel-mount-successor.r3",
        "status": "outer_kernel_mount_map_successor_verified",
        "provider_action_performed": False,
        "r2_kernel_path": str(r2_kernel_path.relative_to(ROOT)),
        "r2_kernel_sha256": sha256_file(r2_kernel_path),
        "r3_kernel_path": str(r3_kernel_path.relative_to(ROOT)),
        "r3_kernel_sha256": sha256_file(r3_kernel_path),
        "r2_archive_sha256": sha256_file(BUNDLE_R2 / "bundle.wnn"),
        "r3_archive_sha256": sha256_file(BUNDLE_R3 / "bundle.wnn"),
        "outer_deadline_seconds": 1800,
        "inner_archive_and_run_labels_unchanged": True,
        "changed_fields": [
            {
                "logical_path": logical_path,
                "before": next(
                    item["filename"]
                    for item in r2_binding["external_mounts"]
                    if item["logical_path"] == logical_path
                ),
                "after": next(
                    item["filename"]
                    for item in input_binding["external_mounts"]
                    if item["logical_path"] == logical_path
                ),
            }
            for logical_path in changed_paths
        ],
    }
    write_json(MOUNT_RECEIPT, kernel_receipt)

    bundle_receipt = {
        "schema": "wnn.student-shape-pair-offline-bundle-successor.r3",
        "status": "r2_archive_reused_two_external_mount_names_corrected",
        "provider_action_performed": False,
        "reason_for_transport_only_successor": (
            "A full local rebuild was blocked because the mutable local "
            "propensity-provenance fixture differs from the frozen code-input pin; "
            "the audited R2 source archive is unchanged and contains no external "
            "teacher or normalization payload bytes."
        ),
        "r2_bundle_build_path": str(
            (PACKET_R2 / "OFFLINE_BUNDLE_BUILD_R2.json").relative_to(ROOT)
        ),
        "r2_bundle_build_sha256": sha256_file(
            PACKET_R2 / "OFFLINE_BUNDLE_BUILD_R2.json"
        ),
        "r2_kernel_seal_path": str(
            (PACKET_R2 / "GENERATED_KERNEL_SEAL_R2.json").relative_to(ROOT)
        ),
        "r2_kernel_seal_sha256": sha256_file(
            PACKET_R2 / "GENERATED_KERNEL_SEAL_R2.json"
        ),
        "r2_source_archive_sha256": r2_seal["bundle"]["archive_sha256"],
        "r3_source_archive_sha256": sha256_file(BUNDLE_R3 / "bundle.wnn"),
        "r3_source_archive_bytes": (BUNDLE_R3 / "bundle.wnn").stat().st_size,
        "archive_byte_identical_to_r2": sha256_file(BUNDLE_R3 / "bundle.wnn")
        == r2_seal["bundle"]["archive_sha256"],
        "r2_source_manifest_sha256": sha256_file(r2_manifest_path),
        "r3_source_manifest_sha256": sha256_file(r3_manifest_path),
        "manifest_changed_fields": [
            f"external_mounts.{logical_path}.filename" for logical_path in changed_paths
        ],
        "kernel_mount_receipt_path": str(MOUNT_RECEIPT.relative_to(ROOT)),
        "kernel_mount_receipt_sha256": sha256_file(MOUNT_RECEIPT),
        "r2_source_dataset_metadata_sha256": sha256_file(
            BUNDLE_R2 / "dataset-metadata.json"
        ),
        "r3_source_dataset_metadata_sha256": sha256_file(
            BUNDLE_R3 / "dataset-metadata.json"
        ),
        "r2_kernel_metadata_sha256": sha256_file(
            BUNDLE_R2 / "kernel/kernel-metadata.json"
        ),
        "r3_kernel_metadata_sha256": sha256_file(
            BUNDLE_R3 / "kernel/kernel-metadata.json"
        ),
        "manifest_mounts_match_r3_binding": new_manifest["external_mounts"]
        == new_expected,
        "kernel_mounts_match_r3_binding": kernel_mount_map(r3_kernel_path)
        == new_expected,
        "changed_mount_count": len(changed_paths),
        "changed_logical_paths": changed_paths,
        "profile_job_labels_inside_archive_unchanged": True,
        "full_bundle_rebuild_failure_receipt": str(
            (PACKET_R3 / "R3_BUNDLE_REBUILD_FAILURE_R1.json").relative_to(ROOT)
        ),
    }
    write_json(BUILD_RECEIPT, bundle_receipt)

    original_rebuild_command = source_binding["offline_bundle_command"]
    source_binding["r2_bundle_rebuild_failure_path"] = str(
        (PACKET_R3 / "R3_BUNDLE_REBUILD_FAILURE_R1.json").relative_to(ROOT)
    )
    source_binding["r2_bundle_rebuild_failure_sha256"] = sha256_file(
        PACKET_R3 / "R3_BUNDLE_REBUILD_FAILURE_R1.json"
    )
    source_binding["bundle_successor_path"] = str(
        (PACKET_R3 / "prepare_bundle_mount_successor_r3.py").relative_to(ROOT)
    )
    source_binding["bundle_successor_sha256"] = sha256_file(Path(__file__).resolve())
    source_binding["bundle_successor_command"] = [
        "/usr/bin/python3",
        str((PACKET_R3 / "prepare_bundle_mount_successor_r3.py").relative_to(ROOT)),
    ]
    source_binding["bundle_successor_receipt_path"] = str(
        BUILD_RECEIPT.relative_to(ROOT)
    )
    source_binding["bundle_successor_receipt_sha256"] = sha256_file(BUILD_RECEIPT)
    source_binding["kernel_mount_successor_receipt_path"] = str(
        MOUNT_RECEIPT.relative_to(ROOT)
    )
    source_binding["kernel_mount_successor_receipt_sha256"] = sha256_file(MOUNT_RECEIPT)
    source_binding["offline_bundle_command"] = source_binding[
        "bundle_successor_command"
    ]
    source_binding["r2_full_bundle_rebuild"] = {
        "command": original_rebuild_command,
        "failure_path": source_binding["r2_bundle_rebuild_failure_path"],
        "failure_sha256": source_binding["r2_bundle_rebuild_failure_sha256"],
        "not_retried": True,
    }
    SOURCE_BINDING.write_text(
        json.dumps(source_binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": bundle_receipt["status"],
                "archive_sha256": bundle_receipt["r3_source_archive_sha256"],
                "manifest_sha256": bundle_receipt["r3_source_manifest_sha256"],
                "kernel_sha256": kernel_receipt["r3_kernel_sha256"],
                "changed_logical_paths": changed_paths,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
