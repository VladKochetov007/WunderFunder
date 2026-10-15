from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PACKET_R4 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r4"
PACKET_R5 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r5"
BUNDLE_R4 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r4"
BUNDLE_R5 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r5"
FIXTURE = ROOT / "private_assets/bootstrap_fixture_student_shape_t4_r5"
SOURCE_DATASET = "vladkochetov07/wnn-f0-student-shape-t4-profile-20261015-r1"
KERNEL_ID = f"{SOURCE_DATASET}-runner"
TARGET_PATHS = ("cache/train_targets.npz", "cache/valid_targets.npz")
OUTPUT_NAMES = (
    "R5_BUNDLE_BUILD.json",
    "R5_BOOTSTRAP_FIXTURE.json",
    "SOURCE_BINDING_R5.json",
    "GENERATED_KERNEL_SEAL_R5.json",
    "OFFLINE_PREFLIGHT_R5.json",
)


def sha256_file(path: Path) -> str:
    """Hash small packet files or the 272 KB source archive in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, maximum: int = 1_000_000) -> dict[str, Any]:
    """Read bounded metadata only."""
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError(f"refusing oversized or empty JSON: {path} ({size} bytes)")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json_new(path: Path, value: object) -> None:
    """Write a deterministic receipt without overwriting earlier evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite R5 artifact: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def link_exact(source: Path, destination: Path) -> None:
    """Hardlink one immutable small staging input without changing its source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    os.link(source, destination)


def input_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    """Reduce the frozen portable input row to the manifest.files schema."""
    return {
        "path": row["logical_path"],
        "sha256": row["sha256"],
        "size": int(row["size"]),
    }


def source_map(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Normalize external mappings to exact dataset and filename pairs."""
    result = {}
    for logical, mapping in manifest["external_mounts"].items():
        if isinstance(mapping, str):
            result[logical] = {"dataset": "vladkochetov07/wunder-funder", "filename": mapping}
        elif isinstance(mapping, dict) and set(mapping) == {"dataset", "filename"}:
            result[logical] = dict(mapping)
        else:
            raise ValueError(f"unsupported external mount mapping for {logical}")
    return result


def prepare_r5_bundle(script_sha256: str) -> dict[str, Any]:
    """Add the two bound external target descriptors; reuse the R4 archive bytes."""
    if BUNDLE_R5.exists():
        raise FileExistsError(f"R5 upload stage already exists: {BUNDLE_R5}")
    if any((PACKET_R5 / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError("R5 receipts already exist; preserve them")

    r4_seal = read_json(PACKET_R4 / "GENERATED_KERNEL_SEAL_R4.json")
    r4_source = read_json(PACKET_R4 / "SOURCE_BINDING_R4.json")
    r4_binding = read_json(PACKET_R4 / "INPUT_BINDING_R4.json")
    r4_terminal = read_json(PACKET_R4 / "PROVIDER_TERMINAL_R4.json")
    r4_manifest = read_json(BUNDLE_R4 / "bundle-manifest.json", maximum=100_000)
    archive_r4 = BUNDLE_R4 / "bundle.wnn"
    kernel_r4 = BUNDLE_R4 / "kernel/kernel.py"

    if r4_terminal.get("kernel", {}).get("version") != 2 or r4_terminal.get("status") != "terminal_bootstrap_infrastructure_failure_before_runner":
        raise ValueError("R5 must bind the exact preserved R4 version-2 bootstrap failure")
    if sha256_file(archive_r4) != r4_seal["bundle"]["archive_sha256"]:
        raise ValueError("R4 archive does not match its sealed identity")
    if sha256_file(kernel_r4) != r4_seal["kernel"]["sha256"]:
        raise ValueError("R4 bootstrap does not match its sealed identity")
    if sha256_file(BUNDLE_R4 / "bundle-manifest.json") != r4_seal["bundle"]["manifest_sha256"]:
        raise ValueError("R4 manifest does not match its sealed identity")
    if sha256_file(PACKET_R4 / "INPUT_BINDING_R4.json") != r4_source["input_binding_sha256"]:
        raise ValueError("R4 input binding differs from its source binding")

    target_rows = sorted(
        (row for row in r4_binding["external_mounts"] if row["logical_path"] in TARGET_PATHS),
        key=lambda row: row["logical_path"],
    )
    if tuple(row["logical_path"] for row in target_rows) != tuple(sorted(TARGET_PATHS)):
        raise ValueError("R4 input binding does not contain exactly the two target cache paths")
    prior_external = source_map(r4_manifest)
    existing_descriptors = {row["path"]: row for row in r4_manifest["files"]}
    additions = [input_descriptor(row) for row in target_rows]
    for row in target_rows:
        logical = row["logical_path"]
        if logical in existing_descriptors:
            raise ValueError(f"R4 manifest already declares the external-only path {logical}")
        descriptor = input_descriptor(row)
        if prior_external.get(logical) != {"dataset": row["dataset"], "filename": row["filename"]}:
            raise ValueError(f"R4 manifest external mount differs from its input binding: {logical}")
        if descriptor["size"] <= 0 or len(descriptor["sha256"]) != 64:
            raise ValueError(f"invalid producer size/hash for {logical}")

    manifest = dict(r4_manifest)
    manifest["files"] = [*r4_manifest["files"], *additions]
    if len({row["path"] for row in manifest["files"]}) != len(manifest["files"]):
        raise ValueError("R5 manifest.files contains duplicate paths")
    if [row["path"] for row in manifest["files"] if row["path"] in TARGET_PATHS] != list(sorted(TARGET_PATHS)):
        raise ValueError("R5 target descriptors are not the exact pair in stable path order")
    if manifest["archive_sha256"] != r4_manifest["archive_sha256"] or manifest["archive_bytes"] != r4_manifest["archive_bytes"]:
        raise ValueError("R5 unexpectedly changed the frozen source archive identity")
    if manifest["expanded_bytes"] != r4_manifest["expanded_bytes"]:
        raise ValueError("R5 changed archive expansion accounting for external-only descriptors")

    BUNDLE_R5.mkdir(parents=True)
    for relative in (
        "bundle.wnn",
        "dataset-metadata.json",
        "kernel/kernel-metadata.json",
        "kernel/kernel.py",
    ):
        link_exact(BUNDLE_R4 / relative, BUNDLE_R5 / relative)
    manifest_path = BUNDLE_R5 / "bundle-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    r5_archive_sha = sha256_file(BUNDLE_R5 / "bundle.wnn")
    r5_kernel_sha = sha256_file(BUNDLE_R5 / "kernel/kernel.py")
    r5_manifest_sha = sha256_file(manifest_path)
    if r5_archive_sha != r4_seal["bundle"]["archive_sha256"]:
        raise ValueError("R5 archive hardlink identity differs from frozen R4")
    if r5_kernel_sha != r4_seal["kernel"]["sha256"]:
        raise ValueError("R5 kernel hardlink identity differs from frozen R4")

    build = {
        "schema": "wnn.student-shape-pair-r5-bundle-build.r1",
        "status": "two_manifest_descriptors_added_archive_and_kernel_reused",
        "provider_action_performed": False,
        "builder_script_sha256": script_sha256,
        "r4_terminal_path": "research/campaign_2026-10-15/student_shape_pair_f0_r4/PROVIDER_TERMINAL_R4.json",
        "r4_terminal_sha256": sha256_file(PACKET_R4 / "PROVIDER_TERMINAL_R4.json"),
        "r4_bundle_manifest_sha256": sha256_file(BUNDLE_R4 / "bundle-manifest.json"),
        "r5_bundle_manifest_path": "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r5/bundle-manifest.json",
        "r5_bundle_manifest_sha256": r5_manifest_sha,
        "r4_manifest_file_count": len(r4_manifest["files"]),
        "r5_manifest_file_count": len(manifest["files"]),
        "only_manifest_files_added": additions,
        "external_mount_map_unchanged": manifest["external_mounts"] == r4_manifest["external_mounts"],
        "archive_sha256_unchanged": r5_archive_sha,
        "archive_bytes_unchanged": (BUNDLE_R5 / "bundle.wnn").stat().st_size,
        "kernel_sha256_unchanged": r5_kernel_sha,
        "expanded_bytes_unchanged": manifest["expanded_bytes"],
        "target_payloads_copied_or_read": False,
        "source_dataset": SOURCE_DATASET,
        "current_source_dataset_version_before_update": 2,
        "expected_source_dataset_version_after_update": 3,
        "expected_kernel_version_after_update": 3,
    }
    write_json_new(PACKET_R5 / OUTPUT_NAMES[0], build)
    return {
        "r4_seal": r4_seal,
        "r4_source": r4_source,
        "r4_binding": r4_binding,
        "r4_manifest": r4_manifest,
        "r5_manifest": manifest,
        "target_rows": target_rows,
        "additions": additions,
        "archive_sha256": r5_archive_sha,
        "kernel_sha256": r5_kernel_sha,
        "manifest_sha256": r5_manifest_sha,
        "build_sha256": sha256_file(PACKET_R5 / OUTPUT_NAMES[0]),
    }


def run_actual_bootstrap_fixture(prepared: dict[str, Any], script_sha256: str) -> dict[str, Any]:
    """Execute the frozen outer kernel on fake physical mounts and intercept its runner."""
    if FIXTURE.exists():
        raise FileExistsError(f"R5 fixture path already exists: {FIXTURE}")
    input_root = FIXTURE / "kaggle/input"
    working_root = FIXTURE / "kaggle/working"
    input_root.mkdir(parents=True)
    working_root.mkdir(parents=True)

    bundle_input = input_root / SOURCE_DATASET.split("/", 1)[1]
    bundle_input.mkdir()
    link_exact(BUNDLE_R5 / "bundle.wnn", bundle_input / "bundle.wnn")
    shutil.copyfile(BUNDLE_R5 / "bundle-manifest.json", bundle_input / "bundle-manifest.json")

    stubs: dict[tuple[str, str], Path] = {}
    for logical, mapping in prepared["r5_manifest"]["external_mounts"].items():
        if isinstance(mapping, str):
            dataset, filename = "vladkochetov07/wunder-funder", mapping
        else:
            dataset, filename = mapping["dataset"], mapping["filename"]
        slug = dataset.split("/", 1)[1]
        candidate = input_root / slug / filename
        candidate.parent.mkdir(parents=True, exist_ok=True)
        identity = (slug, filename)
        prior = stubs.get(identity)
        if prior is None:
            if candidate.exists():
                raise FileExistsError(f"external stub path collides with fixture bundle: {candidate}")
            candidate.write_bytes(b"bootstrap-only external input stub\n")
            stubs[identity] = candidate
        elif prior != candidate:
            raise ValueError(f"duplicate external basename has conflicting fake mount path: {logical}")

    # Change only the four physical Kaggle path literals; execute the actual sealed
    # R4 outer bootstrap, actual R5 manifest and actual source archive below.
    kernel_path = BUNDLE_R5 / "kernel/kernel.py"
    source = kernel_path.read_text(encoding="utf-8")
    replacements = (
        ('Path("/kaggle/input")', f"Path({str(input_root)!r})", 2),
        ('Path("/kaggle/working/experiment")', f"Path({str(working_root / 'experiment')!r})", 1),
        ('"/kaggle/working/results"', repr(str(working_root / "results")), 1),
        ('Path("/kaggle/working/runtime_specs")', f"Path({str(working_root / 'runtime_specs')!r})", 1),
    )
    patched_source = source
    for old, new, expected_count in replacements:
        if patched_source.count(old) != expected_count:
            raise ValueError(f"expected {expected_count} physical path literal(s): {old}")
        patched_source = patched_source.replace(old, new)
    ast.parse(patched_source)

    launch_receipt: dict[str, Any] = {}

    class InterceptedProcess:
        """Return success without importing or running the trainer or accelerator code."""

        def wait(self, timeout: float | None = None) -> int:
            launch_receipt["wait_timeout_seconds"] = timeout
            return 0

    real_popen = subprocess.Popen

    def intercept_popen(command: list[str], **kwargs: Any) -> InterceptedProcess:
        if launch_receipt:
            raise RuntimeError("outer bootstrap attempted more than one child launch")
        cwd = Path(kwargs["cwd"])
        extracted_root = working_root / "experiment"
        if cwd != extracted_root or not (cwd / "scripts/kaggle_gpu_runner.py").is_file():
            raise ValueError("outer bootstrap child command does not target the extracted runner")
        expected_mounts = prepared["r5_manifest"]["external_mounts"]
        mounted = {}
        for logical, mapping in expected_mounts.items():
            if isinstance(mapping, str):
                dataset, filename = "vladkochetov07/wunder-funder", mapping
            else:
                dataset, filename = mapping["dataset"], mapping["filename"]
            expected_input = input_root / dataset.split("/", 1)[1] / filename
            target = extracted_root / logical
            if not target.is_symlink():
                raise ValueError(f"external path was not symlink-mounted before runner launch: {logical}")
            if target.resolve(strict=True) != expected_input.resolve(strict=True):
                raise ValueError(f"external path resolved to the wrong fake input: {logical}")
            mounted[logical] = str(target.resolve(strict=True))
        for logical in TARGET_PATHS:
            if logical not in mounted:
                raise ValueError(f"target cache path absent from mounted inputs: {logical}")
        launch_receipt.update(
            {
                "command": list(command),
                "cwd": str(cwd),
                "mounted_external_paths": sorted(mounted),
                "target_cache_links_present_before_child": True,
                "external_input_symlinks_at_launch": mounted,
            }
        )
        return InterceptedProcess()

    if "torch" in sys.modules:
        raise RuntimeError("bootstrap fixture started with torch already imported")
    subprocess.Popen = intercept_popen  # type: ignore[assignment]
    try:
        namespace = {"__name__": "__main__", "__file__": str(kernel_path)}
        exec(compile(patched_source, str(kernel_path), "exec"), namespace, namespace)
    finally:
        subprocess.Popen = real_popen

    if not launch_receipt:
        raise RuntimeError("actual outer bootstrap did not reach the intercepted runner")
    extracted_root = working_root / "experiment"
    remaining_links = [logical for logical in prepared["r5_manifest"]["external_mounts"] if (extracted_root / logical).is_symlink()]
    if remaining_links:
        raise ValueError(f"outer bootstrap failed to clean external links: {remaining_links[:3]}")
    if "torch" in sys.modules:
        raise RuntimeError("outer bootstrap fixture imported torch")

    with tarfile.open(BUNDLE_R5 / "bundle.wnn", "r:gz") as archive:
        archive_member_count = len(archive.getmembers())
    declared_paths = {row["path"] for row in prepared["r5_manifest"]["files"]}
    descriptor_coverage = {row["logical_path"] for row in prepared["target_rows"]} <= declared_paths
    if not descriptor_coverage:
        raise ValueError("R5 external target descriptors are absent from manifest.files")

    return {
        "schema": "wnn.student-shape-pair-r5-actual-bootstrap-fixture.r1",
        "status": "passed_actual_outer_bootstrap_with_runner_intercepted",
        "provider_action_performed": False,
        "fixture_script_sha256": script_sha256,
        "frozen_kernel_sha256": prepared["kernel_sha256"],
        "r5_archive_sha256": prepared["archive_sha256"],
        "r5_manifest_sha256": prepared["manifest_sha256"],
        "real_archive_member_count": archive_member_count,
        "manifest_files_descriptor_count": len(prepared["r5_manifest"]["files"]),
        "external_mount_count": len(prepared["r5_manifest"]["external_mounts"]),
        "all_external_paths_declared_by_manifest_files": descriptor_coverage,
        "two_target_cache_descriptors": prepared["additions"],
        "target_cache_symlinks_present_at_intercept": True,
        "runner_popen_intercepted_before_trainer_or_gpu_code": True,
        "torch_imported": False,
        "external_symlinks_cleaned_after_intercept": not remaining_links,
        "stubbed_external_files": len(stubs),
        "launch": launch_receipt,
        "qualification": "This exercises the real R4 outer bootstrap and R5 manifest/archive against small local mount stubs. The runner is intercepted before training, so it is not a data-content, GPU, profile, fit, or quality test.",
    }


def main() -> None:
    """Prepare the R5 manifest-only repair and exercise the real bootstrap boundary."""
    script_sha256 = sha256_file(Path(__file__).resolve())
    prepared = prepare_r5_bundle(script_sha256)
    fixture = run_actual_bootstrap_fixture(prepared, script_sha256)
    write_json_new(PACKET_R5 / OUTPUT_NAMES[1], fixture)

    source = {
        "schema": "wnn.student-shape-pair-source-binding.r5",
        "status": "manifest_descriptor_repair_and_actual_bootstrap_fixture_passed",
        "provider_action_performed": False,
        "builder_and_fixture_script_path": "research/campaign_2026-10-15/student_shape_pair_f0_r5/prepare_and_validate_r5.py",
        "builder_and_fixture_script_sha256": script_sha256,
        "r4_source_binding_path": "research/campaign_2026-10-15/student_shape_pair_f0_r4/SOURCE_BINDING_R4.json",
        "r4_source_binding_sha256": sha256_file(PACKET_R4 / "SOURCE_BINDING_R4.json"),
        "r4_terminal_receipt_path": "research/campaign_2026-10-15/student_shape_pair_f0_r4/PROVIDER_TERMINAL_R4.json",
        "r4_terminal_receipt_sha256": sha256_file(PACKET_R4 / "PROVIDER_TERMINAL_R4.json"),
        "r4_kernel_push_preflight_sha256": sha256_file(PACKET_R4 / "PROVIDER_PREFLIGHT_KERNEL_PUSH_R4.json"),
        "input_binding_path": "research/campaign_2026-10-15/student_shape_pair_f0_r4/INPUT_BINDING_R4.json",
        "input_binding_sha256": sha256_file(PACKET_R4 / "INPUT_BINDING_R4.json"),
        "bundle_build_path": "research/campaign_2026-10-15/student_shape_pair_f0_r5/R5_BUNDLE_BUILD.json",
        "bundle_build_sha256": sha256_file(PACKET_R5 / OUTPUT_NAMES[0]),
        "bootstrap_fixture_path": "research/campaign_2026-10-15/student_shape_pair_f0_r5/R5_BOOTSTRAP_FIXTURE.json",
        "bootstrap_fixture_sha256": sha256_file(PACKET_R5 / OUTPUT_NAMES[1]),
        "profile_contract_sha256": prepared["r4_source"]["recipe_contract_sha256"],
        "profile_job_source_pins_unchanged": [
            {
                "name": row["name"],
                "path": row["path"],
                "sha256": row["sha256"],
                "shape": row["shape"],
                "optimizer_attempts_expected": row["optimizer_attempts_expected"],
            }
            for row in prepared["r4_source"]["profile_jobs"]
        ],
        "private_source_dataset": SOURCE_DATASET,
        "current_version_before_update": 2,
        "version_required_for_future_push": 3,
        "kernel_id": KERNEL_ID,
        "current_kernel_version_before_update": 2,
        "kernel_version_required_for_future_push": 3,
        "r5_archive_sha256": prepared["archive_sha256"],
        "r5_archive_bytes": (BUNDLE_R5 / "bundle.wnn").stat().st_size,
        "r5_manifest_sha256": prepared["manifest_sha256"],
        "r5_kernel_sha256": prepared["kernel_sha256"],
        "manifest_file_descriptor_additions": prepared["additions"],
        "scientific_inputs_and_profile_recipe_changed": False,
    }
    write_json_new(PACKET_R5 / OUTPUT_NAMES[2], source)

    seal = {
        "schema": "wnn.student-shape-pair-generated-kernel-seal.r5",
        "status": "offline_r5_manifest_and_bootstrap_consumer_fixture_passed_no_provider_action",
        "provider_action_performed": False,
        "source_binding_sha256": sha256_file(PACKET_R5 / OUTPUT_NAMES[2]),
        "input_binding_sha256": source["input_binding_sha256"],
        "r4_parent_seal_sha256": sha256_file(PACKET_R4 / "GENERATED_KERNEL_SEAL_R4.json"),
        "r4_terminal_receipt_sha256": source["r4_terminal_receipt_sha256"],
        "bundle": {
            "dataset": SOURCE_DATASET,
            "version_required": 3,
            "archive_bytes": (BUNDLE_R5 / "bundle.wnn").stat().st_size,
            "archive_sha256": prepared["archive_sha256"],
            "manifest_bytes": (BUNDLE_R5 / "bundle-manifest.json").stat().st_size,
            "manifest_sha256": prepared["manifest_sha256"],
            "manifest_files_count": len(prepared["r5_manifest"]["files"]),
            "external_mount_count": len(prepared["r5_manifest"]["external_mounts"]),
            "added_external_file_descriptors": prepared["additions"],
        },
        "kernel": {
            "id": KERNEL_ID,
            "version_required": 3,
            "sha256": prepared["kernel_sha256"],
            "unchanged_from_r4": True,
            "outer_deadline_seconds": 1800,
        },
        "profile_jobs": source["profile_job_source_pins_unchanged"],
        "bootstrap_fixture_sha256": source["bootstrap_fixture_sha256"],
    }
    write_json_new(PACKET_R5 / OUTPUT_NAMES[3], seal)

    offline = {
        "schema": "wnn.student-shape-pair-offline-preflight.r5",
        "status": "passed_actual_bootstrap_fixture_no_provider_action",
        "provider_action_performed": False,
        "source_binding_sha256": sha256_file(PACKET_R5 / OUTPUT_NAMES[2]),
        "seal_sha256": sha256_file(PACKET_R5 / OUTPUT_NAMES[3]),
        "builder_and_fixture_script_sha256": script_sha256,
        "checks": [
            "R4 version-2 terminal failure and source pins are preserved and hash-bound",
            "R5 manifest.files differs from R4 only by the two fold0 target-cache file descriptors taken from the frozen producer-bound input binding",
            "R5 private source archive bytes, external mount map, profile job specs, outer kernel code and 1,800-second cap are unchanged from R4",
            "the exact R4 outer bootstrap code was executed with only four Kaggle filesystem root literals redirected to a physical fixture",
            "the real R5 manifest and real 272 KB archive passed bundle discovery, archive hash, safe extraction, external mount validation and runtime spec creation",
            "all declared external symlinks, including both target caches, existed before the intercepted runner launch and were removed afterward",
            "the runner child was intercepted before trainer imports; no provider action, GPU allocation, fit or quality evaluation occurred",
        ],
        "bootstrap_fixture": fixture,
        "provider_action_performed": False,
    }
    write_json_new(PACKET_R5 / OUTPUT_NAMES[4], offline)
    print(
        json.dumps(
            {
                "status": offline["status"],
                "r4_terminal_error": read_json(PACKET_R4 / "PROVIDER_TERMINAL_R4.json")["terminal"]["first_error"],
                "manifest_descriptor_count": len(prepared["r5_manifest"]["files"]),
                "external_mount_count": len(prepared["r5_manifest"]["external_mounts"]),
                "archive_sha256_unchanged": prepared["archive_sha256"] == prepared["r4_seal"]["bundle"]["archive_sha256"],
                "kernel_sha256_unchanged": prepared["kernel_sha256"] == prepared["r4_seal"]["kernel"]["sha256"],
                "fixture_runner_intercepted": fixture["runner_popen_intercepted_before_trainer_or_gpu_code"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
