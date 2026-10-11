"""Create-only R8 finalizer for the root-authorized R5 teacher cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKET))
import build_teacher_cohort_r5 as builder  # noqa: E402

ROOT_RECEIPT_RELATIVE = f"{builder.R5_PACKET}/ROOT_CONDITIONAL_RELEASE_R5.json"
GATE_RELATIVE = f"{builder.R5_PACKET}/CONDITIONAL_FIT_ADMISSION_R5.json"
PRESEAL_RELATIVE = f"{builder.R5_PACKET}/R5_PRESEAL_BINDINGS_R1.json"
KERNEL_STAGE_RELATIVE = Path("bundle_teacher_cohort_r5/kernel/kernel.py")
EXPECTED_ROOT_RECEIPT_SHA256 = "7ba5b10cbab68365334941e18d2d28f217d45961eb5e40f4ac5d8d8de90c290e"
EXPECTED_R7_ARCHIVE_SHA256 = "1cf431474a7aed41a8df6606373c63a4d5816bdf0c77b4deff0c72dfefce2bca"


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_quota_number(value: Any, label: str) -> None:
    if type(value) is int:
        valid = value >= 28_800
    elif type(value) is float:
        valid = math.isfinite(value) and value >= 28_800
    else:
        valid = False
    if not valid:
        raise ValueError(f"root receipt has invalid {label}")


def verify_root_authority(receipt_bytes: bytes, expected_sha: str, preseal: dict[str, Any]) -> dict:
    actual_sha = sha_bytes(receipt_bytes)
    if expected_sha != EXPECTED_ROOT_RECEIPT_SHA256 or actual_sha != expected_sha:
        raise ValueError("root receipt bytes do not match the separately supplied expected SHA")
    receipt = json.loads(receipt_bytes)
    if (
        receipt.get("schema") != "wnn.root_conditional_teacher_cohort_release.r5"
        or receipt.get("status") != "admitted"
        or receipt.get("authorized") is not True
        or receipt.get("binding", {}).get("preseal_r5") != preseal
        or receipt.get("target") != {"dataset": builder.R5_DATASET, "kernel_id": builder.R5_KERNEL}
        or receipt.get("allowed_delta") != builder.R5_ALLOWED_DELTA
    ):
        raise ValueError("root authority does not bind the exact R5 preseal and target")
    quota = receipt.get("quota_budget", {})
    for key, expected in (
        ("wall_ceiling_seconds", 14_400),
        ("two_device_gpu_seconds_upper_bound", 28_800),
        ("reserved_gpu_seconds", 28_800),
    ):
        if type(quota.get(key)) is not int or quota[key] != expected:
            raise ValueError(f"root quota has invalid {key}")
    if quota.get("paid_spending_authorized") is not False:
        raise ValueError("root receipt must not authorize paid spending")
    require_quota_number(quota.get("campaign_headroom_gpu_seconds"), "campaign headroom")
    require_quota_number(quota.get("provider_headroom_gpu_seconds"), "provider headroom")
    return receipt


def replace_constant(source: bytes, name: str, value: str) -> bytes:
    text = source.decode("utf-8")
    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=\s*(['\"])[0-9a-f]{{64}}\1$")
    text, count = pattern.subn(f'{name} = "{value}"', text)
    if count != 1:
        raise ValueError(f"expected one frozen bootstrap constant {name}, found {count}")
    return text.encode("utf-8")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finalize(r7: Path, root_receipt_path: Path, expected_sha: str, output: Path) -> dict:
    r7 = r7.resolve(strict=True)
    root_receipt_path = root_receipt_path.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite identified R8 output: {output}")
    if r7.name != "package_r7":
        raise ValueError("input must be the immutable R7 package directory")
    manifest_path = r7 / "bundle_teacher_cohort_r5/bundle-manifest.json"
    archive_path = r7 / "bundle_teacher_cohort_r5/bundle.wnn"
    manifest = read_json(manifest_path)
    if sha_bytes(archive_path.read_bytes()) != EXPECTED_R7_ARCHIVE_SHA256:
        raise ValueError("R7 archive is not the independently reviewed input")
    preseal = read_json(r7 / "R5_PRESEAL_BINDINGS_R1.json")
    root_bytes = root_receipt_path.read_bytes()
    root_receipt = verify_root_authority(root_bytes, expected_sha, preseal)
    if (PACKET / "R5_FINALIZATION_RECEIPT_R8.json").exists():
        raise FileExistsError(
            "R8 finalization receipt already exists; use a new identified successor"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".r5-r8-finalize-", dir=output.parent) as temporary:
        temp_output = Path(temporary) / "package_r8"
        shutil.copytree(r7, temp_output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (temp_output / "ROOT_CONDITIONAL_RELEASE_R5.json").write_bytes(root_bytes)

        gate_path = temp_output / "CONDITIONAL_FIT_ADMISSION_R5.json"
        gate = read_json(gate_path)
        gate.update(
            {
                "status": "admitted",
                "release_authorized": True,
                "root_release_receipt_path": ROOT_RECEIPT_RELATIVE,
                "root_release_receipt_sha256": expected_sha,
                "note": "Root authorizes only the same-session paired-profile path; teacher fits remain conditional on actual profile, memory/cache handoff, and unchanged strict two-worker admission passing.",
            }
        )
        write_json(gate_path, gate)
        gate_sha = sha_bytes(gate_path.read_bytes())

        bundle_dir = temp_output / "bundle_teacher_cohort_r5"
        archive_path = bundle_dir / "bundle.wnn"
        files = builder.read_tar(archive_path)
        if not {ROOT_RECEIPT_RELATIVE, GATE_RELATIVE} <= set(files):
            raise ValueError("R7 archive is missing a required root/gate member")
        files[ROOT_RECEIPT_RELATIVE] = root_bytes
        files[GATE_RELATIVE] = gate_path.read_bytes()
        builder.write_tar(files, archive_path)
        archive_sha = sha_bytes(archive_path.read_bytes())
        kernel_path = temp_output / KERNEL_STAGE_RELATIVE
        kernel = kernel_path.read_bytes()
        kernel = replace_constant(kernel, "EXPECTED_ARCHIVE_SHA256", archive_sha)
        kernel = replace_constant(kernel, "R5_GATE_SHA256", gate_sha)
        kernel = replace_constant(kernel, "R5_ROOT_RELEASE_RECEIPT_SHA256", expected_sha)
        kernel_path.write_bytes(kernel)

        manifest_path = bundle_dir / "bundle-manifest.json"
        manifest = read_json(manifest_path)
        for row in manifest["files"]:
            relative = row["path"]
            if relative in files:
                payload = files[relative]
                row["size"] = len(payload)
                row["sha256"] = sha_bytes(payload)
        manifest["expanded_bytes"] = sum(len(payload) for payload in files.values())
        manifest["archive_bytes"] = archive_path.stat().st_size
        manifest["archive_sha256"] = archive_sha
        write_json(manifest_path, manifest)

        final_sha = manifest["archive_sha256"]
        receipt = {
            "schema": "wnn.kaggle.teacher_cohort_r5_finalization.r8",
            "status": "sealed_pending_independent_final_review_and_dispatch_admission",
            "authorized_action": "local package finalization only; no Kaggle create/push or fit",
            "root_release_receipt_sha256": expected_sha,
            "root_release_schema": root_receipt["schema"],
            "input_r7": {
                "archive_sha256": EXPECTED_R7_ARCHIVE_SHA256,
                "manifest_sha256": sha_bytes(
                    (r7 / "bundle_teacher_cohort_r5/bundle-manifest.json").read_bytes()
                ),
                "bootstrap_sha256": sha_bytes(
                    (r7 / "bundle_teacher_cohort_r5/kernel/kernel.py").read_bytes()
                ),
            },
            "output_r8": {
                "archive_sha256": final_sha,
                "archive_bytes": manifest["archive_bytes"],
                "manifest_sha256": sha_bytes(manifest_path.read_bytes()),
                "bootstrap_sha256": sha_bytes(kernel_path.read_bytes()),
                "gate_sha256": gate_sha,
                "root_receipt_sha256": sha_bytes(
                    (temp_output / "ROOT_CONDITIONAL_RELEASE_R5.json").read_bytes()
                ),
                "root_authorized": True,
                "root_release_target": root_receipt["target"],
                "fit_still_conditional_on_same_session_profile_and_strict_resource_guard": True,
            },
            "historical_readme_note": "README_R5.md is preserved byte-for-byte from R7 and describes R7 preparation status; this finalization receipt supersedes that pending-state text for the R8 bytes.",
            "provider_actions": 0,
            "science_or_resource_contract_changed": False,
        }
        write_json(temp_output / "R5_FINALIZATION_RECEIPT_R8.json", receipt)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.rename(temp_output, output)
    return receipt


def validate_output(package: Path, fixture_dir: Path, expected_sha: str) -> dict:
    package = package.resolve(strict=True)
    receipt_path = package / "R5_FINALIZATION_RECEIPT_R8.json"
    finalization = read_json(receipt_path)
    root_bytes = (package / "ROOT_CONDITIONAL_RELEASE_R5.json").read_bytes()
    root_sha = sha_bytes(root_bytes)
    preseal = read_json(package / "R5_PRESEAL_BINDINGS_R1.json")
    root = verify_root_authority(root_bytes, expected_sha, preseal)
    gate_path = package / "CONDITIONAL_FIT_ADMISSION_R5.json"
    gate = read_json(gate_path)
    if (
        gate.get("status") != "admitted"
        or gate.get("release_authorized") is not True
        or gate.get("root_release_receipt_path") != ROOT_RECEIPT_RELATIVE
        or gate.get("root_release_receipt_sha256") != root_sha
    ):
        raise ValueError("R8 conditional gate does not bind its exact root receipt")
    bundle = package / "bundle_teacher_cohort_r5"
    manifest_path = bundle / "bundle-manifest.json"
    manifest = read_json(manifest_path)
    archive = bundle / "bundle.wnn"
    archive_sha = sha_bytes(archive.read_bytes())
    if archive_sha != manifest.get("archive_sha256") or archive.stat().st_size != manifest.get(
        "archive_bytes"
    ):
        raise ValueError("R8 archive bytes do not match the manifest")
    files = builder.read_tar(archive)
    manifest_files = {row["path"]: row for row in manifest.get("files", [])}
    if set(files) != set(manifest_files):
        raise ValueError("R8 archive members do not exactly match the manifest inventory")
    for name, payload in files.items():
        row = manifest_files[name]
        if row.get("size") != len(payload) or row.get("sha256") != sha_bytes(payload):
            raise ValueError(f"R8 archive member differs from its manifest binding: {name}")
    if (
        files.get(ROOT_RECEIPT_RELATIVE) != root_bytes
        or files.get(GATE_RELATIVE) != gate_path.read_bytes()
    ):
        raise ValueError("R8 archive does not carry the exact gate and root receipt bytes")
    bootstrap = (bundle / "kernel/kernel.py").read_bytes()
    for name, value in (
        ("EXPECTED_ARCHIVE_SHA256", archive_sha),
        ("R5_GATE_SHA256", sha_bytes(gate_path.read_bytes())),
        ("R5_ROOT_RELEASE_RECEIPT_SHA256", root_sha),
    ):
        match = re.search(rf"(?m)^{name}\s*=\s*['\"]([0-9a-f]{{64}})['\"]$", bootstrap.decode())
        if match is None or match.group(1) != value:
            raise ValueError(f"R8 bootstrap does not pin the exact {name}")
    kernel_metadata = read_json(bundle / "kernel/kernel-metadata.json")
    if (
        kernel_metadata.get("id") != builder.R5_KERNEL
        or kernel_metadata.get("title") != f"{builder.R5_DATASET.rsplit('/', 1)[-1]} runner"
        or kernel_metadata.get("dataset_sources", [None])[0] != builder.R5_DATASET
    ):
        raise ValueError("R8 kernel metadata does not bind the exact title-derived target")
    driver_rel = f"{builder.R5_PACKET}/kernel_support/teacher_cohort_r5_driver.py"
    driver_path = package / "kernel_support/teacher_cohort_r5_driver.py"
    if files.get(driver_rel) != driver_path.read_bytes():
        raise ValueError("R8 driver copy differs from the encoded source")
    driver = types.ModuleType("r8_actual_driver")
    exec(compile(driver_path.read_bytes(), str(driver_path), "exec"), driver.__dict__)
    expected_names = set(driver.PROFILE_NAMES + driver.FIT_NAMES + driver.EXPORT_NAMES)
    spec_names = {Path(item).stem for item in manifest["spec_paths"]}
    if len(expected_names) != 14 or spec_names != expected_names:
        raise ValueError("R8 archive spec inventory does not close over driver names")
    for item in manifest["spec_paths"]:
        spec_value = json.loads(files[item])
        if Path(item).name != f"{spec_value.get('name')}.json":
            raise ValueError(f"R8 spec filename/name mismatch: {item}")
        if preseal["worker_specs_sha256"].get(spec_value["name"]) != sha_bytes(files[item]):
            raise ValueError(f"R8 worker spec is not pinned by preseal: {item}")
    validator_path = PACKET / "validate_teacher_cohort_r5.py"
    spec = importlib.util.spec_from_file_location("r8_actual_archive_validator", validator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the reviewed actual-archive consumer check")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    preflight = json.loads((fixture_dir / "R4_profile_preflight_inventory.json").read_text())
    consumer = validator.validate_archive_consumer(files, preflight, manifest)
    if (
        consumer.get("status") != "pass_actual_archive_consumer"
        or consumer.get("gpu_or_provider_calls") != 0
    ):
        raise ValueError("actual R8 archive consumer check failed")
    final_output = finalization.get("output_r8", {})
    if (
        finalization.get("schema") != "wnn.kaggle.teacher_cohort_r5_finalization.r8"
        or final_output.get("archive_sha256") != archive_sha
        or final_output.get("root_receipt_sha256") != root_sha
        or final_output.get("manifest_sha256") != sha_bytes(manifest_path.read_bytes())
        or final_output.get("bootstrap_sha256") != sha_bytes(bootstrap)
        or finalization.get("provider_actions") != 0
    ):
        raise ValueError("R8 finalization receipt does not bind encoded package bytes")
    result = {
        "schema": "wnn.kaggle.teacher_cohort_r5_finalization_validation.r8",
        "status": "pass_actual_archive_consumer_and_root_binding",
        "archive_sha256": archive_sha,
        "manifest_sha256": sha_bytes(manifest_path.read_bytes()),
        "bootstrap_sha256": sha_bytes(bootstrap),
        "gate_sha256": sha_bytes(gate_path.read_bytes()),
        "root_release_receipt_sha256": root_sha,
        "root_authorized_conditionally": root.get("authorized") is True,
        "spec_names_closed": len(expected_names),
        "actual_archive_consumer": consumer,
        "provider_actions": 0,
    }
    write_json(package / "R5_FINALIZATION_VALIDATION_R8.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r7-package", type=Path, required=True)
    parser.add_argument("--root-release-receipt", type=Path, required=True)
    parser.add_argument("--expected-root-release-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    args = parser.parse_args()
    external_receipt = PACKET / "R5_FINALIZATION_RECEIPT_R8.json"
    if external_receipt.exists() or external_receipt.is_symlink():
        raise FileExistsError(f"refusing to overwrite external R8 receipt: {external_receipt}")
    receipt = finalize(
        args.r7_package, args.root_release_receipt, args.expected_root_release_sha256, args.output
    )
    result = validate_output(args.output, args.fixture_dir, args.expected_root_release_sha256)
    write_json(external_receipt, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
