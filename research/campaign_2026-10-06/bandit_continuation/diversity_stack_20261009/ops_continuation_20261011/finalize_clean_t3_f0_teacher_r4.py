"""Create an R4 admitted-output clone from immutable R3 and root authority."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

R3_RELATIVE = PurePosixPath(
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011/"
    "clean_t3_fold0_h390_r3_teacher_cohort"
)
R4_NAME = "clean_t3_fold0_h390_r4_teacher_cohort"
R4_DATASET = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r4"
ROOT_RECEIPT_NAME = "ROOT_CONDITIONAL_RELEASE_R4.json"
ROOT_RECEIPT_RELATIVE = str(R3_RELATIVE / ROOT_RECEIPT_NAME)
ROOT_SCHEMA = "wnn.root_conditional_teacher_cohort_release.r4"
RELEASE_SCHEMA = "wnn.kaggle.conditional_teacher_fit_admission.r4"
R3_GATE_NAME = "CONDITIONAL_FIT_ADMISSION_R3.json"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_ALLOWED_DELTA = [
    "embed_exact_root_conditional_release_receipt_r4",
    "authorize_only_the_existing_r3_profile_gated_fit_branch",
    "rebind_bootstrap_constants_for_r4_slug_archive_gate_and_release_receipt",
    "reseal_archive_manifest_and_job_gate_descriptors_without_science_changes",
    "rebind_dataset_and_kernel_metadata_to_r4_slug",
]


class FinalizationError(ValueError):
    """Raised when authority, source identity, or output collision checks fail."""


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Match the canonical JSON hash used by the Kaggle worker specs."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def write_json(path: Path, value: object) -> None:
    """Write deterministic human-readable JSON with a trailing newline."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalizationError(f"expected JSON object: {path}")
    return value


def files_with_suffix(directory: Path, suffix: str) -> list[Path]:
    """Return sorted direct children with the requested suffix."""
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix == suffix)


def r3_packet_bindings(packet: Path) -> dict[str, Any]:
    """Compute the immutable R3 source, recipe, job and profile bindings."""
    validator_path = packet / "validate_teacher_cohort_r3.py"
    validator_spec = importlib.util.spec_from_file_location(
        "r3_static_validator_for_r4", validator_path
    )
    if validator_spec is None or validator_spec.loader is None:
        raise FinalizationError("R3 static validator is missing")
    validator = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator)
    project_root = packet.parents[len(R3_RELATIVE.parts) - 1]
    producer_validation = validator.validate(project_root, packet / "bundle_teacher_cohort_r3")
    if producer_validation.get("status") != "PASS":
        raise FinalizationError("frozen R3 actual-producer/source validator did not pass")
    bundle = packet / "bundle_teacher_cohort_r3"
    manifest_path = bundle / "bundle-manifest.json"
    bootstrap_path = bundle / "kernel/kernel.py"
    manifest = read_json(manifest_path)
    config = read_json(packet / "KERNEL_CONFIGURATION_R3.json")
    plan = read_json(packet / "TEACHER_COHORT_PLAN_R3.json")
    job_paths = files_with_suffix(packet / "jobs", ".json")
    if len(job_paths) != 14:
        raise FinalizationError(f"expected 14 frozen R3 job specs; found {len(job_paths)}")
    job_hashes = {path.name: sha256_file(path) for path in job_paths}
    specs = {path.stem: read_json(path) for path in job_paths}
    recipe = {
        "plan": plan,
        "jobs": {
            name: {
                "kind": spec["kind"],
                "argv": spec["argv"],
                "artifacts": spec["artifacts"],
                "limits": spec["limits"],
                "environment": spec["environment"],
            }
            for name, spec in sorted(specs.items())
        },
    }
    source_manifest = packet / "source_snapshot/SOURCE_MANIFEST_H390.json"
    driver = packet / "kernel_support/teacher_cohort_r3_driver.py"
    expected = {
        "r3_packet_relative_path": str(R3_RELATIVE),
        "r3_dataset": manifest["dataset"],
        "r3_archive_sha256": manifest["archive_sha256"],
        "r3_bundle_manifest_sha256": sha256_file(manifest_path),
        "r3_bootstrap_sha256": sha256_file(bootstrap_path),
        "r3_gate_sha256": sha256_file(packet / R3_GATE_NAME),
        "r3_source_manifest_sha256": sha256_file(source_manifest),
        "r3_source_binding_sha256": sha256_file(packet / "SOURCE_BINDING_R3.json"),
        "r3_input_binding_sha256": sha256_file(packet / "INPUT_BINDING_R3.json"),
        "r3_producer_preflight_sha256": sha256_file(
            packet / "ACTUAL_PRODUCER_PREFLIGHT_R2_RECEIPT.json"
        ),
        "r3_profile_guard_contract_sha256": sha256_file(packet / "PROFILE_GUARD_CONTRACT_R3.json"),
        "r3_profile_reference_sha256": sha256_file(packet / "PROFILE_REFERENCE_R3.json"),
        "r3_plan_sha256": sha256_file(packet / "TEACHER_COHORT_PLAN_R3.json"),
        "r3_collection_contract_sha256": sha256_file(packet / "COLLECTION_CONTRACT_R3.json"),
        "r3_source_delta_sha256": sha256_file(packet / "SOURCE_DELTA_R3.json"),
        "r3_static_validation_sha256": sha256_file(packet / "STATIC_VALIDATION_R3.json"),
        "r3_root_release_gates_sha256": sha256_file(packet / "ROOT_RELEASE_GATES_R3.json"),
        "r3_kernel_configuration_sha256": sha256_file(packet / "KERNEL_CONFIGURATION_R3.json"),
        "r3_driver_sha256": sha256_file(driver),
        "r3_job_specs_sha256": job_hashes,
        "r3_science_recipe_sha256": canonical_sha256(recipe),
    }
    if sha256_file(bundle / "bundle.wnn") != manifest["archive_sha256"]:
        raise FinalizationError("R3 archive does not match its frozen manifest")
    if manifest["archive_sha256"] != config["bundle_archive_sha256"]:
        raise FinalizationError("R3 archive hash differs from its kernel configuration")
    if expected["r3_bundle_manifest_sha256"] != config["bundle_manifest_sha256"]:
        raise FinalizationError("R3 manifest hash differs from its kernel configuration")
    if expected["r3_bootstrap_sha256"] != config["bootstrap_sha256"]:
        raise FinalizationError("R3 bootstrap hash differs from its kernel configuration")
    if sha256_file(source_manifest) != config["profile_source_manifest_sha256"]:
        raise FinalizationError("R3 source manifest differs from its kernel configuration")
    gate = read_json(packet / R3_GATE_NAME)
    if (
        gate.get("release_authorized") is not False
        or gate.get("status") != "pending_root_review_and_release"
    ):
        raise FinalizationError("R3 must remain the original pending packet")
    if manifest.get("private") is not True:
        raise FinalizationError("R3 bundle must be private")
    return expected


def validate_root_release(
    receipt_path: Path,
    expected_sha256: str,
    packet_bindings: dict[str, Any],
) -> tuple[dict[str, Any], bytes, str]:
    """Require the exact external root authorization and all frozen R3 bindings."""
    if not SHA_RE.fullmatch(expected_sha256):
        raise FinalizationError("expected root receipt SHA-256 must be 64 lowercase hex chars")
    receipt_bytes = receipt_path.read_bytes()
    observed_sha256 = sha256_bytes(receipt_bytes)
    if observed_sha256 != expected_sha256:
        raise FinalizationError("root receipt does not match the separately supplied expected SHA")
    receipt = json.loads(receipt_bytes)
    if not isinstance(receipt, dict):
        raise FinalizationError("root release receipt must be a JSON object")
    if (
        receipt.get("schema") != ROOT_SCHEMA
        or receipt.get("status") != "admitted"
        or receipt.get("authorized") is not True
    ):
        raise FinalizationError("root release receipt is not authorized under the R4 schema")
    binding = receipt.get("binding")
    if not isinstance(binding, dict) or binding.get("preseal_r3") != packet_bindings:
        raise FinalizationError("root receipt R3 preseal bindings differ from frozen packet")
    profile_collection = binding.get("previous_profile_collection")
    if (
        not isinstance(profile_collection, dict)
        or profile_collection.get("file_count") != 77
        or not isinstance(profile_collection.get("sha256"), str)
        or not SHA_RE.fullmatch(profile_collection["sha256"])
    ):
        raise FinalizationError("root receipt must bind the previous 77-file profile collection")
    quota = binding.get("quota_budget")
    if not isinstance(quota, dict):
        raise FinalizationError("root receipt is missing quota and budget binding")
    if (
        quota.get("wall_ceiling_seconds") != 14_400
        or quota.get("two_device_gpu_seconds_upper_bound") != 28_800
        or quota.get("reserved_gpu_seconds") != 28_800
        or quota.get("paid_spending_authorized") is not False
    ):
        raise FinalizationError("root receipt quota/budget does not match the bounded R3 cohort")
    if (
        not isinstance(quota.get("campaign_headroom_gpu_seconds"), (int, float))
        or quota["campaign_headroom_gpu_seconds"] < 28_800
    ):
        raise FinalizationError("root receipt lacks campaign headroom for the declared reservation")
    if (
        not isinstance(quota.get("provider_headroom_gpu_seconds"), (int, float))
        or quota["provider_headroom_gpu_seconds"] < 28_800
    ):
        raise FinalizationError("root receipt lacks provider headroom for the declared reservation")
    if receipt.get("allowed_delta") != EXPECTED_ALLOWED_DELTA:
        raise FinalizationError(
            "root receipt allowed-delta list differs from the finalizer contract"
        )
    target = receipt.get("target")
    if not isinstance(target, dict) or target.get("dataset") != R4_DATASET:
        raise FinalizationError("root receipt does not bind the intended R4 dataset slug")
    if target.get("kernel_id") != f"{R4_DATASET}-runner":
        raise FinalizationError("root receipt does not bind the title-derived R4 kernel ID")
    return receipt, receipt_bytes, observed_sha256


def read_tar_bytes(archive_path: Path) -> tuple[list[tuple[tarfile.TarInfo, bytes]], list[str]]:
    """Read a small sealed source archive without extracting untrusted paths."""
    entries: list[tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        seen: set[str] = set()
        for member in archive:
            if not member.isfile() or member.name in seen:
                raise FinalizationError("R3 archive contains a non-file or duplicate member")
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise FinalizationError("R3 archive contains an unsafe path")
            source = archive.extractfile(member)
            if source is None:
                raise FinalizationError("R3 archive member has no data")
            with source:
                data = source.read()
            if len(data) != member.size:
                raise FinalizationError(f"short archive member read: {member.name}")
            entries.append((member, data))
            seen.add(member.name)
    return entries, sorted(seen)


def transform_archive(
    source_archive: Path,
    destination_archive: Path,
    manifest: dict[str, Any],
    gate_bytes: bytes,
    receipt_bytes: bytes,
    receipt_sha256: str,
) -> tuple[str, int]:
    """Reseal only gate/spec descriptors plus the root receipt archive member."""
    gate_path = str(R3_RELATIVE / R3_GATE_NAME)
    receipt_path = ROOT_RECEIPT_RELATIVE
    entries, names = read_tar_bytes(source_archive)
    spec_paths = set(manifest["spec_paths"])
    if len(spec_paths) != 14:
        raise FinalizationError("R3 manifest does not contain exactly 14 stage specs")
    manifest_paths = {row["path"] for row in manifest["files"]}
    allowed_names = (manifest_paths - set(manifest["external_mounts"])) | spec_paths
    if set(names) != allowed_names:
        raise FinalizationError("R3 archive member list differs from its manifest")
    if gate_path not in manifest_paths:
        raise FinalizationError("R3 manifest does not declare the admission gate")
    output_entries: list[tuple[tarfile.TarInfo, bytes]] = []
    gate_rows = 0
    for original_info, original_data in entries:
        name = original_info.name
        data = original_data
        if name == gate_path:
            data = gate_bytes
            gate_rows += 1
        elif name in spec_paths:
            spec = json.loads(data)
            rows = [row for row in spec["code"] if row.get("path") == gate_path]
            if len(rows) != 1:
                raise FinalizationError(f"spec gate descriptor is not unique: {name}")
            rows[0]["sha256"] = sha256_bytes(gate_bytes)
            rows[0]["size"] = len(gate_bytes)
            spec["code_sha256"] = canonical_sha256(
                sorted(spec["code"], key=lambda row: row["path"])
            )
            data = (json.dumps(spec, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = original_info.mode
        info.mtime = 0
        output_entries.append((info, data))
    if gate_rows != 1:
        raise FinalizationError("R3 archive gate member is missing or duplicated")
    receipt_info = tarfile.TarInfo(receipt_path)
    receipt_info.size = len(receipt_bytes)
    receipt_info.mode = 0o644
    receipt_info.mtime = 0
    output_entries.append((receipt_info, receipt_bytes))
    with (
        destination_archive.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as output_tar,
    ):
        for info, data in output_entries:
            output_tar.addfile(info, io.BytesIO(data))
    return sha256_file(destination_archive), destination_archive.stat().st_size


def verify_source_closure(packet: Path, manifest: dict[str, Any]) -> None:
    """Verify all packet-side code bytes and immutable source snapshot entries."""
    source_manifest_path = packet / "source_snapshot/SOURCE_MANIFEST_H390.json"
    source_manifest = read_json(source_manifest_path)
    sources = source_manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise FinalizationError("R3 source manifest is empty or malformed")
    for rel, expected in sources.items():
        path = packet / "source_snapshot" / rel
        if not path.is_file() or sha256_file(path) != expected:
            raise FinalizationError(f"R3 source snapshot mismatch: {rel}")
    project_root = packet.parents[len(R3_RELATIVE.parts) - 1]
    for row in manifest["files"]:
        rel = row["path"]
        if rel in manifest["external_mounts"]:
            continue
        member_path = PurePosixPath(rel)
        if member_path.parts[: len(R3_RELATIVE.parts)] == R3_RELATIVE.parts:
            path = packet / Path(*member_path.parts[len(R3_RELATIVE.parts) :])
        else:
            path = project_root / Path(*member_path.parts)
        if (
            not path.is_file()
            or path.stat().st_size != row["size"]
            or sha256_file(path) != row["sha256"]
        ):
            raise FinalizationError(f"R3 packet code member differs from archive manifest: {rel}")


def update_manifest_descriptors(
    manifest: dict[str, Any],
    gate_bytes: bytes,
    receipt_bytes: bytes,
    archive_sha256: str,
    archive_bytes: int,
) -> dict[str, Any]:
    """Update only the expected R4 identity, gate, receipt and archive bindings."""
    result = json.loads(json.dumps(manifest))
    gate_path = str(R3_RELATIVE / R3_GATE_NAME)
    gate_rows = [row for row in result["files"] if row["path"] == gate_path]
    if len(gate_rows) != 1:
        raise FinalizationError("manifest admission gate descriptor is not unique")
    gate_rows[0]["sha256"] = sha256_bytes(gate_bytes)
    gate_rows[0]["size"] = len(gate_bytes)
    if any(row["path"] == ROOT_RECEIPT_RELATIVE for row in result["files"]):
        raise FinalizationError("R3 unexpectedly already contains an R4 authority member")
    result["files"].append(
        {
            "path": ROOT_RECEIPT_RELATIVE,
            "sha256": sha256_bytes(receipt_bytes),
            "size": len(receipt_bytes),
        }
    )
    result["dataset"] = R4_DATASET
    result["archive_sha256"] = archive_sha256
    result["archive_bytes"] = archive_bytes
    result["expanded_bytes"] = sum(
        row["size"] for row in result["files"] if row["path"] not in result["external_mounts"]
    )
    return result


def make_gate(packet: Path, receipt_sha256: str) -> bytes:
    """Create the R4 embedded gate from R3's exact conditional guard contract."""
    gate = read_json(packet / R3_GATE_NAME)
    gate["schema"] = RELEASE_SCHEMA
    gate["status"] = "admitted"
    gate["release_authorized"] = True
    gate["root_release_receipt_sha256"] = receipt_sha256
    gate["root_release_receipt_path"] = ROOT_RECEIPT_RELATIVE
    if gate.get("profile_guard_contract_sha256") != sha256_file(
        packet / "PROFILE_GUARD_CONTRACT_R3.json"
    ):
        raise FinalizationError("R3 gate profile guard hash does not match frozen contract")
    if gate.get("profile_reference_sha256") != sha256_file(packet / "PROFILE_REFERENCE_R3.json"):
        raise FinalizationError("R3 gate profile reference hash does not match frozen reference")
    return (json.dumps(gate, indent=2, sort_keys=True) + "\n").encode()


def update_staged_job_specs(packet: Path, gate_bytes: bytes) -> None:
    """Update only each job's gate descriptor and aggregate code hash."""
    gate_path = str(R3_RELATIVE / R3_GATE_NAME)
    gate_sha = sha256_bytes(gate_bytes)
    for path in files_with_suffix(packet / "jobs", ".json"):
        spec = read_json(path)
        rows = [row for row in spec["code"] if row.get("path") == gate_path]
        if len(rows) != 1:
            raise FinalizationError(f"R4 job gate descriptor is not unique: {path.name}")
        rows[0]["sha256"] = gate_sha
        rows[0]["size"] = len(gate_bytes)
        spec["code_sha256"] = canonical_sha256(sorted(spec["code"], key=lambda row: row["path"]))
        write_json(path, spec)


def update_bootstrap(
    bundle: Path,
    archive_sha256: str,
    gate_sha256: str,
    receipt_sha256: str,
    preseal_bindings: dict[str, Any],
) -> None:
    """Bind the new Kaggle bootstrap to R4 and verify receipt bytes before mounts."""
    kernel_path = bundle / "kernel/kernel.py"
    source = kernel_path.read_text(encoding="utf-8")
    old_slug = "wnn-clean-t3-f0-teachers-20261011-r3"
    new_slug = R4_DATASET.split("/")[1]
    replacements = {
        f"BUNDLE_SLUG = '{old_slug}'": f"BUNDLE_SLUG = '{new_slug}'",
        "EXPECTED_ARCHIVE_SHA256 = 'f6fc9de3b47289bbe7e95dc5cdf446ddd2d4b528ceb6e09cc1437a3f5ce7d2fd'": f"EXPECTED_ARCHIVE_SHA256 = {archive_sha256!r}",
        "R3_GATE_SHA256 = '14644f9feeb9f248dedae2ee2884e68fa394fb9da5aeaf5a23d16aa6b9f12df7'": f"R3_GATE_SHA256 = {gate_sha256!r}",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise FinalizationError(f"unexpected R3 bootstrap binding; cannot replace {old[:50]!r}")
        source = source.replace(old, new, 1)
    receipt_constants = (
        f"ROOT_RELEASE_RECEIPT_RELATIVE_PATH = {ROOT_RECEIPT_RELATIVE!r}\n"
        f"ROOT_RELEASE_RECEIPT_SHA256 = {receipt_sha256!r}\n"
    )
    receipt_constants += f"ROOT_PRESEAL_R3_BINDINGS = {preseal_bindings!r}\n"
    receipt_constants += f"ROOT_ALLOWED_DELTA = {EXPECTED_ALLOWED_DELTA!r}\n"
    receipt_constants += (
        f"ROOT_RELEASE_TARGET = {{'dataset': {R4_DATASET!r}, "
        f"'kernel_id': {f'{R4_DATASET}-runner'!r}}}\n"
    )
    marker = f"R3_GATE_SHA256 = {gate_sha256!r}\n"
    if source.count(marker) != 1:
        raise FinalizationError("R4 gate constant marker is missing after bootstrap reseal")
    source = source.replace(marker, marker + receipt_constants, 1)
    check_marker = (
        '    if gate_receipt.get("status") != "admitted" or gate_receipt.get("release_authorized") is not True or not gate_receipt.get("root_release_receipt_sha256"):\n'
        '        raise PermissionError("R3 conditional admission is pending; refusing before external mounts or workers")\n'
    )
    check_addition = (
        '    if gate_receipt.get("root_release_receipt_sha256") != ROOT_RELEASE_RECEIPT_SHA256:\n'
        '        raise PermissionError("R4 gate does not bind the embedded root release receipt")\n'
        '    if gate_receipt.get("root_release_receipt_path") != ROOT_RELEASE_RECEIPT_RELATIVE_PATH:\n'
        '        raise PermissionError("R4 gate names an unexpected root release receipt path")\n'
        '    if gate_receipt.get("profile_guard_contract_sha256") != ROOT_PRESEAL_R3_BINDINGS["r3_profile_guard_contract_sha256"] or gate_receipt.get("profile_reference_sha256") != ROOT_PRESEAL_R3_BINDINGS["r3_profile_reference_sha256"]:\n'
        '        raise PermissionError("R4 gate does not bind the exact profile guard and reference")\n'
        "    root_receipt_path = root / ROOT_RELEASE_RECEIPT_RELATIVE_PATH\n"
        "    if not root_receipt_path.is_file() or hashlib.sha256(root_receipt_path.read_bytes()).hexdigest() != ROOT_RELEASE_RECEIPT_SHA256:\n"
        '        raise PermissionError("R4 root release receipt bytes are missing or changed")\n'
        "    root_receipt = json.loads(root_receipt_path.read_text())\n"
        '    if root_receipt.get("schema") != "wnn.root_conditional_teacher_cohort_release.r4" or root_receipt.get("status") != "admitted" or root_receipt.get("authorized") is not True:\n'
        '        raise PermissionError("R4 root release receipt is not authorized")\n'
        '    if root_receipt.get("binding", {}).get("preseal_r3") != ROOT_PRESEAL_R3_BINDINGS:\n'
        '        raise PermissionError("R4 root receipt does not bind the exact R3 source and jobs")\n'
        '    if root_receipt.get("target") != ROOT_RELEASE_TARGET:\n'
        '        raise PermissionError("R4 root receipt does not bind the exact title-derived target")\n'
        '    if root_receipt.get("allowed_delta") != ROOT_ALLOWED_DELTA:\n'
        '        raise PermissionError("R4 root receipt allows an unexpected packet delta")\n'
        '    root_binding = root_receipt.get("binding", {})\n'
        '    prior_profile = root_binding.get("previous_profile_collection", {})\n'
        '    if prior_profile.get("file_count") != 77 or not isinstance(prior_profile.get("sha256"), str) or len(prior_profile["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in prior_profile["sha256"]):\n'
        '        raise PermissionError("R4 root receipt omits the prior 77-file profile collection")\n'
        '    quota_budget = root_binding.get("quota_budget", {})\n'
        '    if quota_budget.get("wall_ceiling_seconds") != 14400 or quota_budget.get("two_device_gpu_seconds_upper_bound") != 28800 or quota_budget.get("reserved_gpu_seconds") != 28800 or quota_budget.get("paid_spending_authorized") is not False:\n'
        '        raise PermissionError("R4 root receipt quota/budget fields differ from the sealed ceiling")\n'
        '    if quota_budget.get("campaign_headroom_gpu_seconds", 0) < 28800 or quota_budget.get("provider_headroom_gpu_seconds", 0) < 28800:\n'
        '        raise PermissionError("R4 root receipt lacks headroom for the declared reservation")\n'
    )
    if source.count(check_marker) != 1:
        raise FinalizationError("R3 bootstrap admission boundary changed unexpectedly")
    source = source.replace(check_marker, check_marker + check_addition, 1)
    compile(source, str(kernel_path), "exec")
    kernel_path.write_text(source, encoding="utf-8")


def validate_r4(
    packet: Path, expected_release_sha256: str, source_r3_packet: Path
) -> dict[str, Any]:
    """Validate final hashes, gate, archive members, exact specs and upload slugs."""
    bundle = packet / "bundle_teacher_cohort_r3"
    manifest_path = bundle / "bundle-manifest.json"
    manifest = read_json(manifest_path)
    config = read_json(packet / "KERNEL_CONFIGURATION_R4.json")
    gate = read_json(packet / R3_GATE_NAME)
    receipt_path = packet / ROOT_RECEIPT_NAME
    receipt_sha = sha256_file(receipt_path)
    if receipt_sha != expected_release_sha256:
        raise FinalizationError("staged root release receipt hash differs from requested authority")
    finalization = read_json(packet / "R4_FINALIZATION_RECEIPT_R1.json")
    validate_root_release(receipt_path, expected_release_sha256, finalization["source_bindings"])
    if (
        gate.get("status") != "admitted"
        or gate.get("release_authorized") is not True
        or gate.get("root_release_receipt_sha256") != receipt_sha
    ):
        raise FinalizationError("embedded conditional gate does not bind admitted root receipt")
    if manifest.get("dataset") != R4_DATASET or config.get("dataset") != R4_DATASET:
        raise FinalizationError("R4 dataset slug mismatch")
    if sha256_file(bundle / "bundle.wnn") != manifest["archive_sha256"]:
        raise FinalizationError("R4 archive hash does not match bundle manifest")
    if manifest["archive_sha256"] != config["bundle_archive_sha256"]:
        raise FinalizationError("R4 archive hash does not match kernel configuration")
    if sha256_file(manifest_path) != config["bundle_manifest_sha256"]:
        raise FinalizationError("R4 manifest hash does not match kernel configuration")
    bootstrap = bundle / "kernel/kernel.py"
    bootstrap_source = bootstrap.read_text(encoding="utf-8")
    compile(bootstrap_source, str(bootstrap), "exec")
    for marker in (
        "ROOT_RELEASE_RECEIPT_SHA256",
        "ROOT_PRESEAL_R3_BINDINGS",
        "ROOT_ALLOWED_DELTA",
        "root_receipt_path.is_file()",
        "previous_profile_collection",
        "quota_budget",
    ):
        if marker not in bootstrap_source:
            raise FinalizationError(f"R4 bootstrap lacks the root-receipt guard: {marker}")
    if sha256_file(bootstrap) != config["bootstrap_sha256"]:
        raise FinalizationError("R4 bootstrap hash does not match kernel configuration")
    kernel_meta = read_json(bundle / "kernel/kernel-metadata.json")
    dataset_meta = read_json(bundle / "dataset-metadata.json")
    if (
        kernel_meta.get("id") != f"{R4_DATASET}-runner"
        or kernel_meta.get("title") != f"{R4_DATASET.split('/')[1]} runner"
    ):
        raise FinalizationError("R4 kernel title and title-derived id differ")
    if kernel_meta.get("dataset_sources", [None])[0] != R4_DATASET:
        raise FinalizationError("R4 kernel metadata does not mount the R4 dataset")
    if dataset_meta.get("id") != R4_DATASET or dataset_meta.get("isPrivate") is not True:
        raise FinalizationError("R4 dataset metadata is not the exact private target")
    if (
        kernel_meta.get("is_private") is not True
        or kernel_meta.get("machine_shape") != "NvidiaTeslaT4"
    ):
        raise FinalizationError("R4 kernel metadata changed privacy or device target")
    entries, names = read_tar_bytes(bundle / "bundle.wnn")
    file_paths = {
        row["path"] for row in manifest["files"] if row["path"] not in manifest["external_mounts"]
    }
    allowed = file_paths | set(manifest["spec_paths"])
    if set(names) != allowed:
        raise FinalizationError("R4 archive content differs from its manifest allowlist")
    by_name = {info.name: data for info, data in entries}
    file_rows = {row["path"]: row for row in manifest["files"]}
    for rel in file_paths:
        row = file_rows[rel]
        data = by_name[rel]
        if len(data) != row["size"] or sha256_bytes(data) != row["sha256"]:
            raise FinalizationError(f"R4 archive file identity mismatch: {rel}")
    if by_name[ROOT_RECEIPT_RELATIVE] != receipt_path.read_bytes():
        raise FinalizationError("archive does not carry the exact root release receipt")
    specs = {}
    for rel in manifest["spec_paths"]:
        spec = json.loads(by_name[rel])
        if (
            canonical_sha256(sorted(spec["code"], key=lambda row: row["path"]))
            != spec["code_sha256"]
        ):
            raise FinalizationError(f"R4 spec code aggregate hash mismatch: {rel}")
        gate_rows = [row for row in spec["code"] if row["path"] == str(R3_RELATIVE / R3_GATE_NAME)]
        if len(gate_rows) != 1 or gate_rows[0]["sha256"] != sha256_bytes(
            by_name[str(R3_RELATIVE / R3_GATE_NAME)]
        ):
            raise FinalizationError(f"R4 spec does not bind admitted gate bytes: {rel}")
        specs[spec["name"]] = spec
    if len(specs) != 14:
        raise FinalizationError("R4 archive does not contain 14 original stage specs")
    for name, spec in specs.items():
        original = read_json(source_r3_packet / "jobs" / f"{name}.json")
        for key in (
            "version",
            "name",
            "kind",
            "argv",
            "inputs",
            "artifacts",
            "limits",
            "environment",
        ):
            if original.get(key) != spec.get(key):
                raise FinalizationError(f"R4 changed frozen R3 job recipe {name}:{key}")
        original_code = json.loads(json.dumps(original["code"]))
        gate_bytes = by_name[str(R3_RELATIVE / R3_GATE_NAME)]
        for row in original_code:
            if row["path"] == str(R3_RELATIVE / R3_GATE_NAME):
                row["sha256"] = sha256_bytes(gate_bytes)
                row["size"] = len(gate_bytes)
        if original_code != spec["code"]:
            raise FinalizationError(f"R4 changed more than the gate descriptor in job {name}")
    return {
        "schema": "wnn.kaggle.teacher_cohort_r4_finalization_validation.r1",
        "status": "PASS",
        "dataset": R4_DATASET,
        "kernel_id": kernel_meta["id"],
        "root_release_receipt_sha256": receipt_sha,
        "archive_sha256": manifest["archive_sha256"],
        "archive_bytes": manifest["archive_bytes"],
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "bootstrap_sha256": sha256_file(bootstrap),
        "archive_member_count": len(names),
        "worker_spec_count": len(specs),
        "external_mount_count": len(manifest["external_mounts"]),
        "provider_actions": 0,
    }


def copy_upload_stage(packet: Path, stage: Path) -> dict[str, Any]:
    """Create short physical dataset and kernel staging directories, never reuse."""
    stage = stage.resolve(strict=False)
    if stage.exists():
        raise FinalizationError(f"short upload stage already exists: {stage}")
    dataset_stage = stage / "dataset"
    kernel_stage = stage / "kernel"
    dataset_stage.mkdir(parents=True)
    kernel_stage.mkdir()
    bundle = packet / "bundle_teacher_cohort_r3"
    dataset_files = ["bundle.wnn", "bundle-manifest.json", "dataset-metadata.json"]
    kernel_files = ["kernel.py", "kernel-metadata.json"]
    for name in dataset_files:
        shutil.copyfile(bundle / name, dataset_stage / name)
    for name in kernel_files:
        shutil.copyfile(bundle / "kernel" / name, kernel_stage / name)
    expected = {
        "dataset": {name: sha256_file(dataset_stage / name) for name in dataset_files},
        "kernel": {name: sha256_file(kernel_stage / name) for name in kernel_files},
    }
    for name, digest in expected["dataset"].items():
        if digest != sha256_file(bundle / name):
            raise FinalizationError(f"dataset staging copy mismatch: {name}")
    for name, digest in expected["kernel"].items():
        if digest != sha256_file(bundle / "kernel" / name):
            raise FinalizationError(f"kernel staging copy mismatch: {name}")
    return {"path": str(stage), "files": expected, "provider_actions": 0}


def finalize(
    r3_packet: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
    output_packet: Path,
    upload_stage: Path,
) -> dict[str, Any]:
    """Seal a new R4 output and short upload stage without touching R3."""
    r3_packet = r3_packet.resolve(strict=True)
    output_packet = output_packet.resolve(strict=False)
    upload_stage = upload_stage.resolve(strict=False)
    if output_packet.exists():
        raise FinalizationError(f"R4 output already exists; create-only refusal: {output_packet}")
    if (
        output_packet == r3_packet
        or r3_packet in output_packet.parents
        or output_packet in r3_packet.parents
    ):
        raise FinalizationError("R4 output must be separate from the immutable R3 source")
    if (
        upload_stage == r3_packet
        or r3_packet in upload_stage.parents
        or upload_stage in r3_packet.parents
    ):
        raise FinalizationError("short upload stage must be outside the immutable R3 source tree")
    if (
        upload_stage == output_packet
        or output_packet in upload_stage.parents
        or upload_stage in output_packet.parents
    ):
        raise FinalizationError("short upload stage must be separate from the R4 packet directory")
    if upload_stage.exists():
        raise FinalizationError(
            f"short upload stage already exists; create-only refusal: {upload_stage}"
        )
    packet_bindings = r3_packet_bindings(r3_packet)
    root_receipt, receipt_bytes, receipt_sha = validate_root_release(
        receipt_path.resolve(strict=True), expected_receipt_sha256, packet_bindings
    )
    source_bundle = r3_packet / "bundle_teacher_cohort_r3"
    source_manifest = read_json(source_bundle / "bundle-manifest.json")
    verify_source_closure(r3_packet, source_manifest)
    gate_bytes = make_gate(r3_packet, receipt_sha)
    parent = output_packet.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{R4_NAME}.building-", dir=parent))
    out = stage_dir / R4_NAME
    try:
        shutil.copytree(r3_packet, out)
        bundle = out / "bundle_teacher_cohort_r3"
        write_json(out / R3_GATE_NAME, json.loads(gate_bytes))
        (out / ROOT_RECEIPT_NAME).write_bytes(receipt_bytes)
        update_staged_job_specs(out, gate_bytes)
        manifest_path = bundle / "bundle-manifest.json"
        manifest = read_json(manifest_path)
        archive_sha, archive_bytes = transform_archive(
            source_bundle / "bundle.wnn",
            bundle / "bundle.wnn",
            manifest,
            gate_bytes,
            receipt_bytes,
            receipt_sha,
        )
        updated_manifest = update_manifest_descriptors(
            manifest, gate_bytes, receipt_bytes, archive_sha, archive_bytes
        )
        write_json(manifest_path, updated_manifest)
        write_json(
            bundle / "dataset-metadata.json",
            {
                "id": R4_DATASET,
                "title": R4_DATASET.split("/")[1],
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
        )
        kernel_meta_path = bundle / "kernel/kernel-metadata.json"
        kernel_meta = read_json(kernel_meta_path)
        kernel_meta["id"] = f"{R4_DATASET}-runner"
        kernel_meta["title"] = f"{R4_DATASET.split('/')[1]} runner"
        kernel_meta["dataset_sources"][0] = R4_DATASET
        write_json(kernel_meta_path, kernel_meta)
        kernel_path = bundle / "kernel/kernel.py"
        update_bootstrap(
            bundle, archive_sha, sha256_bytes(gate_bytes), receipt_sha, packet_bindings
        )
        config = {
            "schema": "wnn.kaggle.teacher_cohort_kernel_config.r4",
            "dataset": R4_DATASET,
            "kernel_id": kernel_meta["id"],
            "kernel_title": kernel_meta["title"],
            "bundle_archive_sha256": archive_sha,
            "bundle_manifest_sha256": sha256_file(manifest_path),
            "bootstrap_sha256": sha256_file(kernel_path),
            "bootstrap_bytes": kernel_path.stat().st_size,
            "source_manifest_sha256": packet_bindings["r3_source_manifest_sha256"],
            "r3_archive_sha256": packet_bindings["r3_archive_sha256"],
            "r3_bundle_manifest_sha256": packet_bindings["r3_bundle_manifest_sha256"],
            "r3_bootstrap_sha256": packet_bindings["r3_bootstrap_sha256"],
            "r3_driver_sha256": packet_bindings["r3_driver_sha256"],
            "r3_science_recipe_sha256": packet_bindings["r3_science_recipe_sha256"],
            "root_release_receipt_sha256": receipt_sha,
            "wall_ceiling_seconds": 14_400,
            "two_device_gpu_seconds_upper_bound": 28_800,
            "fit_eta_seconds": None,
            "export_eta_seconds": None,
            "total_eta_seconds": None,
            "final_dispatch_authorized": False,
        }
        write_json(out / "KERNEL_CONFIGURATION_R4.json", config)
        write_json(
            out / "R4_FINALIZATION_RECEIPT_R1.json",
            {
                "schema": "wnn.kaggle.teacher_cohort_r4_finalization.r1",
                "status": "conditional_gate_embedded_final_bytes_pending_root_admission",
                "source_packet": str(R3_RELATIVE),
                "source_bindings": packet_bindings,
                "preseal_root_receipt_sha256": receipt_sha,
                "archive_sha256": archive_sha,
                "bundle_manifest_sha256": sha256_file(manifest_path),
                "bootstrap_sha256": sha256_file(kernel_path),
                "dataset": R4_DATASET,
                "kernel_id": kernel_meta["id"],
                "root_receipt_status": root_receipt["status"],
                "final_dispatch_authorized": False,
                "provider_actions": 0,
            },
        )
        (out / "README_R4.md").write_text(
            "# Clean T3 fold-0 teacher cohort R4\n\n"
            "R4 is a create-only, root-authorized derivative of the frozen R3 science packet. "
            "The embedded R3 source/job identifiers remain unchanged; R4 changes only the "
            "conditional gate, its job-code hashes, the root-receipt verification boundary, "
            "the sealed bundle/manifest, and Kaggle dataset/kernel identity. The root receipt "
            "is embedded byte-for-byte and checked before external mounts or workers.\n\n"
            "The same three H384 Muon-all / propensity-power-2 fold-0 teachers (seeds 3, 4, 5), "
            "paired T4 profile guard, fixed epoch-2 checkpoints, and nine fit/full-valid exports "
            "remain bound to the R3 source and recipe. The fit branch is conditional on both "
            "actual paired profiles satisfying the frozen finite-update and resource guards.\n\n"
            "The prospective cohort ceilings are 14,400 wall seconds and 28,800 two-device "
            "GPU seconds; they are reservations, not measured ETA. This output is not uploaded "
            "or dispatched by the finalizer. A separate root final-admission receipt must bind "
            "the sealed R4 archive, manifest, bootstrap, and embedded receipt before dispatch.\n",
            encoding="utf-8",
        )
        validation = validate_r4(out, receipt_sha, r3_packet)
        write_json(out / "R4_FINALIZATION_VALIDATION_R1.json", validation)
        stage_receipt = copy_upload_stage(out, upload_stage)
        write_json(out / "R4_SHORT_UPLOAD_STAGE_R1.json", stage_receipt)
        if output_packet.exists():
            raise FinalizationError("R4 output appeared during create-only finalization")
        out.rename(output_packet)
        stage_dir.rmdir()
        return {
            **validation,
            "packet_path": str(output_packet),
            "short_upload_stage": stage_receipt,
            "build_temp_parent": str(stage_dir),
            "root_receipt_sha256": receipt_sha,
        }
    except BaseException:
        # Preserve the uniquely named private build directory for diagnosis.
        raise


def main() -> None:
    """Finalize only when a separately supplied, exact root receipt authorizes it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-packet", type=Path, required=True)
    parser.add_argument("--root-release-receipt", type=Path, required=True)
    parser.add_argument("--expected-root-release-receipt-sha256", required=True)
    parser.add_argument("--output-packet", type=Path, required=True)
    parser.add_argument("--short-upload-stage", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        args.r3_packet,
        args.root_release_receipt,
        args.expected_root_release_receipt_sha256,
        args.output_packet,
        args.short_upload_stage,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
