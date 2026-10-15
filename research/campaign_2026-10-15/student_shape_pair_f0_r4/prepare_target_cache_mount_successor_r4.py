from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET_R1 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r1"
PACKET_R3 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r3"
PACKET_R4 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r4"
BUNDLE_R3 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r3"
BUNDLE_R4 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r4"
R3_SEAL = PACKET_R3 / "GENERATED_KERNEL_SEAL_R3_SEALED.json"
R3_BUILD = PACKET_R3 / "OFFLINE_BUNDLE_BUILD_R3.json"
R3_MOUNT = PACKET_R3 / "KERNEL_MOUNT_SUCCESSOR_R3.json"
R3_FAILURE = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r3/PROVIDER_KERNEL_LOG_R3.txt"
INPUT_BINDING = PACKET_R4 / "INPUT_BINDING_R4.json"
SOURCE_BINDING = PACKET_R4 / "SOURCE_BINDING_R4.json"
BUILD_RECEIPT = PACKET_R4 / "OFFLINE_BUNDLE_BUILD_R4.json"
MOUNT_RECEIPT = PACKET_R4 / "KERNEL_MOUNT_SUCCESSOR_R4.json"
SPLIT = MAIN / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PORTABLE = MAIN / "research/campaign_2026-10-04/portable_inputs.json"
FOLD0_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
TARGETS = (
    {
        "logical_path": "cache/train_targets.npz",
        "identity_name": "cache/train_targets.npz",
    },
    {
        "logical_path": "cache/valid_targets.npz",
        "identity_name": "cache/valid_targets.npz",
    },
)
DEFERRED = "datasets/train.parquet,datasets/valid.parquet,cache/train_targets.npz,cache/valid_targets.npz"


def sha256_file(path: Path) -> str:
    """Hash a small source or generated packet artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, max_bytes: int = 4_000_000) -> dict[str, Any]:
    """Read bounded metadata only; never read target cache payloads here."""
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"refusing oversized or empty JSON: {path} ({size} bytes)")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Write a new deterministic packet artifact without replacing evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen R4 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def external_map(binding: dict[str, Any]) -> dict[str, object]:
    """Convert bound assets to the outer-kernel mount-map format."""
    return {
        row["logical_path"]: (
            row["filename"]
            if row["dataset"] == "vladkochetov07/wunder-funder"
            else {"dataset": row["dataset"], "filename": row["filename"]}
        )
        for row in binding["external_mounts"]
    }


def target_descriptors() -> list[dict[str, Any]]:
    """Bind the two target caches to fold0 split and portable producer metadata."""
    split = read_json(SPLIT, max_bytes=2_000_000)
    portable = read_json(PORTABLE, max_bytes=16_000)
    if sha256_file(SPLIT) != "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d":
        raise ValueError("authoritative fold0 split manifest hash changed")
    if sha256_file(PORTABLE) != "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c":
        raise ValueError("fold0 portable input registry hash changed")

    rows = []
    for target in TARGETS:
        logical = target["logical_path"]
        split_identity = split["sources"].get(logical)
        portable_identity = portable["sources"].get(logical)
        if not isinstance(split_identity, dict) or not isinstance(portable_identity, dict):
            raise ValueError(f"fold0 producer identity is missing for {logical}")
        if portable_identity.get("original_identity") != split_identity:
            raise ValueError(f"portable identity differs from frozen split for {logical}")
        content_sha = portable_identity.get("sha256")
        if not isinstance(content_sha, str) or len(content_sha) != 64:
            raise ValueError(f"portable content identity missing for {logical}")
        local = MAIN / logical
        stat = local.stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            split_identity["bytes"],
            split_identity["mtime_ns"],
        ):
            raise ValueError(f"local target cache metadata differs from frozen identity: {logical}")
        # The prior R1 producer manifest records these exact hash-named flat files.
        filename = f"{content_sha}.wnn"
        rows.append(
            {
                "dataset": FOLD0_DATASET,
                "filename": filename,
                "logical_path": logical,
                "sha256": content_sha,
                "size": int(split_identity["bytes"]),
                "version_expected": 1,
                "producer_identity": {
                    "split_path": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
                    "split_sha256": sha256_file(SPLIT),
                    "portable_registry_path": "research/campaign_2026-10-04/portable_inputs.json",
                    "portable_registry_sha256": sha256_file(PORTABLE),
                    "bytes": int(split_identity["bytes"]),
                    "mtime_ns": int(split_identity["mtime_ns"]),
                    "index_sha256": split_identity["index_sha256"],
                    "content_sha256": content_sha,
                    "local_size_mtime_checked_without_payload_read": True,
                },
            }
        )
    if {row["logical_path"] for row in rows} != {
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
    }:
        raise ValueError("target mount set is not exact")
    producer_manifest = read_json(
        PACKET_R1 / "private_readback/bundle-manifest.json", max_bytes=2_000_000
    )
    producer_mounts = producer_manifest.get("external_mounts", {})
    for row in rows:
        observed = producer_mounts.get(row["logical_path"])
        if observed != {"dataset": row["dataset"], "filename": row["filename"]}:
            raise ValueError(f"R1 producer mount differs for {row['logical_path']}")
    return rows


def patch_kernel_mounts(source: str, expected_mounts: dict[str, object]) -> str:
    """Replace only the generated kernel's external-mount assignment."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    found = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_EXTERNAL_MOUNTS"
            for target in node.targets
        ):
            lines[node.lineno - 1 : node.end_lineno] = [
                f"EXPECTED_EXTERNAL_MOUNTS = {expected_mounts!r}\n"
            ]
            found += 1
    if found != 1:
        raise ValueError("expected exactly one outer-kernel mount assignment")
    result = "".join(lines)
    ast.parse(result)
    return result


def r4_job(r3_path: Path, output_path: Path, new_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Add only the two required cache mounts and defer their worker hash check."""
    spec = read_json(r3_path)
    original = copy.deepcopy(spec)
    existing = {row["path"] for row in spec["inputs"]}
    for row in new_inputs:
        if row["logical_path"] in existing:
            raise ValueError(f"R3 job unexpectedly already declares {row['logical_path']}")
        spec["inputs"].append(
            {
                "path": row["logical_path"],
                "size": row["size"],
                "sha256": row["sha256"],
            }
        )
    spec["inputs"].sort(key=lambda row: row["path"])
    prior_deferred = spec["environment"].get("WNN_DEFERRED_IDENTITY_PATHS")
    if prior_deferred != "datasets/train.parquet,datasets/valid.parquet":
        raise ValueError("R3 deferred input list changed")
    spec["environment"]["WNN_DEFERRED_IDENTITY_PATHS"] = DEFERRED
    expected = copy.deepcopy(original)
    expected["inputs"] = spec["inputs"]
    expected["environment"]["WNN_DEFERRED_IDENTITY_PATHS"] = DEFERRED
    if spec != expected or spec["argv"] != original["argv"]:
        raise ValueError("R4 job differs outside two input descriptors and deferred identity paths")
    write_json(output_path, spec)
    return spec


def write_archive(r3_archive: Path, r4_archive: Path, replacements: dict[str, bytes]) -> None:
    """Rewrite only the two small embedded job specs in the frozen R3 archive."""
    seen: set[str] = set()
    with (
        r4_archive.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed,
        tarfile.open(fileobj=compressed, mode="w") as output,
        tarfile.open(r3_archive, mode="r:gz") as source,
    ):
        members = source.getmembers()
        if len(members) > 200:
            raise ValueError("R3 archive member count exceeds the frozen small bundle bound")
        for member in members:
            if not member.isfile() or member.name in seen or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError(f"unsupported or unsafe R3 archive member: {member.name}")
            seen.add(member.name)
            if member.name in replacements:
                payload = replacements[member.name]
            else:
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError(f"R3 archive member cannot be read: {member.name}")
                payload = stream.read(member.size + 1)
                if len(payload) != member.size:
                    raise ValueError(f"R3 archive member size changed: {member.name}")
            if len(payload) > 2_000_000:
                raise ValueError(f"unexpectedly large bundled member: {member.name}")
            updated = copy.copy(member)
            updated.size = len(payload)
            output.addfile(updated, io.BytesIO(payload))
    if set(replacements) - seen:
        raise ValueError("one or more R4 job specs were absent from R3 archive")


def main() -> None:
    """Create an R4 transport successor with the two missing fold0 cache inputs."""
    if any(path.exists() for path in (INPUT_BINDING, SOURCE_BINDING, BUILD_RECEIPT, MOUNT_RECEIPT)):
        raise FileExistsError("R4 evidence already exists; preserve it and create a successor")
    if BUNDLE_R4.exists():
        raise FileExistsError("R4 bundle stage already exists")

    r3_binding = read_json(PACKET_R3 / "INPUT_BINDING_R3.json")
    r3_source = read_json(PACKET_R3 / "SOURCE_BINDING_R3.json")
    r3_seal = read_json(R3_SEAL)
    r3_manifest_path = BUNDLE_R3 / "bundle-manifest.json"
    r3_manifest = read_json(r3_manifest_path)
    if sha256_file(BUNDLE_R3 / "bundle.wnn") != r3_seal["bundle"]["archive_sha256"]:
        raise ValueError("R3 archive differs from frozen terminal source seal")
    if sha256_file(r3_manifest_path) != r3_seal["bundle"]["manifest_sha256"]:
        raise ValueError("R3 manifest differs from frozen terminal source seal")
    new_targets = target_descriptors()

    binding = copy.deepcopy(r3_binding)
    binding["schema"] = "wnn.student-shape-pair-input-binding.r4"
    binding["excluded_inputs"] = [
        row for row in binding["excluded_inputs"]
        if row not in {"train_targets.npz", "valid_targets.npz"}
    ]
    binding["excluded_inputs"].append(
        "target-array payload values remain for the existing trainer's frozen index verification; no cache or teacher targets are recomputed"
    )
    for row in new_targets:
        if any(item["logical_path"] == row["logical_path"] for item in binding["external_mounts"]):
            raise ValueError(f"R3 binding already includes {row['logical_path']}")
        binding["external_mounts"].append(
            {key: row[key] for key in ("dataset", "filename", "logical_path", "sha256", "size", "version_expected")}
        )

    jobs_dir = PACKET_R4 / "jobs"
    job_specs = []
    replacements: dict[str, bytes] = {}
    for old_job in r3_source["profile_jobs"]:
        old_path = ROOT / old_job["path"]
        new_path = jobs_dir / old_path.name
        spec = r4_job(old_path, new_path, new_targets)
        job_specs.append((new_path, spec))
        replacements[f"kaggle_specs/{spec['name']}.json"] = (
            json.dumps(spec, sort_keys=True) + "\n"
        ).encode()

    BUNDLE_R4.mkdir(parents=True)
    for name in ("dataset-metadata.json", "kernel/kernel-metadata.json"):
        target = BUNDLE_R4 / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BUNDLE_R3 / name, target)
    r4_archive = BUNDLE_R4 / "bundle.wnn"
    write_archive(BUNDLE_R3 / "bundle.wnn", r4_archive, replacements)

    mounts = external_map(binding)
    manifest = copy.deepcopy(r3_manifest)
    manifest["archive_sha256"] = sha256_file(r4_archive)
    manifest["archive_bytes"] = r4_archive.stat().st_size
    manifest["external_mounts"] = mounts
    (BUNDLE_R4 / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    kernel_source = (BUNDLE_R3 / "kernel/kernel.py").read_text(encoding="utf-8")
    old_map = external_map(r3_binding)
    old_assignment = f"EXPECTED_EXTERNAL_MOUNTS = {old_map!r}\n"
    if kernel_source.count(old_assignment) != 1:
        raise ValueError("R3 kernel mount assignment is not the exact sealed literal")
    kernel_source = patch_kernel_mounts(kernel_source, mounts)
    kernel_source = kernel_source.replace(
        f"EXPECTED_ARCHIVE_SHA256 = {r3_seal['bundle']['archive_sha256']!r}",
        f"EXPECTED_ARCHIVE_SHA256 = {manifest['archive_sha256']!r}",
        1,
    )
    if f"EXPECTED_ARCHIVE_SHA256 = {manifest['archive_sha256']!r}" not in kernel_source:
        raise ValueError("R4 outer kernel archive identity was not updated")
    (BUNDLE_R4 / "kernel/kernel.py").write_text(kernel_source, encoding="utf-8")

    write_json(INPUT_BINDING, binding)
    changed_paths = [row["logical_path"] for row in new_targets]
    mount_receipt = {
        "schema": "wnn.student-shape-pair-kernel-mount-successor.r4",
        "status": "outer_kernel_mount_map_and_bundle_identity_verified",
        "provider_action_performed": False,
        "r3_kernel_path": str((BUNDLE_R3 / "kernel/kernel.py").relative_to(ROOT)),
        "r3_kernel_sha256": sha256_file(BUNDLE_R3 / "kernel/kernel.py"),
        "r4_kernel_path": str((BUNDLE_R4 / "kernel/kernel.py").relative_to(ROOT)),
        "r4_kernel_sha256": sha256_file(BUNDLE_R4 / "kernel/kernel.py"),
        "changed_logical_paths": changed_paths,
        "changed_field_count": 2,
        "archive_identity_updated": True,
        "outer_deadline_seconds": 1800,
        "scientific_recipe_or_argv_changed": False,
    }
    write_json(MOUNT_RECEIPT, mount_receipt)
    build_receipt = {
        "schema": "wnn.student-shape-pair-offline-bundle-successor.r4",
        "status": "two_required_fold0_target_cache_mounts_added",
        "provider_action_performed": False,
        "r3_kernel_log_path": str(R3_FAILURE.relative_to(ROOT)),
        "r3_kernel_log_sha256": sha256_file(R3_FAILURE),
        "r3_archive_sha256": sha256_file(BUNDLE_R3 / "bundle.wnn"),
        "r4_archive_sha256": sha256_file(r4_archive),
        "r4_archive_bytes": r4_archive.stat().st_size,
        "r4_archive_byte_identical_to_r3": False,
        "code_members_copied_from_r3_without_change": len(r3_manifest["files"]),
        "profile_specs_rewritten_only_for_inputs_and_deferred_identity": [
            {
                "name": spec["name"],
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "argv_unchanged_from_r3": True,
                "added_inputs": changed_paths,
            }
            for path, spec in job_specs
        ],
        "target_mounts": new_targets,
        "payload_copy_or_recompute": False,
        "recipe_arguments_changed": False,
        "r3_source_manifest_sha256": sha256_file(r3_manifest_path),
        "r4_source_manifest_sha256": sha256_file(BUNDLE_R4 / "bundle-manifest.json"),
        "r3_kernel_sha256": sha256_file(BUNDLE_R3 / "kernel/kernel.py"),
        "r4_kernel_sha256": sha256_file(BUNDLE_R4 / "kernel/kernel.py"),
        "kernel_mount_receipt_sha256": sha256_file(MOUNT_RECEIPT),
        "r3_archived_source_code_members_unchanged": True,
        "r3_failed_profile_preserved": True,
    }
    write_json(BUILD_RECEIPT, build_receipt)

    source = copy.deepcopy(r3_source)
    source["schema"] = "wnn.student-shape-pair-source-binding.r4"
    source["input_binding_path"] = str(INPUT_BINDING.relative_to(ROOT))
    source["input_binding_sha256"] = sha256_file(INPUT_BINDING)
    source["bundle_successor_path"] = str(Path(__file__).resolve().relative_to(ROOT))
    source["bundle_successor_sha256"] = sha256_file(Path(__file__).resolve())
    source["bundle_successor_command"] = ["/usr/bin/python3", source["bundle_successor_path"]]
    source["bundle_successor_receipt_path"] = str(BUILD_RECEIPT.relative_to(ROOT))
    source["bundle_successor_receipt_sha256"] = sha256_file(BUILD_RECEIPT)
    source["kernel_mount_successor_receipt_path"] = str(MOUNT_RECEIPT.relative_to(ROOT))
    source["kernel_mount_successor_receipt_sha256"] = sha256_file(MOUNT_RECEIPT)
    source["offline_bundle_command"] = source["bundle_successor_command"]
    source["profile_jobs"] = [
        {
            **old_job,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "input_count": len(spec["inputs"]),
            "external_target_cache_mounts": changed_paths,
        }
        for old_job, (path, spec) in zip(r3_source["profile_jobs"], job_specs, strict=True)
    ]
    source["source_successor"] = {
        "prior_revision": "R3 profile ended before any CUDA update because fold0 split loader inputs were incomplete",
        "added_logical_paths": changed_paths,
        "dataset": FOLD0_DATASET,
        "version_expected": 1,
        "portable_registry_path": "research/campaign_2026-10-04/portable_inputs.json",
        "portable_registry_sha256": sha256_file(PORTABLE),
        "split_manifest_path": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "split_manifest_sha256": sha256_file(SPLIT),
        "profile_cli_argv_identical_to_r3": True,
        "source_code_closure_identical_to_r3": True,
        "input_binding_only_repair": True,
    }
    write_json(SOURCE_BINDING, source)

    print(
        json.dumps(
            {
                "status": build_receipt["status"],
                "r4_archive_sha256": build_receipt["r4_archive_sha256"],
                "new_target_mounts": changed_paths,
                "job_hashes": [row["sha256"] for row in source["profile_jobs"]],
                "argv_unchanged": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
