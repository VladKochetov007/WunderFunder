from __future__ import annotations

import ast
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r4"
PACKET_R3 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r3"
BUNDLE = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r4"
BUNDLE_R3 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r3"
INPUT_BINDING = PACKET / "INPUT_BINDING_R4.json"
SOURCE_BINDING = PACKET / "SOURCE_BINDING_R4.json"
BUILD_RECEIPT = PACKET / "OFFLINE_BUNDLE_BUILD_R4.json"
MOUNT_RECEIPT = PACKET / "KERNEL_MOUNT_SUCCESSOR_R4.json"
SEAL = PACKET / "GENERATED_KERNEL_SEAL_R4.json"
OUTPUT = PACKET / "OFFLINE_SPEC_PREFLIGHT_R4.json"
SPLIT = MAIN / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PORTABLE = MAIN / "research/campaign_2026-10-04/portable_inputs.json"
TARGET_PATHS = {"cache/train_targets.npz", "cache/valid_targets.npz"}
PROFILE_SHAPES = ((256, 2), (512, 1))


def sha256_bytes(data: bytes) -> str:
    """Hash one bounded archive member or JSON byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash an offline packet artifact in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, maximum: int = 4_000_000) -> dict[str, Any]:
    """Read only bounded JSON metadata."""
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError(f"refusing oversized or empty JSON: {path} ({size} bytes)")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def external_map(binding: dict[str, Any]) -> dict[str, object]:
    """Convert R4 input descriptors into the kernel literal mount map."""
    return {
        row["logical_path"]: (
            row["filename"]
            if row["dataset"] == "vladkochetov07/wunder-funder"
            else {"dataset": row["dataset"], "filename": row["filename"]}
        )
        for row in binding["external_mounts"]
    }


def kernel_map(source: str) -> dict[str, object]:
    """Read the static bootstrap mount assignment without executing Kaggle code."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_EXTERNAL_MOUNTS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return value
    raise ValueError("kernel has no literal external-mount map")


def archive_members(path: Path) -> dict[str, bytes]:
    """Read the small source archive with strict member and total-byte bounds."""
    result: dict[str, bytes] = {}
    total = 0
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 200:
            raise ValueError("profile archive exceeds 200-member bound")
        for member in members:
            if (
                not member.isfile()
                or member.name in result
                or member.name.startswith("/")
                or ".." in Path(member.name).parts
                or member.size < 0
                or member.size > 2_000_000
            ):
                raise ValueError(f"unsupported profile archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"archive member missing payload: {member.name}")
            payload = stream.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError(f"archive member byte count changed: {member.name}")
            total += len(payload)
            if total > 5_000_000:
                raise ValueError("profile archive expanded bytes exceed 5 MB check bound")
            result[member.name] = payload
    return result


def main() -> None:
    """Check the complete R4 consumer input set and seal its offline packet."""
    if SEAL.exists() or OUTPUT.exists():
        raise FileExistsError("R4 offline evidence already exists; preserve it")
    binding = read_json(INPUT_BINDING)
    source = read_json(SOURCE_BINDING)
    build = read_json(BUILD_RECEIPT)
    mount = read_json(MOUNT_RECEIPT)
    r3_binding = read_json(PACKET_R3 / "INPUT_BINDING_R3.json")
    r3_source = read_json(PACKET_R3 / "SOURCE_BINDING_R3.json")
    r3_manifest = read_json(BUNDLE_R3 / "bundle-manifest.json")
    manifest = read_json(BUNDLE / "bundle-manifest.json")
    split = read_json(SPLIT, maximum=2_000_000)
    portable = read_json(PORTABLE, maximum=16_000)

    if binding.get("provider_action_performed") is not False or source.get("provider_action_performed") is not False:
        raise ValueError("offline R4 packet must not claim provider actions")
    if sha256_file(SPLIT) != "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d":
        raise ValueError("fold0 split identity differs from the pinned source")
    if sha256_file(PORTABLE) != "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c":
        raise ValueError("portable source registry differs from the pinned source")
    if source["recipe_contract_sha256"] != r3_source["recipe_contract_sha256"]:
        raise ValueError("R4 changed the profile recipe contract")
    if source["consumer_source_sha256"] != r3_source["consumer_source_sha256"]:
        raise ValueError("R4 changed the frozen consumer source")
    if sha256_file(BUNDLE_R3 / "bundle.wnn") != read_json(PACKET_R3 / "GENERATED_KERNEL_SEAL_R3_SEALED.json")["bundle"]["archive_sha256"]:
        raise ValueError("R3 parent archive differs from its frozen seal")
    if build["r3_kernel_log_sha256"] != sha256_file(PACKET_R3 / "PROVIDER_KERNEL_LOG_R3.txt"):
        raise ValueError("R4 does not bind the preserved R3 terminal failure")

    r3_mounts = {row["logical_path"]: row for row in r3_binding["external_mounts"]}
    r4_mounts = {row["logical_path"]: row for row in binding["external_mounts"]}
    if set(r4_mounts) != set(r3_mounts) | TARGET_PATHS:
        raise ValueError("R4 mount set differs from R3 by more than two required target caches")
    if any(r4_mounts[path] != row for path, row in r3_mounts.items()):
        raise ValueError("R4 changed an inherited R3 input descriptor")
    if not TARGET_PATHS <= set(split["sources"]):
        raise ValueError("fold0 split does not require the declared target cache paths")
    if not set(split["sources"]) <= set(r4_mounts):
        missing = sorted(set(split["sources"]) - set(r4_mounts))
        raise ValueError(f"fold0 split consumer paths remain unmounted: {missing}")

    target_rows = [r4_mounts[name] for name in sorted(TARGET_PATHS)]
    for row in target_rows:
        identity = split["sources"][row["logical_path"]]
        portable_row = portable["sources"][row["logical_path"]]
        if portable_row["original_identity"] != identity:
            raise ValueError("portable registry target identity differs from fold0 split")
        if (row["sha256"], row["size"], row["filename"], row["dataset"], row["version_expected"]) != (
            portable_row["sha256"],
            identity["bytes"],
            f"{portable_row['sha256']}.wnn",
            "vladkochetov07/wnn-c20261004-fold0-assets",
            1,
        ):
            raise ValueError(f"R4 target asset descriptor differs from producer registry: {row['logical_path']}")

    if len(source["profile_jobs"]) != 2:
        raise ValueError("R4 must retain the two existing shape jobs")
    archive_r3 = archive_members(BUNDLE_R3 / "bundle.wnn")
    archive_r4 = archive_members(BUNDLE / "bundle.wnn")
    if set(archive_r3) != set(archive_r4):
        raise ValueError("R4 archive changed member closure")
    expected_mount_map = external_map(binding)
    if manifest["external_mounts"] != expected_mount_map:
        raise ValueError("R4 bundle mount map differs from the input binding")
    r3_manifest_mounts = external_map(r3_binding)
    changed_mounts = [path for path in r3_manifest_mounts if r3_manifest_mounts[path] != expected_mount_map.get(path)]
    if changed_mounts or set(expected_mount_map) - set(r3_manifest_mounts) != TARGET_PATHS:
        raise ValueError("outer mount successor did not add exactly the two target cache mounts")
    kernel_path = BUNDLE / "kernel/kernel.py"
    kernel_source = kernel_path.read_text(encoding="utf-8")
    if kernel_map(kernel_source) != expected_mount_map:
        raise ValueError("R4 bootstrap does not mount all exact logical input paths")
    if f"EXPECTED_ARCHIVE_SHA256 = {manifest['archive_sha256']!r}" not in kernel_source:
        raise ValueError("R4 bootstrap archive hash differs from its manifest")
    if "KERNEL_PROCESS_WALL_SECONDS_MAX = 1800" not in kernel_source:
        raise ValueError("R4 outer deadline differs from reviewed 1,800-second cap")

    jobs = []
    for index, shape in enumerate(PROFILE_SHAPES):
        row = source["profile_jobs"][index]
        spec_path = ROOT / row["path"]
        spec = read_json(spec_path, maximum=300_000)
        r3_path = ROOT / r3_source["profile_jobs"][index]["path"]
        r3_spec = read_json(r3_path, maximum=300_000)
        input_rows = {item["path"]: item for item in spec["inputs"]}
        if not TARGET_PATHS <= set(input_rows):
            raise ValueError(f"{spec['name']} omits required target caches")
        for target in target_rows:
            descriptor = input_rows[target["logical_path"]]
            if (descriptor["sha256"], descriptor["size"]) != (target["sha256"], target["size"]):
                raise ValueError(f"{spec['name']} target input identity differs")
        if spec["argv"] != r3_spec["argv"]:
            raise ValueError(f"{spec['name']} CLI argv changed from R3")
        expected_deferred = "datasets/train.parquet,datasets/valid.parquet,cache/train_targets.npz,cache/valid_targets.npz"
        if spec["environment"].get("WNN_DEFERRED_IDENTITY_PATHS") != expected_deferred:
            raise ValueError(f"{spec['name']} does not defer target hashing to the strict portable worker check")
        expected = json.loads(archive_r4[f"kaggle_specs/{spec['name']}.json"])
        if expected != spec:
            raise ValueError(f"{spec['name']} embedded bundle spec differs from reviewed JSON")
        code_paths = {item["path"] for item in spec["code"]}
        if not {"scripts/train_rnn.py", "src/wnn/splits.py", "src/wnn/portability.py"} <= code_paths:
            raise ValueError("profile closure omits the split and portable target-cache consumer")
        if row["sha256"] != sha256_file(spec_path):
            raise ValueError(f"{spec['name']} source binding job hash differs")
        jobs.append(
            {
                "path": str(spec_path.relative_to(ROOT)),
                "sha256": sha256_file(spec_path),
                "shape": list(shape),
                "name": spec["name"],
                "input_count": len(spec["inputs"]),
                "cli_argv_unchanged": True,
                "required_target_paths_present": sorted(TARGET_PATHS),
                "attempts_requested": 50,
            }
        )

    r3_files = {row["path"]: row for row in r3_manifest["files"]}
    for name, payload in archive_r4.items():
        if name.startswith("kaggle_specs/"):
            continue
        identity = r3_files.get(name)
        if identity is None or (identity["size"], identity["sha256"]) != (len(payload), sha256_bytes(payload)):
            raise ValueError(f"R4 changed frozen R3 code/data bundle member: {name}")
    if manifest["files"] != r3_manifest["files"] or manifest["expanded_bytes"] != r3_manifest["expanded_bytes"]:
        raise ValueError("R4 changed the frozen R3 source closure or expanded source size")
    if build["r4_archive_sha256"] != sha256_file(BUNDLE / "bundle.wnn"):
        raise ValueError("R4 build receipt archive identity differs")
    if mount["r4_kernel_sha256"] != sha256_file(kernel_path):
        raise ValueError("R4 mount receipt kernel identity differs")
    if source["input_binding_sha256"] != sha256_file(INPUT_BINDING):
        raise ValueError("R4 source binding input hash differs")

    seal = {
        "schema": "wnn.student-shape-pair-generated-kernel-seal.r4",
        "status": "offline_r4_bundle_and_two_target_mounts_pinned_no_provider_action",
        "provider_action_performed": False,
        "source_binding_sha256": sha256_file(SOURCE_BINDING),
        "input_binding_sha256": sha256_file(INPUT_BINDING),
        "bundle": {
            "dataset": manifest["dataset"],
            "private": manifest["private"],
            "archive_path": str((BUNDLE / "bundle.wnn").relative_to(ROOT)),
            "archive_bytes": (BUNDLE / "bundle.wnn").stat().st_size,
            "archive_sha256": sha256_file(BUNDLE / "bundle.wnn"),
            "manifest_path": str((BUNDLE / "bundle-manifest.json").relative_to(ROOT)),
            "manifest_sha256": sha256_file(BUNDLE / "bundle-manifest.json"),
            "external_mount_count": len(expected_mount_map),
        },
        "kernel": {
            "path": str(kernel_path.relative_to(ROOT)),
            "sha256": sha256_file(kernel_path),
            "kernel_id": read_json(BUNDLE / "kernel/kernel-metadata.json")["id"],
            "outer_deadline_seconds": 1800,
            "mounts_match_input_binding": True,
        },
        "target_cache_mounts": [
            {key: row[key] for key in ("dataset", "filename", "logical_path", "sha256", "size", "version_expected")}
            for row in target_rows
        ],
        "jobs": jobs,
    }
    SEAL.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "wnn.student-shape-pair-offline-spec-preflight.r4",
        "status": "passed_offline_no_provider_action",
        "provider_action_performed": False,
        "preflight_script_sha256": sha256_file(Path(__file__).resolve()),
        "seal_sha256": sha256_file(SEAL),
        "source_binding_sha256": sha256_file(SOURCE_BINDING),
        "input_binding_sha256": sha256_file(INPUT_BINDING),
        "recipe_contract_sha256": source["recipe_contract_sha256"],
        "r3_failed_profile_log_sha256": sha256_file(PACKET_R3 / "PROVIDER_KERNEL_LOG_R3.txt"),
        "checks": [
            "R3 failure is pinned and remains infrastructure-only: missing split target-cache mounts before any CUDA update",
            "exact fold0 producer split and portable registry identities for train/valid target caches",
            "all source-qualified fold0 split consumer paths are mounted in both profile job declarations and the outer bootstrap map",
            "R4 differs from R3 only by two required target-cache inputs, their worker-deferred whole-file identity check, and the corresponding outer mount map/archive identity",
            "profile CLI argv, optimizer, shape, seed, budget, source closure, and model recipe remain identical to R3",
            "R3 archive members outside the two embedded job JSON documents are byte-identical",
            "worker performs strict portable target-content hash and index verification before training",
            "profile jobs remain profile-only with two 50-attempt jobs and the fixed 1,800-second outer deadline",
        ],
        "target_cache_descriptors": seal["target_cache_mounts"],
        "jobs": jobs,
        "archive": seal["bundle"],
        "kernel": seal["kernel"],
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "job_count": len(jobs), "added_target_mounts": sorted(TARGET_PATHS), "archive_sha256": seal["bundle"]["archive_sha256"], "kernel_sha256": seal["kernel"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
