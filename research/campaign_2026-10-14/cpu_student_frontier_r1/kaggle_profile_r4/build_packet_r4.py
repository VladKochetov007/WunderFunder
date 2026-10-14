from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKET_DIR = Path(__file__).resolve().parent
R1_DIR = PACKET_DIR.parent / "kaggle_profile_r3"
PROFILE_ID = "vladkochetov07/wnn-qat-f0-s4-profile-r4"
PROFILE_TIMEOUT_SECONDS = 1_800
FINALIZATION_RESERVE_SECONDS = 300
ASSET_DATASETS = [
    "vladkochetov07/wnn-c20261004-fold0-assets",
    "vladkochetov07/wnn-qat-f0-s4-assets-r1",
]
PROFILE_SPEC_NAMES = [
    "preflight.spec.json",
    "qat_deploy_f0_s4_100pct_fp32_profile_r1.spec.json",
    "qat_deploy_f0_s4_100pct_uniform_w4_profile_r1.spec.json",
    "qat_deploy_f0_s4_100pct_mixed_w4w8_profile_r1.spec.json",
]


def sha256_file(path: Path) -> str:
    """Hash bounded packet source and metadata files in streaming blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically write stable packet metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build(stage_root: Path) -> dict[str, Any]:
    """Build an offline profile-only successor without touching asset payloads."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from kaggle_bundle import build_bundle
    from kaggle_common import canonical_sha256

    stage_root.mkdir(parents=True, exist_ok=True)
    frozen_root = stage_root / "source"
    previous_stage = Path(
        "/home/vlad/development/wunder/qk14r3b/profile_upload"
    )
    old_manifest = json.loads((previous_stage / "bundle-manifest.json").read_text())
    old_archive = previous_stage / "bundle.wnn"
    if (
        sha256_file(old_archive)
        != "6cfc0b87762e9ad91be1c9528ee767cb5c021364f62620d8c5b63e04cf383a6a"
    ):
        raise ValueError("R3 frozen source archive identity differs")
    if old_manifest["expanded_bytes"] > 20_000_000:
        raise ValueError("R3 source staging exceeds bounded expected size")
    frozen_root.mkdir(exist_ok=False)
    with tarfile.open(old_archive, "r:gz") as archive:
        archive.extractall(frozen_root, filter="data")
    shutil.copyfile(
        REPO_ROOT / "scripts/kaggle_qat_kernel_r1.py",
        frozen_root / "scripts/kaggle_qat_kernel_r1.py",
    )
    specs_dir = PACKET_DIR / "specs"
    specs_dir.mkdir(exist_ok=True)
    kernel_source = REPO_ROOT / "scripts/kaggle_qat_kernel_r1.py"
    kernel_sha = sha256_file(kernel_source)
    prepared_specs: list[Path] = []
    for name in PROFILE_SPEC_NAMES:
        source = R1_DIR / "specs" / name
        spec = json.loads(source.read_text(encoding="utf-8"))
        if name != "preflight.spec.json":
            materialized = {
                "cache/clean_t3_f0_h390_t3r5_a9_r2/" + filename
                for filename in ("metadata.json", "main_homogeneous_train.npy", "main_homogeneous_valid.npy")
            }
            deferred = spec["environment"]["WNN_DEFERRED_IDENTITY_PATHS"].split(",")
            if not materialized.issubset(deferred):
                raise ValueError("candidate does not declare all three materialized cache files")
            spec["environment"]["WNN_DEFERRED_IDENTITY_PATHS"] = ",".join(
                path for path in deferred if path not in materialized
            )
        matches = [
            item for item in spec["code"] if item["path"] == "scripts/kaggle_qat_kernel_r1.py"
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one bootstrap descriptor in {name}")
        matches[0].update({"size": kernel_source.stat().st_size, "sha256": kernel_sha})
        spec["code_sha256"] = canonical_sha256(sorted(spec["code"], key=lambda item: item["path"]))
        destination = specs_dir / name
        write_json(destination, spec)
        prepared_specs.append(destination)

    previous_manifest_path = (
        Path("/home/vlad/development/wunder/qatk4a/qat-f0-s4-kaggle-r1")
        / "profile_upload/bundle-manifest.json"
    )
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    external: list[str] = []
    for logical, mapping in sorted(previous_manifest["external_mounts"].items()):
        if isinstance(mapping, dict):
            external.append(f"{logical}={mapping['dataset']}:{mapping['filename']}")
        else:
            external.append(f"{logical}={mapping}")

    destination = stage_root / "profile_upload"
    build_bundle(
        frozen_root,
        prepared_specs,
        destination,
        PROFILE_ID,
        external,
        ASSET_DATASETS,
    )
    manifest_path = destination / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bootstrap_source = kernel_source.read_text(encoding="utf-8")
    replacements = {
        'BUNDLE_SLUG = ""': f"BUNDLE_SLUG = {PROFILE_ID.split('/')[1]!r}",
        'EXPECTED_ARCHIVE_SHA256 = ""': f"EXPECTED_ARCHIVE_SHA256 = {manifest['archive_sha256']!r}",
        "EXPECTED_EXTERNAL_MOUNTS = None": f"EXPECTED_EXTERNAL_MOUNTS = {manifest['external_mounts']!r}",
        "KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200": (
            f"KERNEL_PROCESS_WALL_SECONDS_MAX = {PROFILE_TIMEOUT_SECONDS}"
        ),
    }
    for old, new in replacements.items():
        if bootstrap_source.count(old) != 1:
            raise ValueError(f"bootstrap binding marker missing or ambiguous: {old}")
        bootstrap_source = bootstrap_source.replace(old, new)
    kernel_dir = destination / "kernel"
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(bootstrap_source, encoding="utf-8")
    kernel_metadata_path = kernel_dir / "kernel-metadata.json"
    kernel_metadata = json.loads(kernel_metadata_path.read_text(encoding="utf-8"))
    kernel_metadata.update(
        {
            "id": PROFILE_ID + "-runner",
            "title": PROFILE_ID.split("/")[1] + " runner",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4",
        }
    )
    write_json(kernel_metadata_path, kernel_metadata)
    if kernel_path.stat().st_size >= 1_000_000:
        raise ValueError("Kaggle bootstrap exceeds the conservative 1 MB limit")

    binding = {
        "schema": "qat-kaggle-profile-entrypoint-binding-r4",
        "profile": {
            "dataset": PROFILE_ID,
            "kernel": PROFILE_ID + "-runner",
            "bundle_sha256": manifest["archive_sha256"],
            "bootstrap_path": str(kernel_path),
            "bootstrap_bytes": kernel_path.stat().st_size,
            "bootstrap_sha256": sha256_file(kernel_path),
            "template_path": "scripts/kaggle_qat_kernel_r1.py",
            "template_sha256": kernel_sha,
            "provider_timeout_seconds": PROFILE_TIMEOUT_SECONDS,
            "runner_finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "cache_handoff": "verified mounted inputs -> targeted file-cache advice -> stream-copy three pinned cache files into registered root -> full runner hash verification for materialized regular cache files + unchanged consumer/admission checks",
            "asset_dataset_version": "vladkochetov07/wnn-qat-f0-s4-assets-r1/version1",
        },
    }
    binding_path = PACKET_DIR / "BUNDLE_BINDING_R4.json"
    write_json(binding_path, binding)
    packet = {
        "schema": "qat-kaggle-f0-s4-profile-packet-r4",
        "status": "offline_bundle_built_independent_review_pending",
        "predecessor": "vladkochetov07/wnn-qat-f0-s4-profile-r3/version1 (ERROR: copied regular files incorrectly left in deferred-symlink input list)",
        "profile_bundle": PROFILE_ID,
        "profile_kernel": PROFILE_ID + "-runner",
        "asset_dataset_version_reused": "vladkochetov07/wnn-qat-f0-s4-assets-r1/version1",
        "bundle_binding": binding_path.name,
        "profile_bundle_archive_sha256": manifest["archive_sha256"],
        "preflight_barrier": True,
        "candidate_admission_gate_changed": False,
        "per_worker_ram_reservation_bytes": 8_589_934_592,
        "attempt_policy": {
            "provider_attempt_limit": 1,
            "automatic_retries": 0,
            "provider_mutations_by_builder": 0,
            "root_fresh_admission_required_before_push": True,
        },
        "candidate_jobs": [json.loads(path.read_text())["name"] for path in prepared_specs[1:]],
        "provider_actions_performed": False,
        "large_target_arrays_rehashed": False,
        "build_stage": str(stage_root),
    }
    write_json(PACKET_DIR / "PACKET_R4.json", packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.stage_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
