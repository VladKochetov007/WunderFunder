"""Build the source-only V3 profile packet and its private Kaggle transport staging."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from kaggle_assets import prepare


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
V2_PACKET = CAMPAIGN / "gru512_768_muon_capacity_v2_20261008/packet"
PACKET = CAMPAIGN / "gru512_768_muon_capacity_v3_profile_20261008/packet"
SOURCE_DATASET_SLUG = "vladkochetov07/e529-gru512-768-muon-capacity-v3-source-20261008"
KERNEL_SLUG = "vladkochetov07/e529-gru512-768-muon-capacity-v3-profile-20261008-runner"
RECIPE_NAME = "GRU512_768_MUON_PROFILE_RECIPE_V3.json"
AUTH_NAME = "ROOT_PROFILE_AUTHORIZATION.json"
REVIEW_NAME = "ROOT_PROFILE_SOURCE_REVIEW.json"
PACKET_FILES = (
    "capacity_worker.py",
    "kernel.py",
    "train_entry.py",
    "verified_digest_handoff.py",
    "README.md",
    "kernel-metadata.json",
    "source/source_archive.tar",
    "source/source_archive_manifest.json",
    "fold0.json",
    "portable_inputs.json",
    "prior_input_identity_v2.json",
    RECIPE_NAME,
    AUTH_NAME,
    REVIEW_NAME,
)

DRIVER_TEMPLATE = '''from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath

SOURCE_DATASET = "__SOURCE_DATASET_SLUG__"
EXPECTED_ASSET_MANIFEST_SHA256 = "__ASSET_MANIFEST_SHA256__"
PACKET_ID = "E529_H232_GRU512_768_MUON_SEED9_PROFILE_V3_20261008"
REQUIRED_PACKET_FILES = __REQUIRED_PACKET_FILES__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_dataset_root(input_root: Path) -> Path:
    """Find only the declared source dataset, ignoring same-named other manifests."""
    owner, slug = SOURCE_DATASET.split("/", 1)
    roots = [input_root / "datasets" / owner / slug, input_root / slug]
    matches = [root for root in roots if (root / "assets-manifest.json").is_file()]
    if len(matches) != 1:
        raise ValueError("expected exactly one mount for the bound private source dataset")
    return matches[0]


def _safe_relative(raw: str) -> PurePosixPath:
    relative = PurePosixPath(raw)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\\\" in raw:
        raise ValueError("source packet contains an unsafe relative path")
    return relative


def materialize_packet(input_root: Path, packet_root: Path) -> None:
    """Verify the exact content-addressed private dataset before importing its worker."""
    dataset_root = source_dataset_root(input_root)
    manifest_path = dataset_root / "assets-manifest.json"
    if sha256_file(manifest_path) != EXPECTED_ASSET_MANIFEST_SHA256:
        raise ValueError("private source dataset manifest identity changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != SOURCE_DATASET or manifest.get("private") is not True:
        raise ValueError("private source dataset identity or visibility is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("private source dataset file list is malformed")
    by_path = {}
    for entry in entries:
        relative = _safe_relative(str(entry.get("path", "")))
        if relative.as_posix() in by_path:
            raise ValueError("private source dataset repeats a packet file")
        by_path[relative.as_posix()] = entry
    if set(by_path) != set(REQUIRED_PACKET_FILES):
        raise ValueError("private source dataset file set differs from the reviewed packet")

    temporary = packet_root.with_name(packet_root.name + ".partial")
    if packet_root.exists() or temporary.exists():
        raise FileExistsError("packet materialization target already exists")
    temporary.mkdir(parents=True)
    completed = False
    try:
        for logical in REQUIRED_PACKET_FILES:
            entry = by_path[logical]
            expected_sha = str(entry["sha256"])
            if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
                raise ValueError("private source dataset contains an invalid SHA-256")
            filename = str(entry["filename"])
            if filename != f"{expected_sha}.wnn" or PurePosixPath(filename).name != filename:
                raise ValueError("private source dataset filename is not content addressed")
            source = dataset_root / filename
            if source.is_symlink() or not source.is_file() or source.stat().st_size != int(entry["size"]):
                raise ValueError("private source dataset member size or file type changed")
            relative = _safe_relative(logical)
            target = temporary.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            copied = 0
            with source.open("rb") as reader, target.open("xb") as writer:
                while block := reader.read(1 << 20):
                    digest.update(block)
                    writer.write(block)
                    copied += len(block)
            if copied != int(entry["size"]) or digest.hexdigest() != expected_sha:
                raise ValueError("private source dataset member digest changed during copy")
        temporary.replace(packet_root)
        completed = True
    finally:
        if not completed:
            shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WNN_THREADS"] = "4"
    os.environ["WNN_LOADER_WORKERS"] = "0"
    os.environ["WNN_LOADER_RAM_GB"] = "0.30"
    os.environ["WNN_GPU_FRACTION"] = "0.75"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["PYTHONUNBUFFERED"] = "1"
    packet_root = Path("/kaggle/working/gru512_768_muon_capacity_v3_profile_packet")
    materialize_packet(Path("/kaggle/input"), packet_root)
    sys.path.insert(0, str(packet_root))
    worker = importlib.import_module("capacity_worker")
    if worker.PACKET_ID != PACKET_ID:
        raise ValueError("mounted capacity worker has a different packet identity")
    receipt = worker.run(packet_root)
    print(
        "GRU_CAPACITY_TERMINAL "
        f"status={receipt['status']} fit_started={receipt['fit_started']} "
        f"worker_seconds={receipt['worker_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def bound_file(name: str) -> dict[str, object]:
    path = PACKET / name
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"V3 packet member is missing or not regular: {name}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def make_kernel_metadata() -> dict[str, object]:
    return {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": [
            SOURCE_DATASET_SLUG,
            "vladkochetov07/wunder-funder",
            "vladkochetov07/wnn-c20261004-fold0-assets",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "id": KERNEL_SLUG,
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "title": "GRU512/768 Muon V3 profile only",
    }


def make_authorization(
    recipe_sha: str, bindings: dict[str, dict[str, object]]
) -> dict[str, object]:
    return {
        "schema": "wnn-gru-capacity-root-conditional-authorization-v2",
        "packet_id": "E529_H232_GRU512_768_MUON_SEED9_PROFILE_V3_20261008",
        "status": "pending_root_review",
        "authorized": False,
        "profile_authorized": False,
        "conditional_fit_authorized": False,
        "fit_arm_ids": [],
        "selected_batch": None,
        "max_worker_seconds": 5200,
        "recipe_sha256": recipe_sha,
        "source_archive_sha256": bindings["source/source_archive.tar"]["sha256"],
        "source_manifest_sha256": bindings["source/source_archive_manifest.json"]["sha256"],
        "portable_manifest_sha256": bindings["portable_inputs.json"]["sha256"],
        "split_manifest_sha256": bindings["fold0.json"]["sha256"],
        "prior_input_identity_sha256": bindings["prior_input_identity_v2.json"]["sha256"],
        "kernel_metadata_sha256": bindings["kernel-metadata.json"]["sha256"],
        "worker_sha256": bindings["capacity_worker.py"]["sha256"],
        "entry_sha256": bindings["kernel.py"]["sha256"],
        "train_entry_sha256": bindings["train_entry.py"]["sha256"],
        "digest_handoff_sha256": bindings["verified_digest_handoff.py"]["sha256"],
        "root_review_receipt": REVIEW_NAME,
        "root_review_receipt_sha256": sha256_file(PACKET / REVIEW_NAME),
        "notes": "Profile only after independent source review and fresh provider admission; fit remains disabled.",
    }


def build_packet() -> None:
    if (PACKET / "source_dataset_upload").exists() or (PACKET / "kernel_upload").exists():
        raise FileExistsError(
            "V3 transport output already exists; preserve it and build a new identity"
        )
    v2_recipe = json.loads((V2_PACKET / "GRU512_768_MUON_CAPACITY_RECIPE_V2.json").read_text())
    metadata = make_kernel_metadata()
    write_json(PACKET / "kernel-metadata.json", metadata)
    (PACKET / "README.md").write_text(
        "# GRU512/768 Muon capacity V3 profile-only packet\n\n"
        "This successor preserves V2's seed, clean fold, objective, sampler, model widths, and profile shapes. "
        "It fixes physical-GPU selection/telemetry, source-qualified fit verification, separate trainer-saved "
        "checkpoint identity, and Kaggle transport. Initial authorization is profile-only; no fit can start "
        "from the pending receipt. This local private-dataset staging has not been uploaded or pushed.\n\n"
        "The small Kaggle codefile verifies the exact private source-dataset manifest, materializes the packet "
        "under `/kaggle/working`, then imports the worker from that verified directory. Raw train/valid and fold0 "
        "assets remain separate mounted datasets; the source dataset contains only packet files.\n",
        encoding="utf-8",
    )
    for old_name in (
        "GRU512_768_MUON_CAPACITY_RECIPE_V2.json",
        "ROOT_CONDITIONAL_FIT_AUTHORIZATION.json",
        "PACKET_BUILD_RECEIPT.json",
    ):
        old_path = PACKET / old_name
        if old_path.exists():
            old_path.unlink()
    write_json(
        PACKET / REVIEW_NAME,
        {
            "schema": "wnn-gru-capacity-independent-profile-source-review-v1",
            "packet_id": "E529_H232_GRU512_768_MUON_SEED9_PROFILE_V3_20261008",
            "status": "pending_independent_review",
            "authorized": False,
        },
    )

    bindings = {
        name: bound_file(name)
        for name in PACKET_FILES
        if name not in {RECIPE_NAME, AUTH_NAME, REVIEW_NAME}
    }
    recipe = v2_recipe
    recipe.update(
        {
            "packet_id": "E529_H232_GRU512_768_MUON_SEED9_PROFILE_V3_20261008",
            "status": "profile-only-source-prepared-pending-root-review-and-admission",
        }
    )
    recipe["source"]["independent_source_commit"] = git_commit()
    recipe["source"]["lineage"] = (
        "V2 seed-9 scientific recipe with V3 physical-GPU, producer-verifier and Kaggle transport corrections"
    )
    recipe["source"]["file_bindings"] = bindings
    recipe["runtime"]["requested_kernel"] = KERNEL_SLUG
    recipe["runtime"]["requested_source_dataset"] = SOURCE_DATASET_SLUG
    recipe["runtime"]["initial_execution_scope"] = (
        "profile-only; conditional fit authorization is false"
    )
    recipe["authorization"] = {
        "profile": "requires independent root source review and fresh Kaggle quota admission",
        "conditional_fit": "disabled in this profile-only packet; requires a separately authorized successor",
        "grading": False,
        "provider_push": False,
    }
    write_json(PACKET / RECIPE_NAME, recipe)
    recipe_sha = sha256_file(PACKET / RECIPE_NAME)
    write_json(PACKET / AUTH_NAME, make_authorization(recipe_sha, bindings))

    upload_root = PACKET / "source_dataset_upload"
    prepare(PACKET, list(PACKET_FILES), upload_root, SOURCE_DATASET_SLUG)
    manifest_path = upload_root / "assets-manifest.json"
    manifest_sha = sha256_file(manifest_path)
    driver = DRIVER_TEMPLATE.replace("__SOURCE_DATASET_SLUG__", SOURCE_DATASET_SLUG)
    driver = driver.replace("__ASSET_MANIFEST_SHA256__", manifest_sha)
    driver = driver.replace("__REQUIRED_PACKET_FILES__", repr(PACKET_FILES))
    kernel_upload = PACKET / "kernel_upload"
    kernel_upload.mkdir()
    driver_path = kernel_upload / "kernel.py"
    driver_path.write_text(driver, encoding="utf-8")
    write_json(kernel_upload / "kernel-metadata.json", metadata)
    upload_driver_sha = sha256_file(driver_path)
    write_json(
        PACKET / "PACKET_BUILD_RECEIPT.json",
        {
            "packet_id": recipe["packet_id"],
            "status": recipe["status"],
            "source_snapshot_commit": recipe["source"]["independent_source_commit"],
            "recipe_sha256": recipe_sha,
            "packet_source_manifest_sha256": bindings["source/source_archive_manifest.json"][
                "sha256"
            ],
            "packet_source_archive_sha256": bindings["source/source_archive.tar"]["sha256"],
            "private_source_dataset_slug": SOURCE_DATASET_SLUG,
            "private_source_dataset_manifest_sha256": manifest_sha,
            "private_source_dataset_manifest_bytes": manifest_path.stat().st_size,
            "private_source_dataset_metadata_sha256": sha256_file(
                upload_root / "dataset-metadata.json"
            ),
            "uploaded_kernel_driver_sha256": upload_driver_sha,
            "uploaded_kernel_metadata_sha256": sha256_file(kernel_upload / "kernel-metadata.json"),
            "root_profile_source_review_sha256": sha256_file(PACKET / REVIEW_NAME),
            "uploaded_kernel_code_file": "kernel.py",
            "raw_and_fold_asset_mounts": metadata["dataset_sources"][1:],
            "data_or_provider_reads": False,
            "upload_or_push_performed": False,
            "conditional_fit_authorized": False,
        },
    )
    print(
        json.dumps(
            {
                "packet": str(PACKET),
                "recipe_sha256": recipe_sha,
                "source_dataset_manifest_sha256": manifest_sha,
                "driver_sha256": upload_driver_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    build_packet()
