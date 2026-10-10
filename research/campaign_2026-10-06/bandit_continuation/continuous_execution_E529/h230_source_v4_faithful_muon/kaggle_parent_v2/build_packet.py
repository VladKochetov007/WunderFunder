from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PACKET_ROOT = Path(__file__).resolve().parent
V4_ROOT = PACKET_ROOT.parent
REPO_ROOT = next(
    (parent for parent in PACKET_ROOT.parents if (parent / "specs/README.md").is_file()),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError("could not locate the project root")

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from kaggle_bundle import build_bundle  # noqa: E402
from kaggle_common import canonical_sha256, sha256_file, write_json  # noqa: E402

SOURCE_MANIFEST = V4_ROOT / "H230_FAITHFUL_MUON_SOURCE_MANIFEST_V4_20261008.json"
RECIPE_PATH = V4_ROOT / "H230_STRONG_PARENT_SEED9_RECIPE_V2_20261008.json"
PROFILE_CONTRACT = (
    REPO_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
    "H230_STRONG_PARENT_SEED9_REAL_DATA_PROFILE_CONTRACT_V1_20261008.json"
)
PROFILE_AUTH = (
    REPO_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
    "H230_STRONG_PARENT_REALDATA_PROFILE_ROOT_AUTHORIZATION_20261008_R1.json"
)
PROFILE_RECEIPT = (
    REPO_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
    "H230_STRONG_PARENT_SEED9_REAL_DATA_PROFILE_TERMINAL_RECEIPT_V1_20261008.json"
)
ASSETS_MANIFEST_PATH = REPO_ROOT / "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json"
RAW_DATASET = "vladkochetov07/wunder-funder"
FOLD_ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
CACHE_DATASET = "vladkochetov07/h230-seed9-parent-source-cache-v2"
BUNDLE_DATASET = "vladkochetov07/h230-seed9-parent-fit-v2"
FIT_ATTEMPT_ID = "h230-strong-parent-seed9-kaggle-fit-v2-20261008"
FIT_RUN_NAME = "bd26_h230_strong_parent_seed9_muon_kaggle_v2"
CACHE_ROOT = "cache/bd26_h230_fold0_teacher_family_all_rows_v1"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_manifest_sources(payload_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    source_entries = manifest.get("files")
    if not isinstance(source_entries, list) or len(source_entries) != 57:
        raise ValueError("V4 source manifest must bind exactly 57 files")
    code: dict[str, dict[str, Any]] = {}
    for entry in source_entries:
        rel = entry["path"]
        source = V4_ROOT / rel
        destination = payload_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        observed_size = destination.stat().st_size
        observed_sha = sha256_file(destination)
        if observed_size != entry["bytes"] or observed_sha != entry["sha256"]:
            raise ValueError(f"V4 source member changed while staging: {rel}")
        code[rel] = {"path": rel, "size": observed_size, "sha256": observed_sha}

    additional = [
        ("scripts/kaggle_gpu_runner.py", REPO_ROOT / "scripts/kaggle_gpu_runner.py"),
        ("scripts/kaggle_common.py", REPO_ROOT / "scripts/kaggle_common.py"),
        ("scripts/kaggle_kernel.py", REPO_ROOT / "scripts/kaggle_kernel.py"),
    ]
    for rel, source in additional:
        if rel in code:
            raise ValueError(f"transport helper collides with frozen V4 source: {rel}")
        destination = payload_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        code[rel] = {
            "path": rel,
            "size": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    receipt_files = [
        ("H230_FAITHFUL_MUON_SOURCE_MANIFEST_V4_20261008.json", SOURCE_MANIFEST),
        ("H230_STRONG_PARENT_SEED9_RECIPE_V2_20261008.json", RECIPE_PATH),
        ("H230_STRONG_PARENT_SEED9_REAL_DATA_PROFILE_CONTRACT_V1_20261008.json", PROFILE_CONTRACT),
        ("H230_STRONG_PARENT_REALDATA_PROFILE_ROOT_AUTHORIZATION_20261008_R1.json", PROFILE_AUTH),
        ("H230_STRONG_PARENT_SEED9_REAL_DATA_PROFILE_TERMINAL_RECEIPT_V1_20261008.json", PROFILE_RECEIPT),
    ]
    provenance_root = Path(
        "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    )
    for filename, source in receipt_files:
        rel = (provenance_root / "h230_source_v4_faithful_muon" / filename).as_posix()
        if filename.startswith("H230_STRONG_PARENT_SEED9_REAL_DATA_PROFILE") or filename.startswith(
            "H230_STRONG_PARENT_REALDATA_PROFILE"
        ):
            rel = (provenance_root / filename).as_posix()
        if rel in code:
            raise ValueError(f"provenance member collides with frozen source: {rel}")
        destination = payload_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        code[rel] = {
            "path": rel,
            "size": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    return sorted(code.values(), key=lambda item: item["path"])


def stage_cache_asset(cache_root: Path, destination: Path, metadata_sha256: str) -> dict[str, Any]:
    metadata_path = cache_root / "metadata.json"
    if destination.exists():
        raise FileExistsError(f"cache asset destination already exists: {destination}")
    if sha256_path(metadata_path) != metadata_sha256:
        raise ValueError("cache metadata digest differs from the reviewed cache")
    metadata = json.loads(metadata_path.read_text())
    selected_keys = ("main_source_train", "main_source_valid")
    destination.mkdir(parents=True)
    entries = []
    for key in ("metadata", *selected_keys):
        if key == "metadata":
            source = metadata_path
            rel_path = f"{CACHE_ROOT}/metadata.json"
            digest = metadata_sha256
            expected_size = metadata_path.stat().st_size
        else:
            item = metadata["files"][key]
            rel_path = f"{CACHE_ROOT}/{item['path']}"
            source = cache_root / item["path"]
            digest = item["sha256"]
            expected_size = item["bytes"]
            expected_shape = [10606, 20000, 2] if key.endswith("train") else [1873, 20000, 2]
            if item.get("shape") != expected_shape or item.get("dtype") != "float32":
                raise ValueError(f"selected cache entry has unexpected schema: {key}")
            if item.get("row_groups_sha256") != metadata[f"{key.rsplit('_', 1)[1]}_row_groups_sha256"]:
                raise ValueError(f"selected cache entry has unexpected row order: {key}")
        if source.stat().st_size != expected_size:
            raise ValueError(f"selected cache file size differs: {source}")
        filename = f"{digest}.wnn"
        target = destination / filename
        os.link(source, target)
        source_stat, target_stat = source.stat(), target.stat()
        if (source_stat.st_dev, source_stat.st_ino, source_stat.st_size) != (
            target_stat.st_dev,
            target_stat.st_ino,
            expected_size,
        ):
            raise ValueError(f"staged cache file is not the exact source hardlink: {key}")
        entries.append(
            {
                "filename": filename,
                "path": rel_path,
                "sha256": digest,
                "size": expected_size,
                "staging": "same-inode hardlink; selected large array bytes were not reread locally",
            }
        )
    manifest = {
        "version": 1,
        "dataset": CACHE_DATASET,
        "private": True,
        "files": entries,
        "source_cache_metadata_sha256": metadata_sha256,
        "source_training_scope": metadata["teacher_training_scope"],
        "prediction_coverage": metadata["teacher_prediction_coverage"],
        "included_cache_keys": list(selected_keys),
        "omitted_cache_arrays": [
            key for key in metadata["files"] if key not in selected_keys
        ],
        "omission_reason": "This parent has no family head and uses only main_source; family arrays are required later for a family-head student.",
        "large_array_hash_policy": "SHA-256 values are bound to the already validated immutable cache metadata and prior successful actual-data profile; the Kaggle input gate streams and verifies every mounted input before fit.",
    }
    write_json(destination / "assets-manifest.json", manifest)
    write_json(
        destination / "dataset-metadata.json",
        {
            "id": CACHE_DATASET,
            "title": "H230 seed-9 parent source-cache inputs v1",
            "isPrivate": True,
            "licenses": [{"name": "other"}],
        },
    )
    return manifest


def asset_entry(manifest: dict[str, Any], logical_path: str) -> dict[str, Any]:
    try:
        return next(item for item in manifest["files"] if item["path"] == logical_path)
    except StopIteration as error:
        raise ValueError(f"fold asset is missing: {logical_path}") from error


def build_spec(
    code: list[dict[str, Any]],
    fold_assets: dict[str, Any],
    cache_assets: dict[str, Any],
    recipe: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    external: list[str] = []
    inputs: list[dict[str, Any]] = []
    raw = recipe["data"]
    for split in ("train", "valid"):
        item = raw[split]
        logical = f"datasets/{split}.parquet"
        inputs.append({"path": logical, "size": item["bytes"], "sha256": item["sha256"]})
        external.append(f"{logical}={split}.parquet")

    fold_paths = {
        "cache/train_targets.npz": "cache/train_targets.npz",
        "cache/valid_targets.npz": "cache/valid_targets.npz",
        "artifacts/runs/c20260929_prop_f0/model.pt": "artifacts/runs/c20260929_prop_f0/model.pt",
        "research/campaign_2026-09-29/artifacts/splits/fold0.json": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
    }
    for logical, source_path in fold_paths.items():
        item = asset_entry(fold_assets, source_path)
        inputs.append({"path": logical, "size": item["size"], "sha256": item["sha256"]})
        external.append(f"{logical}={FOLD_ASSET_DATASET}:{item['filename']}")
    fold_manifest = {
        "size": ASSETS_MANIFEST_PATH.stat().st_size,
        "sha256": sha256_file(ASSETS_MANIFEST_PATH),
    }
    fold_manifest_path = ".transport/fold0-assets-manifest.json"
    inputs.append(
        {
            "path": fold_manifest_path,
            "size": fold_manifest["size"],
            "sha256": fold_manifest["sha256"],
        }
    )
    external.append(f"{fold_manifest_path}={FOLD_ASSET_DATASET}:assets-manifest.json")

    cache_paths = {
        f"{CACHE_ROOT}/metadata.json": f"{CACHE_ROOT}/metadata.json",
        f"{CACHE_ROOT}/main_source_train.npy": f"{CACHE_ROOT}/main_source_train.npy",
        f"{CACHE_ROOT}/main_source_valid.npy": f"{CACHE_ROOT}/main_source_valid.npy",
    }
    for logical, source_path in cache_paths.items():
        item = asset_entry(cache_assets, source_path)
        inputs.append({"path": logical, "size": item["size"], "sha256": item["sha256"]})
        external.append(f"{logical}={CACHE_DATASET}:{item['filename']}")
    cache_manifest_path = PACKET_ROOT / "cache_asset_upload/assets-manifest.json"
    cache_manifest = {
        "size": cache_manifest_path.stat().st_size,
        "sha256": sha256_file(cache_manifest_path),
    }
    cache_manifest_logical_path = ".transport/h230-parent-cache-assets-manifest.json"
    inputs.append(
        {
            "path": cache_manifest_logical_path,
            "size": cache_manifest["size"],
            "sha256": cache_manifest["sha256"],
        }
    )
    external.append(
        f"{cache_manifest_logical_path}={CACHE_DATASET}:assets-manifest.json"
    )

    argv = list(recipe["fit_attempt"]["argv"])
    name_index = argv.index("--name") + 1
    argv[name_index] = FIT_RUN_NAME
    cache_index = argv.index("--distill-cache-root") + 1
    argv[cache_index] = CACHE_ROOT
    spec = {
        "version": 1,
        "name": "h230_seed9_parent_fullfit_v1",
        "kind": "fit",
        "argv": ["{python}", *argv],
        "code": code,
        "code_sha256": canonical_sha256(code),
        "inputs": inputs,
        "artifacts": [f"artifacts/runs/{FIT_RUN_NAME}"],
        "limits": {
            "cpus": 2,
            "ram_bytes": 9_000_000_000,
            "timeout_seconds": 5_200,
            "gpu_memory_bytes": 4_300_000_000,
            "gpu_stop_bytes": 12_000_000_000,
        },
        "environment": {
            "PYTHONHASHSEED": "0",
            "WNN_PROJECT_ROOT": "/kaggle/working/experiment",
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "6",
            "WNN_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        },
    }
    return spec, external


def build() -> None:
    for path in (PACKET_ROOT / "payload_root", PACKET_ROOT / "cache_asset_upload", PACKET_ROOT / "bundle"):
        if path.exists():
            raise FileExistsError(f"V1 packet output already exists: {path}")

    recipe = json.loads(RECIPE_PATH.read_text())
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if recipe["source"]["manifest_sha256"] != sha256_file(SOURCE_MANIFEST):
        raise ValueError("recipe source lock no longer matches the V4 manifest")
    if recipe["source"]["manifest_sha256"] != "49a4048f559fb6092e71fa2b447069fbce4956bdf8f062924df0d1f81c1e3ff6":
        raise ValueError("unexpected V4 source manifest digest")
    if not PROFILE_RECEIPT.is_file():
        raise FileNotFoundError("successful actual-data profile terminal receipt is required")

    payload_root = PACKET_ROOT / "payload_root"
    payload_root.mkdir()
    code = copy_manifest_sources(payload_root, manifest)
    fold_assets = json.loads(ASSETS_MANIFEST_PATH.read_text())
    cache_manifest = stage_cache_asset(
        REPO_ROOT / "cache/bd26_h230_fold0_teacher_family_all_rows_v1",
        PACKET_ROOT / "cache_asset_upload",
        recipe["distillation_target_dependency"]["metadata_sha256"],
    )
    spec, external = build_spec(code, fold_assets, cache_manifest, recipe)
    spec_dir = PACKET_ROOT / "spec"
    spec_dir.mkdir()
    spec_path = spec_dir / "h230-seed9-parent-fit-v1.json"
    write_json(spec_path, spec)
    bundle_dir = PACKET_ROOT / "bundle"
    bundle_build_inputs = {
        "spec_sha256": sha256_file(spec_path),
        "code_members": len(code),
        "external_mount_count": len(external),
        "external_mounts": external,
    }
    build_bundle(
        payload_root,
        [spec_path],
        bundle_dir,
        BUNDLE_DATASET,
        external,
        [FOLD_ASSET_DATASET, CACHE_DATASET],
    )
    packet = {
        "schema": "h230-kaggle-parent-fit-transport-v2",
        "packet_id": "H230_STRONG_PARENT_SEED9_KAGGLE_FIT_V2_20261008",
        "created_utc": "2026-10-08",
        "status": "prepared_for_root_review_not_uploaded_not_launched",
        "fit_authorized": False,
        "attempt_id": FIT_ATTEMPT_ID,
        "run_name": FIT_RUN_NAME,
        "recipe_path": str(RECIPE_PATH.relative_to(REPO_ROOT)),
        "recipe_sha256": sha256_file(RECIPE_PATH),
        "source_manifest_path": str(SOURCE_MANIFEST.relative_to(REPO_ROOT)),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "actual_data_profile_receipt_path": str(PROFILE_RECEIPT.relative_to(REPO_ROOT)),
        "actual_data_profile_receipt_sha256": sha256_file(PROFILE_RECEIPT),
        "fit_scope": "Four zero-based epochs over all 10,606 train plus 1,873 supplied valid groups; this is all-data fitting, including exposed validation groups, with no held-out or public-score claim.",
        "reference_fidelity": "Independent V4 method-compatible seed-9 strong parent; clean-fold-trained own teachers, source-scale coefficient mass 0.8125, not exact reference teacher ancestry.",
        "transports": {
            "existing_raw_dataset": RAW_DATASET,
            "existing_fold_assets_dataset": FOLD_ASSET_DATASET,
            "new_cache_asset_dataset": CACHE_DATASET,
            "new_code_bundle_dataset": BUNDLE_DATASET,
            "new_kernel": f"{BUNDLE_DATASET}-runner",
            "upload_or_push_performed": False,
        },
        "fit_limits": spec["limits"],
        "kernel_process_wall_limit_seconds": 6_000,
        "kernel_runner_finalization_reserve_seconds": 300,
        "worker_finalization_reserve_seconds": 300,
        "fit_runtime_evidence": {
            "actual_data_profile_timed_batches": 1,
            "actual_data_profile_stage_sum_ms": 1279.67,
            "source_training_batches_per_epoch": 779,
            "recipe_epochs": 4,
            "arithmetic_stage_sum_projection_seconds": 1279.67 * 779 * 4 / 1000,
            "projection_is_eta": False,
            "limitation": "This arithmetic applies one measured RTX 3060 Laptop batch stage sum uniformly; it is not a wall-time ETA and does not measure Kaggle T4 throughput. The StageTimer interval estimate is NaN with one timed batch. Kaggle device and account quota must be freshly checked before any upload/push/fit.",
        },
        "input_stage_policy": "The bundle runner streams SHA-256 over every declared mounted input before launching training; this includes raw Parquet. The parent trainer then performs its unchanged mandatory all-data cache metadata, membership, array-hash and finiteness validation.",
        "cache_asset_policy": "The uploaded parent cache contains metadata plus main_source train/valid arrays only. It excludes normalized/homogeneous/family arrays; no family head is used in this parent fit. Later family-head students require the separately bound family arrays.",
        "builder_evidence": bundle_build_inputs,
        "future_student_note": "A student is a fresh independent AdamW/OneCycle model per ADR0023; its base output layout is 23 and the appended eight family outputs make 31 total. No parent checkpoint initialization is implied by these dimensions.",
        "non_authorizations": [
            "No Kaggle dataset upload or kernel push was performed.",
            "No Kaggle quota/runtime admission was inferred from list/status queries.",
            "No fit may launch until root reviews this exact bundle/input/timeout contract and writes a separate fit authorization.",
        ],
    }
    write_json(PACKET_ROOT / "H230_KAGGLE_PARENT_FIT_TRANSPORT_V2_20261008.json", packet)
    (PACKET_ROOT / "README.md").write_text(
        "# H230 seed-9 parent Kaggle transport (V2)\n\n"
        "Prepared offline only. The cache asset directory is hard-linked to the already validated local main-source cache; large arrays are not reread by this builder. The kernel runner verifies all declared file hashes after Kaggle mounts them, before training. The parent recipe uses no family head, so the family prediction arrays are intentionally absent.\n\n"
        "No dataset upload, kernel push, or fit has been performed. Root review and fresh Kaggle quota/device admission are still required.\n",
        encoding="utf-8",
    )
    print(json.dumps({"packet": str(PACKET_ROOT), "spec": str(spec_path), **bundle_build_inputs}, sort_keys=True))


if __name__ == "__main__":
    build()
