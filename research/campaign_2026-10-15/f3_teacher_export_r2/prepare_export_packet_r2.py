"""Prepare an offline, content-addressed F3 teacher export packet."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PACKET = Path("research/campaign_2026-10-15/f3_teacher_export_r2")
R1 = Path("research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1")
R5 = Path("research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5")
RUN = "clean_t3_f3_muonall_h384_e3_s3_r5"
ASSET_DATASET = "vladkochetov07/wnn-f3-t3-cp2-assets-20261015-r1"
BUNDLE_DATASET = "vladkochetov07/wnn-f3-t3-cp2-exports-20261015-r2"
ASSET_RELATIVE = Path("artifacts/kaggle_assets") / ASSET_DATASET.split("/", 1)[1]
CHECKPOINT_RELATIVE = Path("artifacts/runs") / RUN / "model_ep2.pt"
PROVENANCE_RELATIVE = Path("artifacts/runs") / RUN / "provenance.json"
EXPECTED_CHECKPOINT_SHA256 = "d67bed97eaa16f73338ea275c316bf76c785b37f8451aa5cee8fe5daadf2f1b5"
EXPECTED_PROVENANCE_SHA256 = "c60719d1db7d44d17e04e0379f576616a43c6ad2564fd58e192ced5453bc8da3"
EXPECTED_SPLIT_SHA256 = "78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343"
JOB_NAME = "export_f3_teacher_cp2_triplet_20261015_r2"
KERNEL_PROCESS_WALL_SECONDS_MAX = 21_600
PREDICTION_OUTPUTS = [
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_train_train.npy",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_train_train.json",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_valid_valid.npy",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_fit_valid_valid.json",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_full_valid_valid.npy",
    "cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_full_valid_valid.json",
    "results/f3_teacher_export_pipeline_r2.json",
]
WORKER_SECONDS = 18_000
WORKER_RAM_BYTES = 8_000_000_000
WORKER_GPU_ALLOCATOR_BYTES = 12_884_901_888
WORKER_GPU_STOP_BYTES = 13_958_643_712


def sha256_file(path: Path) -> str:
    """Hash a file through fixed-size reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, max_bytes: int = 8_000_000) -> Any:
    """Read bounded packet metadata only after checking its length."""
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON metadata exceeds {max_bytes} bytes: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_entry(root: Path, relative: str) -> dict[str, object]:
    """Describe one local file by relative path, exact size, and SHA-256."""
    path = root / relative
    return {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}


def enforce_kernel_wall_cap(bundle: Path) -> None:
    """Set the generated outer kernel deadline to the R2 quota-safe cap."""
    path = bundle / "kernel/kernel.py"
    source = path.read_text(encoding="utf-8")
    old = "KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200"
    pinned = f"KERNEL_PROCESS_WALL_SECONDS_MAX = {KERNEL_PROCESS_WALL_SECONDS_MAX:_}"
    if pinned in source and old not in source:
        return
    if source.count(old) != 1:
        raise ValueError("generated Kaggle bootstrap has an unexpected outer deadline")
    path.write_text(source.replace(old, pinned, 1), encoding="utf-8")


def load_module(name: str, path: Path) -> ModuleType:
    """Load a repository helper from its explicit source path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_checkpoint_assets(source_root: Path) -> tuple[Path, dict[str, Any]]:
    """Hard-link the verified checkpoint and provenance into private staging."""
    checkpoint = source_root / CHECKPOINT_RELATIVE
    provenance = source_root / PROVENANCE_RELATIVE
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("CP2 differs from the root-verified teacher checkpoint")
    if sha256_file(provenance) != EXPECTED_PROVENANCE_SHA256:
        raise ValueError("provenance differs from the root-verified sidecar")

    destination = source_root / ASSET_RELATIVE
    if destination.exists():
        raise FileExistsError(f"refusing to replace staged teacher asset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(destination.parent)], text=True
    ).strip()
    if filesystem in {"tmpfs", "ramfs"} or not filesystem:
        raise ValueError("teacher asset staging requires a physical filesystem")

    helper_path = source_root / "scripts/kaggle_assets.py"
    sys.path.insert(0, str(source_root / "scripts"))
    helper = load_module("f3_export_kaggle_assets", helper_path)
    helper.prepare(
        source_root,
        [CHECKPOINT_RELATIVE.as_posix(), PROVENANCE_RELATIVE.as_posix()],
        destination,
        ASSET_DATASET,
    )
    manifest = read_json(destination / "assets-manifest.json", max_bytes=64_000)
    identities = {item["path"]: item for item in manifest["files"]}
    expected = {
        CHECKPOINT_RELATIVE.as_posix(): EXPECTED_CHECKPOINT_SHA256,
        PROVENANCE_RELATIVE.as_posix(): EXPECTED_PROVENANCE_SHA256,
    }
    if set(identities) != set(expected) or any(
        identities[name]["sha256"] != digest for name, digest in expected.items()
    ):
        raise ValueError("staged asset manifest differs from frozen teacher pins")
    return destination, manifest


def verify_staged_assets(source_root: Path) -> tuple[Path, dict[str, Any]]:
    """Verify an existing content-addressed stage without rewriting its bytes."""
    destination = source_root / ASSET_RELATIVE
    manifest = read_json(destination / "assets-manifest.json", max_bytes=64_000)
    if manifest.get("dataset") != ASSET_DATASET or manifest.get("private") is not True:
        raise ValueError("existing staged teacher assets have an unexpected dataset identity")
    expected = {
        CHECKPOINT_RELATIVE.as_posix(): EXPECTED_CHECKPOINT_SHA256,
        PROVENANCE_RELATIVE.as_posix(): EXPECTED_PROVENANCE_SHA256,
    }
    entries = {item["path"]: item for item in manifest.get("files", [])}
    if set(entries) != set(expected):
        raise ValueError("existing staged teacher asset manifest has unexpected files")
    for relative, digest in expected.items():
        source = source_root / relative
        staged = destination / entries[relative]["filename"]
        if (
            entries[relative].get("sha256") != digest
            or entries[relative].get("size") != source.stat().st_size
            or sha256_file(source) != digest
            or sha256_file(staged) != digest
        ):
            raise ValueError(
                f"existing staged teacher asset differs from its frozen pin: {relative}"
            )
    return destination, manifest


def make_export_job(
    root: Path, source_root: Path, asset_manifest: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Build one strict job whose worker sequentially exports all three scopes."""
    base_path = root / R1 / "jobs/export_clean_t3_f3_muonall_h384_e3_s3_r1_fit_train.json"
    spec = read_json(base_path)
    if spec["name"] != "export_clean_t3_f3_muonall_h384_e3_s3_r1_fit_train":
        raise ValueError("unexpected R1 base prediction recipe")

    driver_path = f"{PACKET.as_posix()}/kernel_support/f3_teacher_export_driver_r2.py"
    code = {item["path"]: item for item in spec["code"]}
    for runtime_path in (
        "scripts/kaggle_common.py",
        "scripts/kaggle_gpu_runner.py",
        "scripts/kaggle_kernel.py",
    ):
        code[runtime_path] = file_entry(root, runtime_path)
    code[driver_path] = file_entry(root, driver_path)
    spec["code"] = sorted(code.values(), key=lambda item: item["path"])

    inputs = {item["path"]: item for item in spec["inputs"]}
    for relative, digest in (
        (CHECKPOINT_RELATIVE.as_posix(), EXPECTED_CHECKPOINT_SHA256),
        (PROVENANCE_RELATIVE.as_posix(), EXPECTED_PROVENANCE_SHA256),
    ):
        observed = file_entry(source_root, relative)
        if observed["sha256"] != digest:
            raise ValueError(f"frozen asset changed while building job: {relative}")
        inputs[relative] = observed
    spec["inputs"] = sorted(inputs.values(), key=lambda item: item["path"])

    spec["name"] = JOB_NAME
    spec["argv"] = ["{python}", driver_path]
    spec["artifacts"] = PREDICTION_OUTPUTS
    spec["limits"] = {
        "cpus": 2,
        "gpu_memory_bytes": WORKER_GPU_ALLOCATOR_BYTES,
        "gpu_stop_bytes": WORKER_GPU_STOP_BYTES,
        "ram_bytes": WORKER_RAM_BYTES,
        "timeout_seconds": WORKER_SECONDS,
    }
    spec["kind"] = "gpu"
    spec["code_sha256"] = None

    path = root / PACKET / f"jobs/{JOB_NAME}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_bundle = read_json(root / R5 / "bundle_f3_teacher_seed3_r5/bundle-manifest.json")
    external = dict(source_bundle["external_mounts"])
    asset_entries = {item["path"]: item for item in asset_manifest["files"]}
    for relative in (CHECKPOINT_RELATIVE.as_posix(), PROVENANCE_RELATIVE.as_posix()):
        external[relative] = {
            "dataset": ASSET_DATASET,
            "filename": asset_entries[relative]["filename"],
        }
    declared_paths = {item["path"] for item in spec["inputs"] + spec["code"]}
    external = {path: mount for path, mount in external.items() if path in declared_paths}
    missing_external = {
        item["path"]
        for item in spec["inputs"]
        if item["path"].startswith(("datasets/", "cache/", "artifacts/runs/"))
        and not (root / item["path"]).is_file()
        and item["path"] not in external
    }
    if missing_external:
        raise ValueError(f"unmounted nonlocal inputs: {sorted(missing_external)}")

    sys.path.insert(0, str(root / "scripts"))
    from kaggle_common import canonical_sha256

    spec["code_sha256"] = canonical_sha256(spec["code"])
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, external


def write_source_binding(
    root: Path,
    source_root: Path,
    asset_destination: Path,
    asset_manifest: dict[str, Any],
    job_path: Path,
    external: dict[str, Any],
    bundle: Path,
) -> dict[str, Any]:
    """Bind producer, fit lineage, assets, mounted sources, and package hashes."""
    spec = read_json(job_path)
    verification_path = R5 / "ROOT_CHECKPOINT_VERIFICATION_R1.json"
    verification = read_json(root / verification_path)
    if (
        verification.get("checkpoint_epoch") != 2
        or verification.get("split_sha256") != EXPECTED_SPLIT_SHA256
        or verification.get("pending")
        != [
            "fit_train_predictions",
            "fit_valid_predictions",
            "full_valid_predictions",
            "teacher_grade",
        ]
    ):
        raise ValueError("root checkpoint verification no longer matches the export-only tranche")

    manifest_path = bundle / "bundle-manifest.json"
    bundle_manifest = read_json(manifest_path)
    cp2_asset = next(
        item for item in asset_manifest["files"] if item["path"] == CHECKPOINT_RELATIVE.as_posix()
    )
    provenance_asset = next(
        item for item in asset_manifest["files"] if item["path"] == PROVENANCE_RELATIVE.as_posix()
    )
    job_inputs = {item["path"]: item for item in spec["inputs"]}
    split_input = job_inputs.get("research/campaign_2026-09-29/artifacts/splits/fold3.json")
    if split_input is None or split_input["sha256"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("job does not bind the expected source-qualified fold3 split")
    provenance = read_json(source_root / PROVENANCE_RELATIVE, max_bytes=1_000_000)
    source_binding = {
        "schema": "wnn.f3-teacher-cp2-export-source-binding.r2",
        "provider_action_performed": False,
        "evaluation_scope": "causal prediction export only; no training, labels, grading, or score selection",
        "producer": {
            "r5_checkpoint_verification": file_entry(root, verification_path.as_posix()),
            "run": RUN,
            "seed": 3,
            "source_qualified_split_manifest": {
                "path": split_input["path"],
                "size": split_input["size"],
                "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            },
            "checkpoint": {
                "path": CHECKPOINT_RELATIVE.as_posix(),
                "size": cp2_asset["size"],
                "sha256": EXPECTED_CHECKPOINT_SHA256,
                "selected_epoch": 2,
            },
            "provenance": {
                "path": PROVENANCE_RELATIVE.as_posix(),
                "size": provenance_asset["size"],
                "sha256": EXPECTED_PROVENANCE_SHA256,
                "selected_epoch": 2,
                "training_completed_updates": 22310,
                "amp_skipped_updates": 10,
                "training_args": {
                    key: provenance["args"].get(key)
                    for key in (
                        "name",
                        "seed",
                        "arch",
                        "cell",
                        "hidden",
                        "layers",
                        "train_source",
                        "valid_fold",
                        "epochs",
                        "checkpoint_epoch",
                        "batch",
                        "chunk",
                        "loss",
                        "loss_norm",
                        "precision",
                        "propensity",
                        "prop_power",
                        "prop_eps",
                        "valid_true_mask",
                    )
                },
            },
            "split_membership": {
                "manifest_sha256": EXPECTED_SPLIT_SHA256,
                "fit_train_groups": 10539,
                "fit_valid_groups": 1368,
                "evaluation_groups": 505,
                "full_valid_groups": 1873,
            },
        },
        "asset_dataset": {
            "id": ASSET_DATASET,
            "staged_at": str(asset_destination),
            "stager": file_entry(source_root, "scripts/kaggle_assets.py"),
            "manifest_sha256": sha256_file(asset_destination / "assets-manifest.json"),
            "files": asset_manifest["files"],
        },
        "frozen_source": {
            "r1_source_binding": file_entry(root, f"{R1.as_posix()}/SOURCE_BINDING_R1.json"),
            "kernel_bootstrap_template": file_entry(root, "scripts/kaggle_kernel.py"),
            "kernel_bootstrap_override": {
                "template_wall_seconds": 43_200,
                "packet_wall_seconds": KERNEL_PROCESS_WALL_SECONDS_MAX,
                "reason": "R1 template exceeds the currently reported remaining GPU quota; R2 binds a six-hour outer deadline",
            },
            "packet_builder": file_entry(root, f"{PACKET.as_posix()}/prepare_export_packet_r2.py"),
            "predictor": file_entry(
                root,
                f"{R1.as_posix()}/source_snapshot/scripts/predict_run.py",
            ),
            "wrapper": file_entry(
                root,
                f"{R1.as_posix()}/source_snapshot/scripts/cleanfold_exec.py",
            ),
            "driver": file_entry(
                root,
                f"{PACKET.as_posix()}/kernel_support/f3_teacher_export_driver_r2.py",
            ),
            "collector": file_entry(root, f"{PACKET.as_posix()}/collect_terminal_r2.py"),
            "collector_helper": file_entry(
                root,
                "research/campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r4/collect_terminal_r4.py",
            ),
            "job_spec": file_entry(root, job_path.relative_to(root).as_posix()),
        },
        "external_mounts": external,
        "bundle": {
            "dataset": BUNDLE_DATASET,
            "kernel_process_wall_seconds_max": KERNEL_PROCESS_WALL_SECONDS_MAX,
            "archive_bytes": bundle_manifest["archive_bytes"],
            "archive_sha256": bundle_manifest["archive_sha256"],
            "expanded_bytes": bundle_manifest["expanded_bytes"],
            "kernel_id": f"vladkochetov07/{BUNDLE_DATASET.split('/', 1)[1]}-runner",
            "kernel_metadata_sha256": sha256_file(bundle / "kernel/kernel-metadata.json"),
            "bootstrap_sha256": sha256_file(bundle / "kernel/kernel.py"),
        },
        "resource_plan": {
            "kernel_process_wall_seconds_max": KERNEL_PROCESS_WALL_SECONDS_MAX,
            "one_serial_worker": True,
            "worker_timeout_seconds": WORKER_SECONDS,
            "worker_ram_reservation_bytes": WORKER_RAM_BYTES,
            "gpu_allocator_bytes": WORKER_GPU_ALLOCATOR_BYTES,
            "observed_gpu_stop_bytes": WORKER_GPU_STOP_BYTES,
            "triplet_raw_prediction_payload_bytes": 2204800000,
            "minimum_working_free_disk_bytes_before_export": 6000000000,
            "source_hash_scan_evidence_seconds": 1125.241,
            "prediction_durations": "not measured; timeout is the sum of three existing 6000-second per-export recipe caps",
            "quota_gate": "root must obtain a fresh typed quota snapshot with at least 21600 GPU seconds remaining for the 21600-second outer kernel cap before upload/create/push",
        },
    }
    binding_path = root / PACKET / "SOURCE_BINDING_R2.json"
    binding_path.write_text(
        json.dumps(source_binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return source_binding


def offline_preflight(root: Path, bundle: Path, spec_path: Path) -> dict[str, Any]:
    """Check exact offline slug, mounted inputs, bundle, and output declaration."""
    manifest = read_json(bundle / "bundle-manifest.json")
    metadata = read_json(bundle / "kernel/kernel-metadata.json")
    dataset_metadata = read_json(bundle / "dataset-metadata.json")
    bootstrap = (bundle / "kernel/kernel.py").read_text(encoding="utf-8")
    spec = read_json(spec_path)
    code_entries = {item["path"]: item for item in spec["code"]}
    kernel_template = file_entry(root, "scripts/kaggle_kernel.py")
    kernel_template_source = (root / "scripts/kaggle_kernel.py").read_text(encoding="utf-8")
    binding = read_json(root / PACKET / "SOURCE_BINDING_R2.json")
    bundle_binding = binding["bundle"]
    archive = bundle / "bundle.wnn"
    slug = BUNDLE_DATASET.split("/", 1)[1]
    expected_kernel = f"vladkochetov07/{slug}-runner"
    expected_sources = {BUNDLE_DATASET} | {
        value["dataset"] if isinstance(value, dict) else "vladkochetov07/wunder-funder"
        for value in manifest.get("external_mounts", {}).values()
    }
    expected_bootstrap_pins = (
        f"BUNDLE_SLUG = {slug!r}",
        f"EXPECTED_ARCHIVE_SHA256 = {manifest['archive_sha256']!r}",
        f"EXPECTED_EXTERNAL_MOUNTS = {manifest['external_mounts']!r}",
    )
    if (
        metadata.get("id") != expected_kernel
        or metadata.get("code_file") != "kernel.py"
        or metadata.get("is_private") is not True
        or metadata.get("enable_gpu") is not True
        or metadata.get("enable_internet") is not False
        or metadata.get("machine_shape") != "NvidiaTeslaT4"
        or set(metadata.get("dataset_sources", [])) != expected_sources
        or metadata.get("dataset_sources", []).count(BUNDLE_DATASET) != 1
        or dataset_metadata.get("id") != BUNDLE_DATASET
        or dataset_metadata.get("isPrivate") is not True
        or len(bootstrap.encode("utf-8")) >= 1_000_000
        or any(pin not in bootstrap for pin in expected_bootstrap_pins)
        or manifest.get("dataset") != BUNDLE_DATASET
        or manifest.get("archive_sha256") != sha256_file(archive)
        or f"KERNEL_PROCESS_WALL_SECONDS_MAX = {KERNEL_PROCESS_WALL_SECONDS_MAX:_}" not in bootstrap
        or kernel_template_source.count("KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200") != 1
        or code_entries.get("scripts/kaggle_kernel.py") != kernel_template
        or bundle_binding.get("archive_sha256") != manifest.get("archive_sha256")
        or bundle_binding.get("kernel_id") != metadata.get("id")
        or bundle_binding.get("kernel_process_wall_seconds_max") != KERNEL_PROCESS_WALL_SECONDS_MAX
        or bundle_binding.get("kernel_metadata_sha256")
        != sha256_file(bundle / "kernel/kernel-metadata.json")
        or bundle_binding.get("bootstrap_sha256") != sha256_file(bundle / "kernel/kernel.py")
        or manifest.get("external_mounts")
        != read_json(root / PACKET / "SOURCE_BINDING_R2.json")["external_mounts"]
        or spec.get("name") != JOB_NAME
        or spec.get("artifacts") != PREDICTION_OUTPUTS
    ):
        raise ValueError("offline generated-kernel identity/mount/output preflight failed")
    expected_kernel_slug = metadata["id"].split("/", 1)[1]
    expected_artifacts = set(spec["artifacts"])
    if not all(
        name.startswith("cache/pred_") or name == "results/f3_teacher_export_pipeline_r2.json"
        for name in expected_artifacts
    ):
        raise ValueError("job declares an output outside the exact prediction triplet")
    if expected_kernel_slug != f"{slug}-runner":
        raise ValueError("generated title-derived kernel slug differs from the packet")
    return {
        "schema": "wnn.f3-teacher-cp2-export-offline-preflight.r2",
        "status": "passed",
        "provider_api_called": False,
        "bundle_slug": slug,
        "kernel_id": metadata["id"],
        "kernel_process_wall_seconds_max": KERNEL_PROCESS_WALL_SECONDS_MAX,
        "kernel_bootstrap_template_sha256": kernel_template["sha256"],
        "bootstrap_bytes": len(bootstrap.encode("utf-8")),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": manifest["archive_sha256"],
        "kernel_metadata_sha256": bundle_binding["kernel_metadata_sha256"],
        "kernel_bootstrap_sha256": bundle_binding["bootstrap_sha256"],
        "source_binding_sha256": sha256_file(root / PACKET / "SOURCE_BINDING_R2.json"),
        "external_mount_count": len(manifest["external_mounts"]),
        "asset_dataset_mounted": ASSET_DATASET,
        "declared_outputs": sorted(expected_artifacts),
    }


def main() -> None:
    """Stage assets and construct a private Kaggle packet without provider calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="isolated packet worktree")
    parser.add_argument("--source-root", type=Path, required=True, help="read-only main checkout")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="verify existing local assets/bundle and refresh only local binding receipts",
    )
    parser.add_argument(
        "--rebuild-local",
        action="store_true",
        help="rebuild the local job/bundle from verified staged assets without provider calls",
    )
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    if args.refresh_existing and args.rebuild_local:
        parser.error("--refresh-existing and --rebuild-local are mutually exclusive")

    bundle = root / PACKET / f"bundle_{BUNDLE_DATASET.split('/', 1)[1]}"
    if args.refresh_existing:
        asset_destination, asset_manifest = verify_staged_assets(source_root)
        job_path = root / PACKET / f"jobs/{JOB_NAME}.json"
        external = read_json(bundle / "bundle-manifest.json")["external_mounts"]
        if not job_path.is_file() or not bundle.is_dir():
            raise FileNotFoundError(
                "cannot refresh receipts without the existing frozen job and bundle"
            )
    elif args.rebuild_local:
        asset_destination, asset_manifest = verify_staged_assets(source_root)
        job_path, external = make_export_job(root, source_root, asset_manifest)
        candidate = bundle.with_name(bundle.name + ".candidate")
        if candidate.exists():
            raise FileExistsError(f"refusing to overwrite local candidate bundle: {candidate}")
        sys.path.insert(0, str(root / "scripts"))
        from kaggle_bundle import build_bundle

        asset_datasets = sorted(
            {mount["dataset"] for mount in external.values() if isinstance(mount, dict)}
        )
        external_cli = [
            f"{logical}={mapping['dataset']}:{mapping['filename']}"
            if isinstance(mapping, dict)
            else f"{logical}={mapping}"
            for logical, mapping in sorted(external.items())
        ]
        build_bundle(root, [job_path], candidate, BUNDLE_DATASET, external_cli, asset_datasets)
        if bundle.exists():
            superseded = bundle.with_name(bundle.name + ".superseded_unlaunched_r2")
            if superseded.exists():
                raise FileExistsError(
                    f"prior unlaunched bundle archive already exists: {superseded}"
                )
            bundle.rename(superseded)
        candidate.rename(bundle)
    else:
        asset_destination, asset_manifest = stage_checkpoint_assets(source_root)
        job_path, external = make_export_job(root, source_root, asset_manifest)
        sys.path.insert(0, str(root / "scripts"))
        from kaggle_bundle import build_bundle

        asset_datasets = sorted(
            {mount["dataset"] for mount in external.values() if isinstance(mount, dict)}
        )
        external_cli = [
            f"{logical}={mapping['dataset']}:{mapping['filename']}"
            if isinstance(mapping, dict)
            else f"{logical}={mapping}"
            for logical, mapping in sorted(external.items())
        ]
        build_bundle(root, [job_path], bundle, BUNDLE_DATASET, external_cli, asset_datasets)

    enforce_kernel_wall_cap(bundle)
    binding = write_source_binding(
        root, source_root, asset_destination, asset_manifest, job_path, external, bundle
    )
    preflight = offline_preflight(root, bundle, job_path)
    preflight_path = root / PACKET / "OFFLINE_PREFLIGHT_R2.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": preflight["status"],
                "job_spec": str(job_path),
                "bundle": str(bundle),
                "source_binding_sha256": hashlib.sha256(
                    (root / PACKET / "SOURCE_BINDING_R2.json").read_bytes()
                ).hexdigest(),
                "offline_preflight_sha256": sha256_file(preflight_path),
                "asset_manifest_sha256": sha256_file(asset_destination / "assets-manifest.json"),
                "archive_sha256": preflight["archive_sha256"],
                "checkpoint_sha256": binding["producer"]["checkpoint"]["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
