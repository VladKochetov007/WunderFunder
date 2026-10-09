from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any

KERNEL_TITLE = "WNN Loss Pair GPU 20261009 V4"
KERNEL_ID = "vladkochetov07/wnn-loss-pair-gpu-20261009-v4"
SOURCE_DATASET_TITLE = "WNN Loss Pair Code 20261009 R4"
SOURCE_DATASET_ID = "vladkochetov07/wnn-loss-pair-code-20261009-r4"
FOLD0_ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
FOLD0_ASSET_MANIFEST_SHA256 = (
    "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
)
CPU_SOURCE_KERNEL = "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1"
CPU_SOURCE_COMMIT = "d5daaa0f90b0362877522703c7e17aa78c19cac9"
KERNEL_FILENAME = "wnn-loss-pair-gpu-20261009-v4.py"
ARCHIVE_PATH = "wnn_loss_pair_src.tar.gz"
CODE_MANIFEST_PATH = "source-code-manifest.json"
SPLIT_PATH = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
REGISTRY_PATH = "research/campaign_2026-10-04/portable_inputs.json"
MEMBERSHIP_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "loss_pair_r1_membership_fold0_f03_s20261009.json"
)
BOOTSTRAP_PATH = Path("kernel") / KERNEL_FILENAME
EXPECTED_SOURCE_PINS = {
    SPLIT_PATH: "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    REGISTRY_PATH: "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c",
    MEMBERSHIP_PATH: "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd",
}


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def title_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def source_files(code_root: Path) -> list[Path]:
    paths = []
    for path in sorted(code_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(code_root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        paths.append(path)
    return paths


def deterministic_archive(code_root: Path, paths: list[Path]) -> bytes:
    output = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive,
    ):
        for path in paths:
            name = path.relative_to(code_root).as_posix()
            payload = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def write_json(path: Path, value: Any) -> bytes:
    raw = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def seal_candidate(candidate: Path, source_commit: str) -> dict[str, Any]:
    code_root = candidate / "code"
    kernel_root = candidate / "kernel"
    source_asset_root = candidate / "source_asset"
    if not code_root.is_dir() or not kernel_root.is_dir():
        raise FileNotFoundError("candidate must contain code/ and kernel/ directories")
    if source_asset_root.exists():
        raise FileExistsError("source_asset/ already exists; preserve the sealed candidate")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    for relative, expected in EXPECTED_SOURCE_PINS.items():
        path = code_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source input pin differs: {relative}")
    paths = source_files(code_root)
    rows = [
        {
            "path": path.relative_to(code_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    if not any(row["path"] == "scripts/loss_pair_gpu_worker.py" for row in rows):
        raise ValueError("sealed source lacks its deterministic GPU worker")
    code_manifest = {
        "schema": "wnn.kaggle-loss-pair-code-manifest",
        "format_version": 1,
        "source_commit": source_commit,
        "trainer_base_commit": CPU_SOURCE_COMMIT,
        "files": rows,
    }
    code_manifest_bytes = canonical_json(code_manifest)
    archive_bytes = deterministic_archive(code_root, paths)
    asset_payloads = {
        ARCHIVE_PATH: archive_bytes,
        CODE_MANIFEST_PATH: code_manifest_bytes,
        **{relative: (code_root / relative).read_bytes() for relative in EXPECTED_SOURCE_PINS},
    }
    asset_rows = [
        {
            "filename": f"{sha256_bytes(payload)}.wnn",
            "path": relative,
            "sha256": sha256_bytes(payload),
            "size": len(payload),
        }
        for relative, payload in sorted(asset_payloads.items())
    ]
    asset_manifest = {
        "dataset": SOURCE_DATASET_ID,
        "private": True,
        "version": 1,
        "files": asset_rows,
    }
    asset_manifest_bytes = canonical_json(asset_manifest)
    asset_manifest_sha = sha256_bytes(asset_manifest_bytes)
    source_asset_root.mkdir(parents=True)
    for relative, payload in asset_payloads.items():
        filename = next(row["filename"] for row in asset_rows if row["path"] == relative)
        (source_asset_root / filename).write_bytes(payload)
    (source_asset_root / "assets-manifest.json").write_bytes(asset_manifest_bytes)
    write_json(
        source_asset_root / "dataset-metadata.json",
        {
            "id": SOURCE_DATASET_ID,
            "title": SOURCE_DATASET_TITLE,
            "isPrivate": True,
            "licenses": [{"name": "other"}],
        },
    )

    bootstrap_path = kernel_root / KERNEL_FILENAME
    bootstrap = bootstrap_path.read_text(encoding="utf-8")
    replacements = {
        "__ASSET_MANIFEST_SHA256__": asset_manifest_sha,
        "__ARCHIVE_SHA256__": sha256_bytes(archive_bytes),
        "__CODE_MANIFEST_SHA256__": sha256_bytes(code_manifest_bytes),
        "__SOURCE_COMMIT__": source_commit,
    }
    for placeholder, value in replacements.items():
        bootstrap = bootstrap.replace(placeholder, value)
    if any(placeholder in bootstrap for placeholder in replacements):
        raise ValueError("bootstrap contains an unresolved content pin")
    bootstrap_path.write_text(bootstrap, encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": KERNEL_FILENAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [SOURCE_DATASET_ID, FOLD0_ASSET_DATASET],
        "kernel_sources": [CPU_SOURCE_KERNEL],
        "competition_sources": [],
        "model_sources": [],
    }
    metadata_bytes = write_json(kernel_root / "kernel-metadata.json", metadata)
    if title_slug(KERNEL_TITLE) != KERNEL_ID.rsplit("/", 1)[1]:
        raise ValueError("GPU kernel title-derived slug differs from the metadata ID")
    if title_slug(SOURCE_DATASET_TITLE) != SOURCE_DATASET_ID.rsplit("/", 1)[1]:
        raise ValueError("source dataset title-derived slug differs from its metadata ID")
    script_bytes = bootstrap_path.stat().st_size
    if script_bytes >= 1_000_000:
        raise ValueError("Kaggle bootstrap exceeds the conservative 1 MB source envelope")
    packet = {
        "schema": "wnn.kaggle-loss-pair-gpu-packet.v4",
        "packet_id": KERNEL_ID,
        "supersedes": "vladkochetov07/wnn-loss-pair-gpu-20261009-v3",
        "status": "sealed_candidate_waiting_root_review_and_admission",
        "authorized_for_execution": False,
        "root_admission_required_before_dataset_creation_or_kernel_push": True,
        "repair": {
            "predecessor_failure_report": "gpu_loss_pair_r3/GPU_R3_WORKER_TERMINAL_DIAGNOSTIC_R1.json",
            "predecessor_failure_commit": "cbf77aa485f072cf92bc123b240ce16c78a5eb30",
            "cause": "ParsedSequenceCache.open passed the raw validation-receipt JSON bytes as validation_receipt_sha256. The split loader correctly rejected the non-64-character value before fitting.",
            "change": "Hash the already-verified raw receipt bytes before constructing VerifiedParsedCacheSources; retain the exact split-identity check and exercise the actual open-to-split-loader bridge in a regression test.",
            "scientific_recipe_changed": False,
        },
        "source": {
            "source_commit": source_commit,
            "trainer_base_commit": CPU_SOURCE_COMMIT,
            "source_dataset": SOURCE_DATASET_ID,
            "source_dataset_title": SOURCE_DATASET_TITLE,
            "source_dataset_title_slug_matches_id": True,
            "fold0_asset_dataset": FOLD0_ASSET_DATASET,
            "fold0_asset_manifest_sha256": FOLD0_ASSET_MANIFEST_SHA256,
            "asset_manifest_sha256": asset_manifest_sha,
            "archive_sha256": sha256_bytes(archive_bytes),
            "archive_size_bytes": len(archive_bytes),
            "code_manifest_sha256": sha256_bytes(code_manifest_bytes),
            "code_files": rows,
            "upload_inventory": asset_rows,
            "private_upload_files": [
                {
                    "path": "assets-manifest.json",
                    "size_bytes": len(asset_manifest_bytes),
                    "sha256": asset_manifest_sha,
                },
                *[
                    {"path": row["filename"], "size_bytes": row["size"], "sha256": row["sha256"]}
                    for row in asset_rows
                ],
                {
                    "path": "dataset-metadata.json",
                    "size_bytes": (source_asset_root / "dataset-metadata.json").stat().st_size,
                    "sha256": sha256_file(source_asset_root / "dataset-metadata.json"),
                },
            ],
        },
        "kernel": {
            "title": KERNEL_TITLE,
            "title_slug_matches_id": True,
            "code_file": KERNEL_FILENAME,
            "code_file_sha256": sha256_file(bootstrap_path),
            "code_file_size_bytes": script_bytes,
            "metadata_sha256": sha256_bytes(metadata_bytes),
            "metadata": metadata,
            "source_kernel": CPU_SOURCE_KERNEL,
            "provider_timeout_seconds": 3600,
            "worker_wall_seconds": 3300,
            "accelerator": "NvidiaTeslaT4",
            "actual_runtime_gate": "exactly two T4 devices; FP32 with CUDA TF32 disabled",
        },
        "producer": {
            "source_stage_receipt_sha256": "557cd0a61f1129f811dc104398c3c9c7295c3343a1c9a302018fcb588c0c1de7",
            "cache_manifest_sha256": "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b",
            "cache_key": "949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6",
            "cpu_validation_sha256": "84686dff926036b37362a3ea28edcf2166d0bb44bee11b2fcbd64082e4b5cf35",
            "cpu_profile_sha256": "c371f3c70e79e69de8b382b442869d2e06e72ea4890a16d19b22923d7c82825e",
            "source_verification_sha256": "1ccd91ccec85bd157cbd0244ae9032316b88d781a0caf19daf7a027f3a908567",
            "phase_log_sha256": "cee62c0cdfa0a8b13998690fa49a7b111b56798a8ff6baf09e0a06b315c008e5",
            "membership_sha256": EXPECTED_SOURCE_PINS[MEMBERSHIP_PATH],
            "split_sha256": EXPECTED_SOURCE_PINS[SPLIT_PATH],
            "portable_registry_sha256": EXPECTED_SOURCE_PINS[REGISTRY_PATH],
            "mount_validation": "runtime one-pass full cache hashing with DONTNEED, then read-only mmap; no Parquet and no cache copy",
            "fold0_auxiliary_mount_validation": {
                "dataset": FOLD0_ASSET_DATASET,
                "asset_manifest_sha256": FOLD0_ASSET_MANIFEST_SHA256,
                "files": [
                    {
                        "path": "cache/chain_folds.npz",
                        "sha256": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
                        "size_bytes": 550072,
                    },
                    {
                        "path": "cache/chains.npz",
                        "sha256": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
                        "size_bytes": 650142,
                    },
                    {
                        "path": "cache/train_targets.npz",
                        "sha256": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
                        "size_bytes": 1909335790,
                    },
                    {
                        "path": "cache/valid_targets.npz",
                        "sha256": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
                        "size_bytes": 337186198,
                    },
                ],
                "method": "one bounded SHA-256 pass with DONTNEED advice and exact split/registry binding; symlink into cache; each fit process checks the sealed receipt plus path/stat identity without rehashing the large NPZs",
            },
        },
        "experiment": {
            "hypothesis": "On an exposed fold-0 3% whole-component screen, weighted Huber delta 1 may change predictive residuals relative to weighted MSE when architecture, seed, members, AdamW and OneCycle are held fixed.",
            "contrast": "only --loss wmse versus --loss huber",
            "fold": 0,
            "fit_train_count": 350,
            "fit_valid_count_retained_not_used": 28,
            "evaluation_count": 468,
            "updates_per_arm": 220,
            "output_cap_bytes": 12_000_000_000,
            "minimum_working_disk_free_bytes": 4_000_000_000,
            "maximum_cgroup_current_bytes": 24_000_000_000,
            "minimum_mem_available_bytes": 2_000_000_000,
            "minimum_cgroup_headroom_bytes": 2_000_000_000,
            "provisional_gpu_seconds": 7200,
            "predictions_and_checkpoints_retained": True,
            "per_update_diagnostics": [
                "preclip gradient norms",
                "clip and skip flags",
                "LR",
                "beta1",
                "update timing",
            ],
        },
    }
    write_json(candidate / "gpu_loss_pair.packet.json", packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    packet = seal_candidate(args.candidate, args.source_commit)
    print(
        json.dumps(
            {
                "packet_id": packet["packet_id"],
                "asset_manifest_sha256": packet["source"]["asset_manifest_sha256"],
                "code_file_sha256": packet["kernel"]["code_file_sha256"],
                "packet_path": str(args.candidate / "gpu_loss_pair.packet.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
