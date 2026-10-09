"""Seal a per-fold parsed-cache CPU stage packet (multifold successor of the fold0 sealer)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "d5daaa0f90b0362877522703c7e17aa78c19cac9"
RAW_DATASET = "vladkochetov07/wunder-funder"
CODE_FILES = (
    "src/wnn/__init__.py",
    "src/wnn/data.py",
    "src/wnn/splits.py",
    "src/wnn/parsed_sequence_cache.py",
    "src/wnn/parsed_sequence_cache_reader.py",
    "scripts/parsed_sequence_cache.py",
    "scripts/profile_parsed_sequence_cache.py",
)
SOURCE_DATASET = SOURCE_DATASET_TITLE = KERNEL_TITLE = KERNEL_ID = KERNEL_FILENAME = ""
MEMBERSHIP_PATH = EXPECTED_MEMBERSHIP_SHA256 = ""
MULTIFOLD_DIR = "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1"


def configure_fold(fold: int, repo: Path, seed: int = 20261009) -> None:
    """Bind per-fold dataset, kernel and membership identities."""
    global SOURCE_DATASET, SOURCE_DATASET_TITLE, KERNEL_TITLE, KERNEL_ID, KERNEL_FILENAME
    global MEMBERSHIP_PATH, EXPECTED_MEMBERSHIP_SHA256
    tag = f"F{fold}" if seed == 20261009 else f"F{fold} S{seed}"
    SOURCE_DATASET_TITLE = f"WNN Parsed Cache Source 20261009 {tag}"
    SOURCE_DATASET = "vladkochetov07/" + _title_slug(SOURCE_DATASET_TITLE)
    KERNEL_TITLE = f"WNN Parsed Cache Stage 20261009 {tag}"
    KERNEL_ID = "vladkochetov07/" + _title_slug(KERNEL_TITLE)
    KERNEL_FILENAME = _title_slug(KERNEL_TITLE) + ".py"
    MEMBERSHIP_PATH = f"{MULTIFOLD_DIR}/membership_fold{fold}_f03_s{seed}.json"
    EXPECTED_MEMBERSHIP_SHA256 = _sha256((repo / MEMBERSHIP_PATH).read_bytes())


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_bytes(repo: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{relative}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _title_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _write_json(path: Path, value: Any) -> bytes:
    raw = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def seal_candidate(repo: Path, output: Path, *, commit: str = SOURCE_COMMIT) -> dict[str, Any]:
    """Build an opaque, hash-bound CPU kernel and source dataset candidate."""
    if commit != SOURCE_COMMIT:
        raise ValueError("source closure must come from the reviewed parsed-cache commit")
    if output.exists():
        raise FileExistsError(f"candidate destination already exists: {output}")

    source_payloads = {name: _git_bytes(repo, commit, name) for name in CODE_FILES}
    membership = (repo / MEMBERSHIP_PATH).read_bytes()
    membership_sha256 = _sha256(membership)
    if membership_sha256 != EXPECTED_MEMBERSHIP_SHA256:
        raise ValueError("actor membership bytes differ from their frozen SHA-256")
    code_manifest = {
        "schema": "wnn.kaggle-source-code-manifest",
        "format_version": 1,
        "commit": commit,
        "files": [
            {
                "path": name,
                "size": len(source_payloads[name]),
                "sha256": _sha256(source_payloads[name]),
            }
            for name in CODE_FILES
        ],
    }
    code_manifest_bytes = _canonical_json(code_manifest)
    archive = _archive(source_payloads)
    asset_payloads = {
        "wnn_src.tar.gz": archive,
        "source-code-manifest.json": code_manifest_bytes,
        MEMBERSHIP_PATH: membership,
    }
    assets_manifest = {
        "dataset": SOURCE_DATASET,
        "private": True,
        "version": 1,
        "files": [
            {
                "filename": f"{_sha256(payload)}.wnn",
                "path": name,
                "sha256": _sha256(payload),
                "size": len(payload),
            }
            for name, payload in sorted(asset_payloads.items())
        ],
    }
    assets_manifest_bytes = _canonical_json(assets_manifest)
    assets_manifest_sha256 = _sha256(assets_manifest_bytes)

    candidate = output
    source_dir = candidate / "source_asset"
    kernel_dir = candidate / "kernel"
    source_dir.mkdir(parents=True)
    kernel_dir.mkdir()
    for name, payload in asset_payloads.items():
        path = next(row["filename"] for row in assets_manifest["files"] if row["path"] == name)
        (source_dir / path).write_bytes(payload)
    (source_dir / "assets-manifest.json").write_bytes(assets_manifest_bytes)
    _write_json(
        source_dir / "dataset-metadata.json",
        {
            "id": SOURCE_DATASET,
            "title": SOURCE_DATASET_TITLE,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        },
    )

    template_root = Path(__file__).parent
    kernel_template = (template_root / "kernel.py").read_text(encoding="utf-8")
    runtime_source = (template_root / "cpu_stage_runtime.py").read_text(encoding="utf-8")
    if not kernel_template.endswith("# CPU_STAGE_RUNTIME_INSERTION_POINT\n"):
        raise ValueError("kernel template insertion point is missing or changed")
    runtime_lines = runtime_source.splitlines()
    if runtime_lines[0] != "from __future__ import annotations":
        raise ValueError("runtime module must begin with future annotations")
    runtime = "\n".join(runtime_lines[2:])
    replacements = {
        "__ASSET_MANIFEST_SHA256__": assets_manifest_sha256,
        "__ARCHIVE_SHA256__": _sha256(archive),
        "__CODE_MANIFEST_SHA256__": _sha256(code_manifest_bytes),
        "__SOURCE_DATASET__": SOURCE_DATASET,
        "__MEMBERSHIP_SHA256__": EXPECTED_MEMBERSHIP_SHA256,
        "__MEMBERSHIP_PATH__": MEMBERSHIP_PATH,
    }
    for placeholder, value in replacements.items():
        runtime = runtime.replace(placeholder, value)
    entrypoint = (
        kernel_template.removesuffix("# CPU_STAGE_RUNTIME_INSERTION_POINT\n")
        + runtime
        + "\nraise SystemExit(run())\n"
    )
    if (
        "__ASSET_MANIFEST_SHA256__" in entrypoint
        or "__MEMBERSHIP_PATH__" in entrypoint
        or "CPU_STAGE_RUNTIME_INSERTION_POINT" in entrypoint
    ):
        raise ValueError("sealed kernel still contains unresolved placeholders")
    kernel_path = kernel_dir / KERNEL_FILENAME
    kernel_path.write_text(entrypoint, encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": KERNEL_FILENAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [SOURCE_DATASET, RAW_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata_bytes = _write_json(metadata_path, metadata)
    if _title_slug(KERNEL_TITLE) != KERNEL_ID.rsplit("/", 1)[1]:
        raise ValueError("kernel title-derived slug does not equal the metadata id")
    if _title_slug(SOURCE_DATASET_TITLE) != SOURCE_DATASET.rsplit("/", 1)[1]:
        raise ValueError("source dataset title-derived slug does not equal the metadata id")

    raw_source_manifest = json.loads(membership)
    packet = {
        "schema": "wnn.kaggle-parsed-cache-source-stage-packet.v1",
        "packet_id": KERNEL_ID,
        "status": "sealed_candidate_waiting_root_review_and_admission",
        "root_admission_required_before_any_upload_or_push": True,
        "kernel": {
            "title": KERNEL_TITLE,
            "title_slug_matches_metadata_id": True,
            "script": {"path": f"kernel/{KERNEL_FILENAME}", "sha256": _sha256(entrypoint.encode())},
            "metadata": {"path": "kernel/kernel-metadata.json", "sha256": _sha256(metadata_bytes)},
            "gpu": False,
            "tpu": False,
            "provider_timeout_seconds": 3600,
            "worker_wall_seconds": 3300,
        },
        "source_asset": {
            "id": SOURCE_DATASET,
            "title": SOURCE_DATASET_TITLE,
            "title_slug_matches_metadata_id": True,
            "version": 1,
            "assets_manifest_sha256": assets_manifest_sha256,
            "code_manifest_sha256": _sha256(code_manifest_bytes),
            "archive_sha256": _sha256(archive),
            "archive_bytes": len(archive),
            "archive_entry_sha256s": code_manifest["files"],
            "upload_file_inventory": [
                {
                    "path": name,
                    "filename": row["filename"],
                    "size": row["size"],
                    "sha256": row["sha256"],
                }
                for name in sorted(("wnn_src.tar.gz", "source-code-manifest.json", MEMBERSHIP_PATH))
                for row in assets_manifest["files"]
                if row["path"] == name
            ],
            "membership_path": MEMBERSHIP_PATH,
            "membership_sha256": membership_sha256,
            "raw_sources": raw_source_manifest["sources"],
        },
        "cache_contract": {
            "fold": json.loads(membership)["fold"],
            "fit_train_sequences": 350,
            "fit_valid_sequences_retained_but_not_used_for_fit": 28,
            "evaluation_sequences": 468,
            "train_source_bytes": 29_240_219_011,
            "valid_source_bytes": 5_615_509_128,
            "cache_array_payload_bytes": 7_732_440_000,
            "output_cap_bytes": 12_000_000_000,
            "disk_reserves_bytes": {
                "output": 2_000_000_000,
                "checkpoint": 0,
                "free": 4_000_000_000,
            },
            "cache_output_path": "/kaggle/working/out/parsed_cache",
            "records_actual_parse_count_and_os_io_separately": True,
        },
        "scope": {
            "cpu_source_profile_only": True,
            "no_gpu_or_tpu_fit": True,
            "no_provider_action_authorized": True,
            "no_local_parquet_or_cache_build": True,
        },
        "proposed_commands": {
            "dataset_create": "kaggle datasets create -p <candidate>/source_asset",
            "kernel_push": "kaggle kernels push -p <candidate>/kernel -t 3600",
        },
    }
    _write_json(candidate / "cpu_source_stage.packet.json", packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    configure_fold(args.fold, args.repo, args.seed)
    packet = seal_candidate(args.repo, args.output)
    packet_sha = _sha256((args.output / "cpu_source_stage.packet.json").read_bytes())
    print(
        json.dumps({"packet_id": packet["packet_id"], "packet_sha256": packet_sha}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
