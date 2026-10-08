from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath

SOURCE_DATASET = "vladkochetov07/e529-gru512-768-muon-capacity-v3-source-20261008"
EXPECTED_ASSET_MANIFEST_SHA256 = "20d3c0178d3723a6884f905dfd7945cfe3b02ce549cad7ee1577c135b782b0f0"
PACKET_ID = "E529_H232_GRU512_768_MUON_SEED9_PROFILE_V3_20261008"
REQUIRED_PACKET_FILES = ('capacity_worker.py', 'kernel.py', 'train_entry.py', 'verified_digest_handoff.py', 'README.md', 'kernel-metadata.json', 'source/source_archive.tar', 'source/source_archive_manifest.json', 'fold0.json', 'portable_inputs.json', 'prior_input_identity_v2.json', 'GRU512_768_MUON_PROFILE_RECIPE_V3.json', 'ROOT_PROFILE_AUTHORIZATION.json', 'ROOT_PROFILE_SOURCE_REVIEW.json')


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
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in raw:
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
