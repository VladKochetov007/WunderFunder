"""Pinned TL3 ZIP verification and create-only student-member extraction."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast
from zipfile import ZipFile, ZipInfo

SOURCE_ARCHIVE = Path(
    "/home/vlad/development/wunder/wnn_connectome_starterpack/"
    "submissions/team_tl3_original_20261018.zip"
)
SOURCE_ARCHIVE_SHA256 = "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569"
SOURCE_RECEIPT = Path(
    "/home/vlad/development/wunder/wnn_connectome_starterpack/"
    "research/campaign_2026-10-20/submission_delivery_r1/"
    "tl3_member_adaptation/SOURCE_ARCHIVE_CONTENTS_R1.json"
)
SOURCE_RECEIPT_SHA256 = "73cd201b165424f4b7ec72fe4bf4d9d573f56d93c0501519ed3bc6a3ea11a2bd"
SCRATCH_ROOT = Path("/home/vlad/development/wunder/research_tmp")
SOURCE_MEMBER = "weights_s5.npz"
EXTRACTED_MEMBER = "source_weights_s5.npz"


def file_sha256(path: Path) -> str:
    """Hash a file with bounded memory for source and output provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_archive(
    archive_path: Path,
) -> tuple[str, str, dict[str, dict[str, object]]]:
    """Verify the exact source ZIP and its root-committed entry-hash receipt.

    Args:
        archive_path: Candidate source ZIP path.

    Returns:
        Archive SHA-256, receipt SHA-256, and the pinned entry manifest.

    Raises:
        OSError: If the archive or receipt cannot be read.
        ValueError: If either pinned identity or the receipt content differs.
    """
    archive_path = archive_path.resolve(strict=True)
    if not archive_path.is_file() or file_sha256(archive_path) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source archive SHA-256 is not the pinned TL3 ZIP")
    receipt_bytes = SOURCE_RECEIPT.read_bytes()
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha256 != SOURCE_RECEIPT_SHA256:
        raise ValueError("pinned source archive receipt changed")
    receipt: object = json.loads(receipt_bytes)
    if not isinstance(receipt, dict) or receipt.get("archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source receipt does not bind the pinned archive")
    entries = receipt.get("entries")
    if not isinstance(entries, dict) or SOURCE_MEMBER not in entries:
        raise ValueError("source receipt has no pinned student member")
    return SOURCE_ARCHIVE_SHA256, receipt_sha256, cast(dict[str, dict[str, object]], entries)


def resolve_create_only_output_dir(requested: Path) -> Path:
    """Require a new absolute output directory below physical research scratch."""
    if not requested.is_absolute():
        raise ValueError("--output-dir must be an absolute path on physical scratch")
    scratch_root = SCRATCH_ROOT.resolve(strict=True)
    output_dir = requested.parent.resolve(strict=True) / requested.name
    if output_dir == scratch_root or not output_dir.is_relative_to(scratch_root):
        raise ValueError(f"--output-dir must be inside {scratch_root}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"create-only output directory already exists: {output_dir}")
    return output_dir


def _validate_archive_paths(infos: list[ZipInfo], entries: Mapping[str, dict[str, object]]) -> None:
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("source ZIP contains duplicate entry names")
    if set(names) != set(entries):
        raise ValueError("source ZIP has missing or unexpected entries")
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            path.is_absolute()
            or "\\" in info.filename
            or ".." in path.parts
            or info.is_dir()
            or stat.S_ISLNK(mode)
            or info.flag_bits & 1
        ):
            raise ValueError(f"unsafe or unsupported ZIP entry: {info.filename}")
        record = entries[info.filename]
        size, crc32 = record.get("bytes"), record.get("crc32")
        if not isinstance(size, int) or not isinstance(crc32, str):
            raise ValueError(f"invalid pinned metadata for {info.filename}")
        if info.file_size != size or f"{info.CRC:08x}" != crc32:
            raise ValueError(f"ZIP size/CRC differs from pinned receipt: {info.filename}")


def extract_verified_entries(
    archive_path: Path,
    output_dir: Path,
    entries: Mapping[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Hash every original ZIP entry while extracting only the student NPZ.

    Args:
        archive_path: Previously hash-verified pinned ZIP.
        output_dir: Newly created destination directory.
        entries: Entry manifest from the pinned source receipt.

    Returns:
        The verified byte count, SHA-256 and CRC32 for every original entry.

    Raises:
        OSError: If archive or extracted member bytes cannot be read or written.
        ValueError: If ZIP paths, entry hashes, sizes, or CRCs differ from the receipt.
    """
    records: dict[str, dict[str, object]] = {}
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        _validate_archive_paths(infos, entries)
        for info in infos:
            record = entries[info.filename]
            expected_size = cast(int, record["bytes"])
            expected_hash = cast(str, record["sha256"])
            expected_crc = cast(str, record["crc32"])
            output_member = (
                output_dir / EXTRACTED_MEMBER if info.filename == SOURCE_MEMBER else None
            )
            sink: BinaryIO | None = output_member.open("xb") if output_member is not None else None
            digest = hashlib.sha256()
            size = 0
            try:
                with archive.open(info) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        size += len(chunk)
                        digest.update(chunk)
                        if sink is not None:
                            sink.write(chunk)
            finally:
                if sink is not None:
                    sink.close()
            actual_hash = digest.hexdigest()
            if size != expected_size or actual_hash != expected_hash:
                raise ValueError(f"ZIP entry bytes differ from pinned receipt: {info.filename}")
            records[info.filename] = {"bytes": size, "sha256": actual_hash, "crc32": expected_crc}
    return records
