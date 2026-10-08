"""Reuse the exact sealed V5R2 runner archive for the V5R3 validator successor."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

PACKET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    "/gru512_768_muon_fit_v5r3_skip_accounting_20261008/packet"
)
V5R2_PACKET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    "/gru512_768_muon_fit_v5r2_memory_safe_20261008/packet"
)
V5R2_ARCHIVE_SHA256 = "338443c53d63be1b7a1e9c06ba6f8545524c25626ac3aefeb881c130afa3a618"
V5R2_MANIFEST_SHA256 = "0d4a828d940111758aba8108d8245f25aa20cbd21567ccd98e47bf67f2fe7b6f"


def sha256_file(path: Path) -> str:
    """Hash one frozen source artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _archive_source(packet_root: Path, repo_root: Path) -> tuple[Path, Path]:
    """Copy V5R2's exact runner/trainer source archive without mutation."""
    parent = repo_root / V5R2_PACKET_RELATIVE
    parent_archive = parent / "source/source_archive.tar"
    parent_manifest = parent / "source/source_archive_manifest.json"
    if (
        not parent_archive.is_file()
        or not parent_manifest.is_file()
        or sha256_file(parent_archive) != V5R2_ARCHIVE_SHA256
        or sha256_file(parent_manifest) != V5R2_MANIFEST_SHA256
    ):
        raise ValueError("pinned V5R2 source archive or manifest is absent or changed")

    source_root = packet_root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    archive_path = source_root / "source_archive.tar"
    manifest_path = source_root / "source_archive_manifest.json"
    shutil.copyfile(parent_archive, archive_path)
    shutil.copyfile(parent_manifest, manifest_path)

    evidence_root = packet_root / "producer_evidence/v5r2"
    evidence_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parent_archive, evidence_root / "source_archive.tar")
    shutil.copyfile(parent_manifest, evidence_root / "source_archive_manifest.json")
    probe = parent / "producer_evidence/v5r2_page_cache_probe_20261008_r1.json"
    if not probe.is_file():
        raise FileNotFoundError("V5R2 bounded page-cache probe receipt is absent")
    shutil.copyfile(probe, evidence_root / "page_cache_probe_20261008_r1.json")

    if sha256_file(archive_path) != V5R2_ARCHIVE_SHA256:
        raise ValueError("V5R3 copied source archive differs from frozen V5R2 bytes")
    if sha256_file(manifest_path) != V5R2_MANIFEST_SHA256:
        raise ValueError("V5R3 copied source manifest differs from frozen V5R2 bytes")
    return archive_path, manifest_path
