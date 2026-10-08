"""Build V5R2 from the exact V5R1 source archive with inventory-only telemetry."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

PACKET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    "/gru512_768_muon_fit_v5r2_memory_safe_20261008/packet"
)
V5R1_PACKET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    "/gru512_768_muon_fit_v5r1_20261008/packet"
)
V5R1_ARCHIVE_SHA256 = "f175c2d364d982a883e0c7be9a6341dbdeb8e60a652aa1296eb65fa9fe80ad9e"


def sha256_file(path: Path) -> str:
    """Hash one file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _patch_runner(content: str) -> str:
    """Add measured cgroup/system fields while preserving the existing RAM gate."""
    old = """    capacity = memory.total
    available = memory.available
    path = Path("/sys/fs/cgroup/memory.max")
    if path.exists() and path.read_text().strip() != "max":
        capacity = min(capacity, int(path.read_text().strip()))
        current = Path("/sys/fs/cgroup/memory.current")
        if current.exists():
            available = min(available, capacity - int(current.read_text().strip()))
"""
    new = """    capacity = memory.total
    system_available = memory.available
    available = system_available
    cgroup_current_bytes = None
    cgroup_max_bytes = None
    cgroup_available_bytes = None
    maximum_path = Path("/sys/fs/cgroup/memory.max")
    current_path = Path("/sys/fs/cgroup/memory.current")
    if current_path.exists():
        cgroup_current_bytes = int(current_path.read_text().strip())
    if maximum_path.exists() and maximum_path.read_text().strip() != "max":
        cgroup_max_bytes = int(maximum_path.read_text().strip())
        capacity = min(capacity, cgroup_max_bytes)
        if cgroup_current_bytes is not None:
            cgroup_available_bytes = cgroup_max_bytes - cgroup_current_bytes
            available = min(system_available, capacity - cgroup_current_bytes)
"""
    if content.count(old) != 1:
        raise ValueError("V5R1 host inventory source anchor is absent or ambiguous")
    content = content.replace(old, new, 1)
    old_return = """        "capacity_bytes": capacity,
        "available_bytes": min(available, capacity),
        "affinity": sorted(os.sched_getaffinity(0)),
"""
    new_return = """        "capacity_bytes": capacity,
        "available_bytes": min(available, capacity),
        "system_available_bytes": system_available,
        "cgroup_current_bytes": cgroup_current_bytes,
        "cgroup_max_bytes": cgroup_max_bytes,
        "cgroup_available_bytes": cgroup_available_bytes,
        "affinity": sorted(os.sched_getaffinity(0)),
"""
    if content.count(old_return) != 1:
        raise ValueError("V5R1 host inventory return anchor is absent or ambiguous")
    return content.replace(old_return, new_return, 1)


def _archive_source(packet_root: Path, repo_root: Path) -> tuple[Path, Path]:
    """Copy the reviewed V5R1 tree and change only the runner inventory fields."""
    v5r1_packet = repo_root / V5R1_PACKET_RELATIVE
    base_archive = v5r1_packet / "source/source_archive.tar"
    base_manifest = v5r1_packet / "source/source_archive_manifest.json"
    if not base_archive.is_file() or sha256_file(base_archive) != V5R1_ARCHIVE_SHA256:
        raise ValueError("pinned V5R1 source archive is absent or changed")
    packet_base_archive = packet_root / "producer_evidence/v5r1/source_archive.tar"
    packet_base_manifest = packet_root / "producer_evidence/v5r1/source_archive_manifest.json"
    packet_base_archive.parent.mkdir(parents=True, exist_ok=True)
    if not packet_base_archive.exists():
        shutil.copyfile(base_archive, packet_base_archive)
    if not packet_base_manifest.exists():
        shutil.copyfile(base_manifest, packet_base_manifest)
    if sha256_file(packet_base_archive) != V5R1_ARCHIVE_SHA256:
        raise ValueError("preserved V5R1 source archive differs from its pin")

    source_root = packet_root / "source"
    archive_path = source_root / "source_archive.tar"
    manifest_path = source_root / "source_archive_manifest.json"
    base_members: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="gru-capacity-v5r2-archive-") as temporary:
        extracted = Path(temporary)
        with tarfile.open(base_archive, "r") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    raise ValueError("V5R1 source archive contains a nonregular member")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("V5R1 source member cannot be read")
                target = extracted / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())
                base_members[member.name] = target

        runner_path = extracted / "scripts/kaggle_gpu_runner.py"
        if not runner_path.is_file():
            raise ValueError("V5R1 archive lacks its frozen Kaggle runner")
        original_runner_sha256 = sha256_file(runner_path)
        runner_path.write_text(
            _patch_runner(runner_path.read_text(encoding="utf-8")), encoding="utf-8"
        )
        members = [
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in sorted(base_members.items())
        ]
        manifest = {
            "schema": "wnn-gru-capacity-source-archive-v1",
            "packet_id": "E529_H232_GRU512_768_MUON_SEED9_FIT_V5R2_20261008",
            "parent_packet": "E529_H232_GRU512_768_MUON_SEED9_FIT_V5R1_20261008",
            "parent_source_archive_sha256": V5R1_ARCHIVE_SHA256,
            "parent_source_manifest_sha256": sha256_file(base_manifest),
            "instrumentation_only_changes": [
                "add measured cgroup current/maximum/available and system available memory fields to the pre-admission inventory; preserve capacity and available_bytes gates",
            ],
            "runner_member_parent_sha256": original_runner_sha256,
            "member_count": len(members),
            "members": members,
        }
        source_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as archive:
            for name, path in sorted(base_members.items()):
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return archive_path, manifest_path


def main() -> None:
    """Rebuild V5R2 from pinned V5R1 source without reading training arrays."""
    script_path = Path(__file__).resolve()
    packet_root = script_path.parent
    repo_root = next(parent for parent in script_path.parents if (parent / "AGENTS.md").is_file())
    _archive_source(packet_root, repo_root)


if __name__ == "__main__":
    main()
