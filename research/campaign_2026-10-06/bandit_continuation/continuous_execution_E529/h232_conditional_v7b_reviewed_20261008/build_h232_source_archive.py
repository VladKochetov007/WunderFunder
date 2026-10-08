from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
PACKET = ROOT / "packet"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build() -> dict[str, object]:
    """Rebuild the unchanged V6 source tar and verify byte-identical ancestry."""
    baseline_archive = ROOT / "basis/V6_SOURCE_ARCHIVE.tar"
    baseline_manifest = ROOT / "basis/V6_SOURCE_ARCHIVE_MANIFEST.json"
    if not baseline_archive.is_file() or not baseline_manifest.is_file():
        raise FileNotFoundError("immutable V6 source archive basis is missing")
    baseline_archive_bytes = baseline_archive.read_bytes()
    baseline_manifest_bytes = baseline_manifest.read_bytes()
    if sha256_bytes(baseline_archive_bytes) != (
        "f1333c63eca38e69149e14dbc94c6a701ba97e113d6679b8939d7c860f1a95e4"
    ):
        raise ValueError("V6 source archive basis hash differs from the terminal packet")
    if sha256_bytes(baseline_manifest_bytes) != (
        "6ee80717ab4cec6da8b9ef3f124439a7598c2dfde2fa25de9fb35bf47a0adc42"
    ):
        raise ValueError("V6 source manifest basis hash differs from the terminal packet")
    baseline = json.loads(baseline_manifest_bytes)
    paths = sorted(
        path
        for path in SOURCE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    members: list[dict[str, object]] = []
    with tarfile.open(PACKET / "source_archive.tar", "w", format=tarfile.USTAR_FORMAT) as archive:
        for path in paths:
            relative = path.relative_to(SOURCE).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
            members.append({"path": relative, "bytes": len(data), "sha256": sha256_bytes(data)})
    manifest = {
        "schema": "h232-source-archive-manifest-v1",
        "source_root": "h232_source_v6",
        "member_count": len(members),
        "members": members,
        "source_change": {
            "changed_member": "src/wnn/data.py",
            "change": "Opt-in append-only trace of verified raw train/valid rowgroup reads and compressed byte ranges from the already-open PyArrow reader for bounded page-cache advice; no returned array or training value changes.",
            "numeric_recipe_change": False,
        },
    }
    if members != baseline["members"]:
        raise ValueError("V7 source tree differs from the immutable V6 member hashes")
    manifest_path = PACKET / "source_archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tar_bytes = (PACKET / "source_archive.tar").read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    if tar_bytes != baseline_archive_bytes or manifest_bytes != baseline_manifest_bytes:
        raise ValueError("V7 must preserve the byte-identical V6 source archive and manifest")
    return {
        "source_archive_sha256": sha256_bytes(tar_bytes),
        "source_archive_bytes": len(tar_bytes),
        "source_archive_manifest_sha256": sha256_bytes(manifest_bytes),
        "parent_v6_source_archive_sha256": sha256_bytes(baseline_archive_bytes),
        "source_archive_byte_identical_to_v6": True,
        "member_count": len(members),
        "members": members,
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
