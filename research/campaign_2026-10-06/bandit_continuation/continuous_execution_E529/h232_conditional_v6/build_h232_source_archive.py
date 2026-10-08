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
    """Create a deterministic V6 source tar and member-identity manifest."""
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
    manifest_path = PACKET / "source_archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tar_bytes = (PACKET / "source_archive.tar").read_bytes()
    return {
        "source_archive_sha256": sha256_bytes(tar_bytes),
        "source_archive_bytes": len(tar_bytes),
        "source_archive_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "member_count": len(members),
        "members": members,
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
