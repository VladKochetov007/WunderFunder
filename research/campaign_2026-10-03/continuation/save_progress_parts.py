"""Preserve larger research progress as independently verifiable bounded archives."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from save_progress_artifacts import INITIAL, KINDS, ROOT, STATE

LIMIT = 960 * 1024**2


class HashingReader:
    """Hash exactly the bounded byte stream written into a tar member."""

    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.checksum = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int) -> bytes:
        """Serve tarfile's chunked reads without materializing a state capture."""
        if size < 0:
            raise ValueError("unbounded archive read forbidden")
        data = self.source.read(size)
        self.checksum.update(data)
        self.bytes_read += len(data)
        return data


def main() -> None:
    """Write parts compatible with the existing paused companion restore."""
    snapshot = Path((STATE / "LATEST_SNAPSHOT").read_text().strip())
    if not snapshot.is_absolute():
        rooted = ROOT / snapshot
        snapshot = rooted if rooted.is_dir() else STATE / "snapshots" / snapshot
    if not snapshot.is_dir():
        raise FileNotFoundError("portable snapshot must exist first")
    roots = [
        ROOT / "research/campaign_2026-10-03/eda",
        ROOT / "research/campaign_2026-10-03/opt2",
        STATE / "reproduction",
        STATE / "round2",
        STATE / "round3",
        STATE / "round4",
        STATE / "round5",
        STATE / "round6",
    ]
    paths = sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix in KINDS
        and not any(part in {"pytest_tmp", "__pycache__", "baseline"} for part in path.parts)
        and path.stat().st_mtime >= INITIAL.stat().st_mtime
    )
    parts: list[list[Path]] = [[]]
    size = 0
    for path in paths:
        length = path.stat().st_size
        if length > LIMIT:
            raise ValueError(f"single artifact exceeds bounded part: {path}")
        if size + length > LIMIT:
            parts.append([])
            size = 0
        parts[-1].append(path)
        size += length
    names = [
        "progress-artifacts.tar" if index == 0 else f"progress-artifacts-{index + 1:03}.tar"
        for index in range(len(parts))
    ]
    if any((snapshot / name).exists() for name in names):
        raise FileExistsError("never overwrite an existing progress part")
    reports = []
    for index, (name, members) in enumerate(zip(names, parts, strict=True)):
        destination = snapshot / name
        partial = destination.with_suffix(".tar.partial")
        manifest = []
        with tarfile.open(partial, "w") as bundle:
            for path in members:
                before = path.stat()
                relative = str(path.relative_to(ROOT))
                info = tarfile.TarInfo(relative)
                info.size, info.mode = before.st_size, 0o644
                with path.open("rb") as source:
                    reader = HashingReader(source)
                    bundle.addfile(info, reader)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"artifact still being written: {path}")
                if reader.bytes_read != before.st_size:
                    raise ValueError(f"artifact byte count changed: {path}")
                manifest.append(
                    {
                        "path": relative,
                        "bytes": before.st_size,
                        "sha256": reader.checksum.hexdigest(),
                    }
                )
            body = json.dumps(
                {
                    "members": manifest,
                    "part": index + 1,
                    "parts": len(parts),
                    "created_at": datetime.now().astimezone().isoformat(),
                },
                indent=2,
            ).encode()
            info = tarfile.TarInfo("progress_manifest.json")
            info.size, info.mode = len(body), 0o644
            bundle.addfile(info, io.BytesIO(body))
        partial.replace(destination)
        checksum = hashlib.sha256()
        with destination.open("rb") as stored:
            for chunk in iter(lambda: stored.read(1024**2), b""):
                checksum.update(chunk)
        report = {
            "archive": name,
            "sha256": checksum.hexdigest(),
            "members": len(manifest),
            "payload_bytes": sum(entry["bytes"] for entry in manifest),
        }
        reports.append(report)
        with (snapshot / "SHA256SUMS").open("a") as output:
            output.write(f"{checksum.hexdigest()}  {name}\n")
    (snapshot / "PROGRESS_PARTS.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
