from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = Path(__file__).resolve().parent
INITIAL = STATE / "snapshots/2026-10-03-153300/research-session.tar"
KINDS = {".npz", ".npy", ".png", ".html", ".csv", ".pt"}


def main() -> None:
    """Back up new numerical/visual research artifacts without duplicating raw history."""
    snapshot = STATE / "snapshots" / (STATE / "LATEST_SNAPSHOT").read_text().strip()
    # LATEST_SNAPSHOT may contain an absolute path or its directory basename.
    if not snapshot.is_dir():
        snapshot = Path((STATE / "LATEST_SNAPSHOT").read_text().strip())
    if not snapshot.is_dir():
        raise FileNotFoundError("portable snapshot must be saved first")
    destination = snapshot / "progress-artifacts.tar"
    if destination.exists():
        raise FileExistsError(destination)
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
    total = sum(path.stat().st_size for path in paths)
    if total > 512 * 1024**2:
        raise ValueError(f"progress payload exceeds bounded512MiB: {total}; use save_progress_parts.py to preserve all artifacts in bounded parts")
    partial = destination.with_suffix(".tar.partial")
    manifest = []
    with tarfile.open(partial, "w") as bundle:
        for path in paths:
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"artifact still being written: {path}")
            relative = str(path.relative_to(ROOT))
            digest = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(data), 0o644
            bundle.addfile(info, io.BytesIO(data))
            manifest.append({"path": relative, "bytes": len(data), "sha256": digest})
        body = json.dumps(
            {
                "members": manifest,
                "initial_private_archive": str(INITIAL),
                "created_at": datetime.now().astimezone().isoformat(),
            },
            indent=2,
        ).encode()
        info = tarfile.TarInfo("progress_manifest.json")
        info.size, info.mode = len(body), 0o644
        bundle.addfile(info, io.BytesIO(body))
    partial.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stored:
        for chunk in iter(lambda: stored.read(1024**2), b""):
            digest.update(chunk)
    with (snapshot / "SHA256SUMS").open("a") as output:
        output.write(f"{digest.hexdigest()}  progress-artifacts.tar\n")
    print(
        json.dumps(
            {
                "archive": str(destination),
                "artifacts": len(paths),
                "payload_bytes": total,
                "sha256": digest.hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
