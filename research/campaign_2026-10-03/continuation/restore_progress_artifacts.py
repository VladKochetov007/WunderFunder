from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

CONTINUATION = "research/campaign_2026-10-03/continuation"
ROOTS = (
    "research/campaign_2026-10-03/eda/",
    "research/campaign_2026-10-03/opt2/",
    f"{CONTINUATION}/reproduction/",
    f"{CONTINUATION}/round2/",
    f"{CONTINUATION}/round3/",
    f"{CONTINUATION}/round4/",
    f"{CONTINUATION}/round5/",
    f"{CONTINUATION}/round6/",
)
KINDS = {".npz", ".npy", ".png", ".html", ".csv", ".pt"}
LIMIT = 1024 * 1024**2


def digest(stream: BinaryIO) -> str:
    """Hash a bounded artifact with constant-size read buffers."""
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024**2), b""):
        result.update(chunk)
    return result.hexdigest()


def restore(archive: Path, expected_sha256: str, destination: Path) -> dict[str, object]:
    """Verify and restore research artifacts only into a paused staging tree."""
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or archive.stat().st_size > LIMIT + 8 * 1024**2
    ):
        raise ValueError("invalid archive checksum/size input")
    with archive.open("rb") as source:
        if digest(source) != expected_sha256:
            raise ValueError("archive checksum mismatch")
    if destination.is_symlink():
        raise ValueError("staging destination cannot be a symlink")
    destination = destination.resolve(strict=True)
    for marker in (
        f"{CONTINUATION}/RESOURCE_FAILURE",
        "research/campaign_2026-10-01/QUEUE_PAUSE",
        "research/campaign_2026-09-29/STOP_NEW_JOBS",
    ):
        if not (destination / marker).is_file():
            raise ValueError(f"missing paused staging marker: {marker}")
    if (destination / CONTINUATION / "READY").exists():
        raise ValueError("READY must remain absent in staging")
    with tarfile.open(archive, "r:") as bundle:
        members = {}
        total = 0
        for member in bundle:
            name = member.name
            path = PurePosixPath(name)
            if (
                not member.isfile()
                or name in members
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or str(path) != name
            ):
                raise ValueError(f"unsafe/duplicate member: {name}")
            if name != "progress_manifest.json" and (
                not name.startswith(ROOTS) or path.suffix not in KINDS
            ):
                raise ValueError(f"unexpected artifact member: {name}")
            total += member.size
            if member.size < 0 or total > LIMIT + 8 * 1024**2 or len(members) >= 20000:
                raise ValueError("artifact payload exceeds bounds")
            members[name] = member
        if (
            "progress_manifest.json" not in members
            or members["progress_manifest.json"].size > 8 * 1024**2
        ):
            raise ValueError("missing/oversize artifact manifest")
        stream = bundle.extractfile(members["progress_manifest.json"])
        assert stream is not None
        manifest = json.load(stream)
        expected = {entry["path"]: entry for entry in manifest["members"]}
        if len(expected) != len(manifest["members"]) or set(expected) != set(members) - {
            "progress_manifest.json"
        }:
            raise ValueError("manifest/member mismatch")
        for name, entry in expected.items():
            stream = bundle.extractfile(members[name])
            assert stream is not None
            if members[name].size != entry["bytes"] or digest(stream) != entry["sha256"]:
                raise ValueError(f"member checksum mismatch: {name}")
            target = destination / name
            for parent in (target, *target.parents):
                if parent == destination:
                    break
                if parent.is_symlink():
                    raise ValueError(f"symlink in staging path: {parent}")
            if target.exists():
                with target.open("rb") as source:
                    if digest(source) != entry["sha256"]:
                        raise FileExistsError(
                            f"refuse to overwrite different staging artifact: {name}"
                        )
        for name in expected:
            target = destination / name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = bundle.extractfile(members[name])
            assert stream is not None
            with target.open("xb") as output:
                shutil.copyfileobj(stream, output, 1024**2)
            target.chmod(0o644)
    result = {
        "status": "PASS",
        "paused": True,
        "members": len(expected),
        "payload_bytes": sum(entry["bytes"] for entry in expected.values()),
        "sha256": expected_sha256,
        "services_started": 0,
    }
    (destination / "_recovery/progress_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (destination / "_recovery/progress_dry_run.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    """Restore verified artifacts without executing restored code or starting jobs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--restore-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, args.expected_sha256, args.restore_dir)))


if __name__ == "__main__":
    main()
