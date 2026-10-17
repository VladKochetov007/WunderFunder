"""Bind relocated campaign inputs to freshly verified full-file content hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def build_registry(root: Path, split: Path, output: Path) -> None:
    """Hash each source without loading its payload or modifying its metadata."""
    if output.exists():
        raise FileExistsError(output)
    if split.stat().st_size > 2_000_000:
        raise ValueError("split manifest exceeds the known metadata bound")
    raw = split.read_bytes()
    manifest = json.loads(raw)
    registry = {
        "version": 1,
        "split_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "created_utc": datetime.now(UTC).isoformat(),
        "sources": {},
    }
    for name, identity in manifest["sources"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source name must remain within the logical data root")
        path = root / relative
        before = path.stat()
        if before.st_size != identity["bytes"] or before.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError(f"source metadata changed before hashing: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"source changed while hashing: {name}")
        value = digest.hexdigest()
        if identity.get("sha256", value) != value:
            raise ValueError(f"source hash differs from the frozen split: {name}")
        registry["sources"][name] = {"original_identity": identity, "sha256": value}
        print(json.dumps({"source": name, "bytes": before.st_size, "sha256": value}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(registry, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    """Create the content registry for the existing portability verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_registry(args.root, args.split, args.output)


if __name__ == "__main__":
    main()
