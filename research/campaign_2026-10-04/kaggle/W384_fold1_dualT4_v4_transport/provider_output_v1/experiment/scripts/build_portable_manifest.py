"""Bind a frozen split's source identities to portable, full-file digests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.portability import sha256_file


def build(root: Path, split_manifest: Path, output: Path) -> None:
    """Write a new content registry without changing the frozen split manifest."""
    if output.exists():
        raise FileExistsError(output)
    split = json.loads(split_manifest.read_text())
    sources = {}
    for name, identity in split["sources"].items():
        path = root / name
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
            raise ValueError(f"original source metadata changed: {name}")
        sources[name] = {"original_identity": identity, "sha256": sha256_file(path)}
    payload = {
        "version": 1,
        "split_manifest_sha256": sha256_file(split_manifest),
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("x") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)


def main() -> None:
    """Build the registry from original local files, before remote transfer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.split_manifest, args.output)


if __name__ == "__main__":
    main()
