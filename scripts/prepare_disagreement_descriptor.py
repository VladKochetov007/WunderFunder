from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Write a descriptor pinning three existing teacher-cache manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != 3:
        raise ValueError("exactly three --input directories are required")
    roots = [path.resolve(strict=True) for path in args.input]
    if len(set(roots)) != 3:
        raise ValueError("teacher directories must be distinct")
    inputs = []
    for root in roots:
        manifest = root / "manifest.json"
        if manifest.is_symlink() or manifest.stat().st_size > 32_000_000:
            raise ValueError("invalid teacher manifest")
        with manifest.open("rb") as stream:
            raw = stream.read(32_000_001)
        if len(raw) > 32_000_000:
            raise ValueError("teacher manifest exceeds maximum size")
        inputs.append({"path": str(root), "manifest_sha256": hashlib.sha256(raw).hexdigest()})
    args.output.mkdir(parents=False, exist_ok=False)
    descriptor = {"schema": "sequence-disagreement-input-v1", "inputs": inputs}
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(descriptor, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
