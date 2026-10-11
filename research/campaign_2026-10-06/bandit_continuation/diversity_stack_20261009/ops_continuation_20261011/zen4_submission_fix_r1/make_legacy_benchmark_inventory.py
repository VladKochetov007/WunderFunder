#!/usr/bin/env python3
"""Create non-provenance inventory metadata for timing legacy submission ZIPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO


def digest_stream(stream: BinaryIO) -> str:
    """Hash a ZIP member using bounded reads."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Write a benchmark-only manifest derived from exact legacy ZIP bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive_path = args.zip.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    zip_hash = hashlib.sha256()
    with archive_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            zip_hash.update(chunk)

    member_hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("legacy ZIP CRC check failed")
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or not name.parts:
                raise ValueError(f"unsafe ZIP path: {info.filename!r}")
            if info.is_dir():
                member_hashes[info.filename] = hashlib.sha256(b"").hexdigest()
                continue
            with archive.open(info) as member:
                member_hashes[info.filename] = digest_stream(member)
        fixed_members = json.loads(archive.read("members.json"))
        precision = archive.read("precision.txt").decode("ascii").strip()
        solution_sha256 = member_hashes["solution.py"]

    if precision != "q8vnni":
        raise ValueError(f"expected a shipped Q8VNNI ZIP, found {precision!r}")
    if not isinstance(fixed_members, list) or not fixed_members:
        raise ValueError("members.json must contain a nonempty member list")
    for member in fixed_members:
        weight_path = member.get("weights")
        if weight_path not in member_hashes:
            raise ValueError(f"members.json refers to missing weights: {weight_path!r}")

    inventory = {
        "schema": "wnn.legacy-zip-benchmark-inventory.v1",
        "purpose": "Adapter for the fixed local timing harness only; derived from these ZIP bytes and not original build provenance.",
        "candidate": args.candidate,
        "artifact": {"sha256": zip_hash.hexdigest(), "members": member_hashes},
        "fixed_members": fixed_members,
        "members": [
            {
                "run_reference": member.get("run_reference", member.get("source")),
                "sources": [],
            }
            for member in fixed_members
        ],
        "precision": precision,
        "gate_status": "package_crc_passed; offline_import_and_streaming_gate_pending",
        "source_code": {"sha256": solution_sha256},
        "native_runtime": {
            "source": "native libraries and precision file hashed from ZIP entries",
            "libraries": {
                name: digest
                for name, digest in member_hashes.items()
                if name.endswith("/wnn_native.abi3.so")
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"zip_sha256": zip_hash.hexdigest(), "inventory": str(output_path)}))


if __name__ == "__main__":
    main()
