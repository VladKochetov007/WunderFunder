#!/usr/bin/env python3
"""Create a new submission ZIP replacing only its v3 native extension."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

TARGET_MEMBER = "v3/wnn_native.abi3.so"
MAX_SUBMISSION_BYTES = 20_000_000


def sha256_bytes(data: bytes) -> str:
    """Hash a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_inventory(path: Path) -> dict[str, str]:
    """Return uncompressed member hashes after CRC validation."""
    result: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC check failed for ZIP member {bad_member!r}")
        for info in archive.infolist():
            if info.filename in result:
                raise ValueError(f"duplicate ZIP member {info.filename!r}")
            result[info.filename] = sha256_bytes(archive.read(info))
    return result


def main() -> None:
    """Write an immutable optimized ZIP and a source-qualified receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--optimization-record", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()

    base_zip = args.base_zip.resolve(strict=True)
    base_receipt_path = args.base_receipt.resolve(strict=True)
    library = args.library.resolve(strict=True)
    native_source = args.native_source.resolve(strict=True)
    optimization_record_path = args.optimization_record.resolve(strict=True)
    optimization_record: dict[str, Any] = json.loads(optimization_record_path.read_text())
    output_zip = args.output_zip.resolve()
    output_receipt_path = args.output_receipt.resolve()
    if output_zip.exists() or output_receipt_path.exists():
        raise FileExistsError("refusing to overwrite optimized ZIP or receipt")
    if output_zip == base_zip or output_receipt_path == base_receipt_path:
        raise ValueError("optimized outputs must not replace their inputs")

    base_receipt: dict[str, Any] = json.loads(base_receipt_path.read_text())
    base_zip_sha256 = sha256_file(base_zip)
    if base_receipt["artifact"]["sha256"] != base_zip_sha256:
        raise ValueError("base ZIP does not match its build receipt")
    base_members = zip_inventory(base_zip)
    if base_members != base_receipt["artifact"]["members"]:
        raise ValueError("base ZIP members do not match its build receipt")
    if TARGET_MEMBER not in base_members:
        raise ValueError(f"base ZIP has no {TARGET_MEMBER!r}")
    if base_receipt.get("precision") != "q8vnni":
        raise ValueError("this transformer only accepts the reviewed Q8VNNI package")

    library_sha256 = sha256_file(library)
    source_files = sorted(
        path for path in native_source.rglob("*") if path.is_file() and "target" not in path.parts
    )
    native_source_hashes = {
        path.relative_to(native_source).as_posix(): sha256_file(path) for path in source_files
    }
    if not native_source_hashes:
        raise ValueError("native source tree is empty")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(base_zip) as input_archive,
        zipfile.ZipFile(output_zip, mode="x", compression=zipfile.ZIP_DEFLATED) as output_archive,
    ):
        for info in input_archive.infolist():
            data = (
                library.read_bytes() if info.filename == TARGET_MEMBER else input_archive.read(info)
            )
            output_archive.writestr(info, data)

    if output_zip.stat().st_size > MAX_SUBMISSION_BYTES:
        output_zip.unlink()
        raise ValueError(
            f"optimized ZIP is {output_zip.stat().st_size} bytes, over the 20 MB limit"
        )
    output_members = zip_inventory(output_zip)
    if set(output_members) != set(base_members):
        raise ValueError("optimized ZIP changed its member-name set")
    changed = {name for name in base_members if base_members[name] != output_members[name]}
    if changed != {TARGET_MEMBER} or output_members[TARGET_MEMBER] != library_sha256:
        raise ValueError(f"unexpected member changes: {sorted(changed)}")

    receipt = copy.deepcopy(base_receipt)
    receipt["artifact"].update(
        {
            "path": str(output_zip),
            "bytes": output_zip.stat().st_size,
            "sha256": sha256_file(output_zip),
            "members": output_members,
        }
    )
    receipt.setdefault("native_runtime", {}).setdefault("libraries", {})[TARGET_MEMBER] = (
        library_sha256
    )
    optimization_record["base_zip_sha256"] = base_zip_sha256
    optimization_record["base_build_receipt_sha256"] = sha256_file(base_receipt_path)
    optimization_record["only_changed_zip_member"] = TARGET_MEMBER
    optimization_record["new_native_library_sha256"] = library_sha256
    optimization_record["native_source_files_sha256"] = native_source_hashes
    optimization_record["optimization_record_sha256"] = sha256_file(optimization_record_path)
    receipt["runtime_optimization"] = optimization_record
    output_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_receipt_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "zip": str(output_zip),
                "zip_sha256": receipt["artifact"]["sha256"],
                "bytes": receipt["artifact"]["bytes"],
                "receipt": str(output_receipt_path),
                "native_library_sha256": library_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
