"""Create a selective FP16 disk-transport child of an existing Q8 package."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

CONVERTED_TENSORS = {
    "gru.weight_ih_l0",
    "gru.weight_hh_l0",
    "gru.weight_ih_l1",
    "gru.weight_hh_l1",
    "head.0.weight",
}


def sha256_bytes(value: bytes) -> str:
    """Hash one byte payload."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(array: np.ndarray) -> str:
    """Hash a contiguous C-order tensor payload."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def validate_archive_paths(names: list[str]) -> None:
    """Reject duplicate and unsafe ZIP member paths."""
    if len(names) != len(set(names)):
        raise ValueError("source package contains duplicate ZIP members")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"unsafe package path: {name!r}")


def convert_npz(payload: bytes, member_name: str) -> tuple[bytes, dict[str, object]]:
    """Round only the five selected dense weights to IEEE binary16 storage."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        names = list(archive.files)
        if len(names) != len(set(names)):
            raise ValueError(f"{member_name} contains duplicate NPZ tensor names")
        missing = CONVERTED_TENSORS - set(names)
        if missing:
            raise ValueError(f"{member_name} lacks required dense matrices: {sorted(missing)}")
        arrays: dict[str, np.ndarray] = {}
        tensor_records: list[dict[str, object]] = []
        for name in names:
            source = np.asarray(archive[name])
            if source.dtype != np.dtype("float32"):
                raise ValueError(f"{member_name}:{name} source dtype is {source.dtype}, not f32")
            original = np.ascontiguousarray(source)
            if not np.isfinite(original).all():
                raise ValueError(f"{member_name}:{name} contains non-finite source values")
            if name in CONVERTED_TENSORS:
                stored = np.ascontiguousarray(original.astype(np.float16))
                if not np.isfinite(stored).all():
                    raise ValueError(f"{member_name}:{name} overflows in binary16 storage")
            else:
                stored = original
            arrays[name] = stored
            tensor_records.append(
                {
                    "name": name,
                    "shape": list(original.shape),
                    "source_dtype": original.dtype.str,
                    "source_tensor_sha256": tensor_sha256(original),
                    "stored_dtype": stored.dtype.str,
                    "stored_tensor_sha256": tensor_sha256(stored),
                    "converted": name in CONVERTED_TENSORS,
                    "max_abs_rounding_error": (
                        float(np.max(np.abs(original - stored.astype(np.float32))))
                        if name in CONVERTED_TENSORS
                        else 0.0
                    ),
                }
            )
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue(), {"member": member_name, "tensors": tensor_records}


def build_transport(
    source_zip: Path,
    source_receipt: Path,
    destination_zip: Path,
    destination_receipt: Path,
    max_bytes: int,
    candidate_id: str | None,
) -> dict[str, object]:
    """Write a create-only ZIP whose loader expands transported weights to FP32."""
    for path in (source_zip, source_receipt):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination_zip.exists() or destination_receipt.exists():
        raise FileExistsError("refusing to replace an existing transport package or receipt")
    source = json.loads(source_receipt.read_text())
    if source.get("schema") != "wnn.submission-build-receipt.v1":
        raise ValueError("source package receipt must be the original package build receipt")
    source_hash = sha256_file(source_zip)
    if source.get("artifact", {}).get("sha256") != source_hash:
        raise ValueError("source ZIP hash differs from its package receipt")
    if source.get("precision") != "q8vnni":
        raise ValueError("selective transport requires an actual Q8 package source")

    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    destination_receipt.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as archive:
        names = archive.namelist()
        validate_archive_paths(names)
        source_members = {name: archive.read(name) for name in names}
    member_spec = json.loads(source_members["members.json"])
    if not isinstance(member_spec, list) or not member_spec:
        raise ValueError("source package has no fixed ensemble members")
    if source_members["precision.txt"].decode("utf-8").strip() != "q8vnni":
        raise ValueError("package precision.txt does not request Q8VNNI")
    if b"precision=PRECISION" not in source_members["solution.py"]:
        raise ValueError("package solution does not route the precision file into the native API")

    output_members = dict(source_members)
    conversion_records: list[dict[str, object]] = []
    converted_files = {member["weights"] for member in member_spec}
    for member_name in sorted(converted_files):
        if member_name not in source_members:
            raise ValueError(f"member weight payload missing from source ZIP: {member_name}")
        converted, record = convert_npz(source_members[member_name], member_name)
        output_members[member_name] = converted
        conversion_records.append(record)

    destination_zip_tmp = destination_zip.with_name(destination_zip.name + ".partial")
    if destination_zip_tmp.exists():
        raise FileExistsError(destination_zip_tmp)
    try:
        with zipfile.ZipFile(
            destination_zip_tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in sorted(output_members):
                archive.writestr(name, output_members[name])
        if destination_zip_tmp.stat().st_size > max_bytes:
            raise ValueError(
                f"transport ZIP is {destination_zip_tmp.stat().st_size} bytes; limit is {max_bytes}"
            )
        with zipfile.ZipFile(destination_zip_tmp) as archive:
            if archive.testzip() is not None:
                raise ValueError("transport ZIP failed its CRC check")
            actual_names = archive.namelist()
            validate_archive_paths(actual_names)
            if set(actual_names) != set(source_members):
                raise ValueError("transport ZIP changed the package file set")
            payload_hashes = {
                name: sha256_bytes(archive.read(name)) for name in sorted(actual_names)
            }
        preserved = [name for name in sorted(source_members) if name not in converted_files]
        mismatches = [
            name for name in preserved if payload_hashes[name] != sha256_bytes(source_members[name])
        ]
        if mismatches:
            raise ValueError(f"non-weight package payloads changed: {mismatches}")
        os.link(destination_zip_tmp, destination_zip)
    finally:
        destination_zip_tmp.unlink(missing_ok=True)

    converter_path = Path(__file__).resolve()
    receipt: dict[str, object] = {
        "schema": "wnn.selective_fp16_weight_transport_receipt.r1",
        "candidate": candidate_id or source.get("candidate"),
        "precision": "q8vnni",
        "interpretation": (
            "Selected NPZ matrices are stored as IEEE binary16 and loaded back as FP32 before "
            "the unchanged Q8VNNI native backend. This is disk transport with one rounding step, "
            "not native FP16 inference."
        ),
        "source_package": {
            "path": str(source_zip.resolve()),
            "sha256": source_hash,
            "bytes": source_zip.stat().st_size,
            "build_receipt_path": str(source_receipt.resolve()),
            "build_receipt_sha256": sha256_file(source_receipt),
        },
        "fixed_members": source["fixed_members"],
        "transport": {
            "converted_tensor_names": sorted(CONVERTED_TENSORS),
            "converted_member_files": sorted(converted_files),
            "member_records": conversion_records,
            "unchanged_package_files": [
                {"path": name, "sha256": sha256_bytes(source_members[name])} for name in preserved
            ],
            "solution_sha256": payload_hashes["solution.py"],
            "precision_file": source_members["precision.txt"].decode("utf-8").strip(),
        },
        "converter": {"path": str(converter_path), "sha256": sha256_file(converter_path)},
        "artifact": {
            "path": str(destination_zip.resolve()),
            "bytes": destination_zip.stat().st_size,
            "sha256": sha256_file(destination_zip),
            "members": payload_hashes,
        },
    }
    with destination_receipt.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Create one FP16-storage transport variant from an exact Q8 package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--destination-zip", type=Path, required=True)
    parser.add_argument("--destination-receipt", type=Path, required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--max-bytes", type=int, default=20_000_000)
    args = parser.parse_args()
    receipt = build_transport(
        args.source_zip,
        args.source_receipt,
        args.destination_zip,
        args.destination_receipt,
        args.max_bytes,
        args.candidate_id,
    )
    print(
        json.dumps(
            {"artifact": receipt["artifact"], "candidate": receipt["candidate"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
