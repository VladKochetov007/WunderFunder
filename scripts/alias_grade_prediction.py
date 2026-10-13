"""Bind a custom-named full-valid export to the run-name path consumed by its grader.

The prediction exporter can write ``pred_<output-name>_valid.npy`` while the
fold grader reads ``pred_<run>_valid.npy``. This command validates the producer
sidecar, checkpoint, split and NPY format, then creates a zero-copy hardlink
pair without changing either artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_SHAPE = (1873, 20_000, 2)
HASH_BLOCK_BYTES = 8 << 20
MAX_NPY_HEADER_BYTES = 10_000
MAX_SIDECAR_BYTES = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype[Any], bool, int]:
    """Read a small NPY header only after bounding its declared length."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
            raise ValueError(f"invalid NPY magic in {path}")
        version = tuple(prefix[6:8])
        if version == (1, 0):
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                raise ValueError(f"truncated NPY header length in {path}")
            header_length = struct.unpack("<H", length_bytes)[0]
            prefix_bytes = 10
        elif version in {(2, 0), (3, 0)}:
            length_bytes = stream.read(4)
            if len(length_bytes) != 4:
                raise ValueError(f"truncated NPY header length in {path}")
            header_length = struct.unpack("<I", length_bytes)[0]
            prefix_bytes = 12
        else:
            raise ValueError(f"unsupported NPY version {version}")
        if not 0 < header_length <= MAX_NPY_HEADER_BYTES:
            raise ValueError(
                f"NPY header exceeds the {MAX_NPY_HEADER_BYTES}-byte bound: {header_length}"
            )
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"truncated NPY header in {path}")

    encoding = "utf-8" if version == (3, 0) else "latin1"
    try:
        header = ast.literal_eval(header_bytes.decode(encoding).strip())
    except (SyntaxError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid NPY header in {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"NPY header is not a dictionary in {path}")
    try:
        raw_shape = header["shape"]
        if not isinstance(raw_shape, tuple) or any(
            type(dimension) is not int or dimension < 0 for dimension in raw_shape
        ):
            raise ValueError("shape must contain nonnegative integers")
        shape = tuple(raw_shape)
        dtype = np.dtype(header["descr"])
        fortran_order = header["fortran_order"]
        if type(fortran_order) is not bool:
            raise ValueError("fortran_order must be a boolean")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid NPY header fields in {path}: {error}") from error
    return shape, dtype, fortran_order, prefix_bytes + header_length


def _simple_name(value: str, flag: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{flag} must be a plain filename component")
    return value


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")


def _verify_same_file(source: Path, destination: Path) -> None:
    _require_regular_file(source)
    _require_regular_file(destination)
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (source_stat.st_dev, source_stat.st_ino) != (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        raise ValueError(f"destination is not the producer artifact's hardlink: {destination}")


def _link_pair(source: Path, sidecar: Path, destination: Path, destination_sidecar: Path) -> str:
    source_exists = os.path.lexists(destination)
    sidecar_exists = os.path.lexists(destination_sidecar)
    if source_exists or sidecar_exists:
        if not (source_exists and sidecar_exists):
            raise FileExistsError("only one grader alias path exists; refusing partial repair")
        _verify_same_file(source, destination)
        _verify_same_file(sidecar, destination_sidecar)
        return "verified-existing-hardlinks"

    os.link(source, destination, follow_symlinks=False)
    try:
        os.link(sidecar, destination_sidecar, follow_symlinks=False)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    _verify_same_file(source, destination)
    _verify_same_file(sidecar, destination_sidecar)
    return "created-hardlinks"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="project root containing cache/ and artifacts/"
    )
    parser.add_argument(
        "--run",
        required=True,
        help="run name embedded in the prediction sidecar and expected by the grader",
    )
    parser.add_argument(
        "--export-name", required=True, help="custom output name used by predict_run.py"
    )
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="fixed inference checkpoint used for export"
    )
    parser.add_argument("--mode", choices=("create", "verify-only"), required=True)
    args = parser.parse_args()

    run = _simple_name(args.run, "--run")
    export_name = _simple_name(args.export_name, "--export-name")
    if args.epoch < 0:
        raise ValueError("--epoch must be nonnegative")

    root = args.root.resolve(strict=True)
    cache = (root / "cache").resolve(strict=True)
    if cache.parent != root:
        raise ValueError("project cache path escaped the selected root")
    source = cache / f"pred_{export_name}_valid.npy"
    source_sidecar = source.with_suffix(".json")
    destination = cache / f"pred_{run}_valid.npy"
    destination_sidecar = destination.with_suffix(".json")
    split_manifest = args.split_manifest.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    for path in (source, source_sidecar, split_manifest, checkpoint):
        _require_regular_file(path)

    shape, dtype, fortran_order, header_bytes = _npy_header(source)
    if shape != EXPECTED_SHAPE:
        raise ValueError(
            f"unexpected full-valid prediction shape {shape}; expected {EXPECTED_SHAPE}"
        )
    if dtype != np.dtype(np.float32) or dtype.hasobject or fortran_order:
        raise ValueError(
            f"unexpected prediction dtype/layout: dtype={dtype}, fortran={fortran_order}"
        )
    expected_size = header_bytes + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if source.stat().st_size != expected_size:
        raise ValueError("NPY file size does not match its header and uncompressed payload")

    sidecar_size = source_sidecar.stat().st_size
    if sidecar_size > MAX_SIDECAR_BYTES:
        raise ValueError(f"prediction sidecar exceeds the {MAX_SIDECAR_BYTES}-byte metadata bound")
    sidecar_bytes = source_sidecar.read_bytes()
    sidecar = json.loads(sidecar_bytes)
    if not isinstance(sidecar, dict):
        raise ValueError("prediction sidecar must be a JSON object")
    expected = {
        "run": run,
        "split": "valid",
        "epoch": args.epoch,
        "split_manifest_sha256": _sha256(split_manifest),
        "checkpoint_sha256": _sha256(checkpoint),
        "inference_precision": "fp32",
    }
    disagreements = {
        key: (sidecar.get(key), value)
        for key, value in expected.items()
        if sidecar.get(key) != value
    }
    if disagreements:
        raise ValueError(f"prediction sidecar is bound to a different export: {disagreements}")
    row_groups = sidecar.get("row_groups")
    if (
        not isinstance(row_groups, list)
        or len(row_groups) != EXPECTED_SHAPE[0]
        or any(type(group) is not int for group in row_groups)
        or len(set(row_groups)) != EXPECTED_SHAPE[0]
    ):
        raise ValueError("prediction sidecar has invalid full-valid row groups")
    if (
        sidecar.get("evaluation_membership_sha256") is not None
        or sidecar.get("fit_subset_manifest_sha256") is not None
    ):
        raise ValueError("full-valid export unexpectedly carries subset-only membership metadata")

    prediction_sha = _sha256(source)
    if sidecar.get("prediction_sha256") != prediction_sha:
        raise ValueError("prediction bytes do not match the producer sidecar hash")

    if args.mode == "verify-only":
        if not os.path.lexists(destination) or not os.path.lexists(destination_sidecar):
            raise FileNotFoundError("verify-only requires both grader alias paths")
        mode_result = _link_pair(source, source_sidecar, destination, destination_sidecar)
    else:
        mode_result = _link_pair(source, source_sidecar, destination, destination_sidecar)

    if destination_sidecar.read_bytes() != sidecar_bytes:
        raise ValueError("grader alias sidecar bytes differ from the producer sidecar")

    print(
        json.dumps(
            {
                "status": "pass",
                "mode": args.mode,
                "alias": mode_result,
                "run": run,
                "epoch": args.epoch,
                "shape": shape,
                "dtype": dtype.str,
                "bytes": source.stat().st_size,
                "prediction_sha256": prediction_sha,
                "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                "checkpoint_sha256": expected["checkpoint_sha256"],
                "split_manifest_sha256": expected["split_manifest_sha256"],
                "data_hardlink": os.path.samefile(source, destination),
                "sidecar_hardlink": os.path.samefile(source_sidecar, destination_sidecar),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
