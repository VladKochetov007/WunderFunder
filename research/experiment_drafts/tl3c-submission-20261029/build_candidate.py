from __future__ import annotations

import argparse
import ast
import hashlib
import logging
import platform
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from candidate_contract import PR49_COMMIT, load_manifest
from candidate_io import JsonObject, file_identity, sha256, write_json
from checkpoint_export import export_member

RUNTIME_SHA256 = "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569"
NATIVE_NAME = "wnn_cpp.cpython-311-x86_64-linux-gnu.so"
LIBRARY_SHA256 = {
    "avx2": "4be7fa2ba2c17dea7d8631d6bb0667fd5a592d23321ebd81ee9f0a297359c7db",
    "v2": "3f36627c32e56f0d2f26de66e565f7a88bf58ba1590abb6e8d3094effd17f717",
    "v3": "3cb4546229e80cca81cda9cdec1b9c6146b380271077e26f9324d8a4b6ecad9a",
}
LIBRARY_BYTES = {"avx2": 874656, "v2": 879032, "v3": 883320}
MAX_ZIP_BYTES = 20_000_000
PENDING_GATES = [
    "python311_abi",
    "opaque_native_semantics",
    "chunk4000_parity",
    "full_valid_score",
    "teacher_prediction_consumption_lineage",
    "sequence_reset_and_causality",
    "cpu_1vcpu_budget",
    "rss_16gb",
    "portal_result",
]


def copy_native_binaries(archive_path: Path, stage: Path) -> list[JsonObject]:
    """Copy only three hash-pinned ELF libraries; historical weights/wrapper are never read."""
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("runtime archive must be a regular nonsymlink file")
    historical_archive = sha256(archive_path) == RUNTIME_SHA256
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        expected_names = {f"{level}/{NATIVE_NAME}" for level in LIBRARY_SHA256}
        if not historical_archive and (
            set(archive.namelist()) != expected_names or len(archive.namelist()) != 3
        ):
            raise ValueError("use the pinned historical archive or an exact three-library handoff")
        for level, digest in LIBRARY_SHA256.items():
            name = f"{level}/{NATIVE_NAME}"
            if archive.namelist().count(name) != 1:
                raise ValueError(f"runtime library missing or duplicated: {name}")
            if archive.getinfo(name).file_size != LIBRARY_BYTES[level]:
                raise ValueError(f"runtime library size differs: {name}")
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"runtime library hash differs: {name}")
            if payload[:6] != b"\x7fELF\x02\x01" or payload[18:20] != b"\x3e\x00":
                raise ValueError(f"expected a little-endian 64-bit x86-64 ELF library: {name}")
            destination = stage / name
            destination.parent.mkdir()
            with destination.open("xb") as stream:
                stream.write(payload)
            records.append({"path": name, "sha256": digest, "bytes": len(payload)})
    return records


def write_archive(stage: Path, destination: Path, expected: set[str]) -> JsonObject:
    """Build a deterministic flat-root ZIP and inspect names, sizes and CRC without importing it."""
    files = sorted(path for path in stage.rglob("*") if path.is_file())
    if {path.relative_to(stage).as_posix() for path in files} != expected:
        raise ValueError("unexpected submission staging content")
    with zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError("submission files must not be symlinks")
            info = zipfile.ZipInfo(
                path.relative_to(stage).as_posix(), date_time=(2026, 10, 29, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    size = destination.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise ValueError(f"ZIP {size} bytes exceeds the {MAX_ZIP_BYTES}-byte submission cap")
    with zipfile.ZipFile(destination) as archive:
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError("ZIP layout differs from the frozen export")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failed: {bad_member}")
    return {"path": destination.name, "sha256": sha256(destination), "bytes": size}


def build(args: argparse.Namespace) -> None:
    """Export fresh receiver checkpoints into a candidate that still requires numerical gates."""
    manifest, split = load_manifest(args.manifest)
    # Reserve an entirely new directory. Failures preserve artifacts for inspection, never overwrite.
    output = args.output_dir.resolve()
    output.mkdir(parents=False, exist_ok=False)
    stage = output / "submission"
    stage.mkdir()
    source = Path(__file__).resolve().parent
    wrapper = source / "solution.py"
    ast.parse(wrapper.read_text(encoding="utf-8"), filename="solution.py", feature_version=(3, 11))
    shutil.copyfile(wrapper, stage / "solution.py")
    libraries = copy_native_binaries(args.runtime_archive, stage)
    selected = (
        list(zip(manifest["members"], (0.34, 0.33, 0.33), strict=True))
        if args.candidate == "blend"
        else [(manifest["members"][1], 1.0)]
    )
    members = [export_member(record, split, stage, coef) for record, coef in selected]
    write_json(
        stage / "members.json",
        {
            "schema": "tl3c-native-members-v1",
            "mode": args.mode,
            "members": [
                {
                    "run": member["run"],
                    "coef": member["coef"],
                    "family": member["family"],
                    "weights": member["weight_file"],
                }
                for member in members
            ],
        },
    )
    layout = {"solution.py", "members.json"}
    layout.update(str(member["weight_file"]) for member in members)
    layout.update(str(library["path"]) for library in libraries)
    artifact = write_archive(stage, output / "submission.zip", layout)
    receipt = {
        "schema": "tl3c-submission-candidate-v1",
        "gate_status": "pending",
        "created_at": datetime.now(UTC).isoformat(),
        "artifact": artifact,
        "candidate": args.candidate,
        "mode": args.mode,
        "members": members,
        "input_manifest": file_identity(args.manifest),
        "split_manifest": manifest["split_manifest"],
        "training_reference_commit": PR49_COMMIT,
        "runtime_archive": file_identity(args.runtime_archive),
        "native_libraries": libraries,
        "source_sha256": {path.name: sha256(path) for path in sorted(source.glob("*.py"))},
        "files": {name: file_identity(stage / name) for name in sorted(layout)},
        "structural_checks": {
            "zip_crc": "pass",
            "zip_layout": "pass",
            "size_limit": "pass",
            "python311_source_syntax": "pass",
            "max_zip_bytes": MAX_ZIP_BYTES,
        },
        "arithmetic": {
            "fast_activations": args.mode == "dynamic_int8",
            "half_weights": False,
            "dynamic_int8": args.mode == "dynamic_int8",
            "quant_backend": "auto",
        },
        "native_semantics": "opaque historical binary; schema compatibility is not numerical parity",
        "known_parity_risk": "tiedlead training evaluation resets raw deltas at each 4000-row chunk",
        "lineage_scope": "declared parent/initializer bytes verified; teacher-cache consumption unproven",
        "metrics": None,
        "pending_gates": PENDING_GATES,
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    write_json(output / "receipt.json", receipt)
    logging.info(
        "Prepared %s (%d bytes); numerical/runtime gates remain pending", output, artifact["bytes"]
    )


def main() -> None:
    """Parse an explicit receiver packaging request; never train, infer, or upload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runtime-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("f32", "dynamic_int8"))
    parser.add_argument("--candidate", default="blend", choices=("blend", "tiedlead-s5"))
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
