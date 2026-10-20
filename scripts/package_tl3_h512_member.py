"""Create a TL3 ZIP with one provenance-bound H512 student member replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
from package_submission import checkpoint_path, load_checkpoint_member, write_json_create_only
from team_member_export_lib import MemberArchitecture, infer_architecture
from tl3_source_archive import (
    SOURCE_ARCHIVE,
    SOURCE_MEMBER,
    extract_verified_entries,
    file_sha256,
    resolve_create_only_output_dir,
    verify_source_archive,
)

ROOT = Path(__file__).resolve().parents[1]
MAX_ZIP_BYTES = 20_000_000
EXPECTED_ARCHITECTURE = MemberArchitecture(211, 512, 1, 512, 2)
EXPECTED_MEMBERS = ("weights_s5.npz", "weights_tl5.npz", "weights_tl6.npz")
MODEL_KEYS = (
    {"mean", "std", "inp.weight", "inp.bias"}
    | {f"head.{layer}.{tensor}" for layer in (0, 2) for tensor in ("weight", "bias")}
    | {f"gru.{kind}_l0" for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")}
)


def _assert_fp32_checkpoint(run_reference: str, runs_dir: Path) -> Path:
    import torch

    path = checkpoint_path(run_reference, runs_dir)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args")
    state = checkpoint.get("model")
    if not isinstance(args, dict) or args.get("precision") != "fp32":
        raise ValueError("checkpoint must declare strict FP32 training")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no model state dictionary")
    if any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.float32
        for value in state.values()
    ):
        raise ValueError("checkpoint model tensors must all be FP32")
    del checkpoint, state
    return path


def _validate_member(
    state: dict[str, np.ndarray], source_record: dict[str, object]
) -> dict[str, object]:
    if set(state) != MODEL_KEYS:
        raise ValueError("checkpoint state keys differ from the canonical plain GRU")
    if any(value.dtype != np.float32 for value in state.values()):
        raise ValueError("exported checkpoint tensors must be FP32")
    if any(not np.isfinite(value).all() for value in state.values()):
        raise ValueError("exported checkpoint contains nonfinite tensors")
    if np.any(state["std"] <= 0):
        raise ValueError("input normalization std must be strictly positive")
    shapes = {name: tuple(value.shape) for name, value in state.items()}
    architecture = infer_architecture(shapes)
    if architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(f"expected {EXPECTED_ARCHITECTURE}, got {architecture}")
    layout = source_record.get("input_layout")
    if not isinstance(layout, dict) or layout.get("input_width") != 211:
        raise ValueError("checkpoint metadata does not bind the 211-input member")
    if layout.get("representation_mode") != "none":
        raise ValueError("checkpoint must use the plain base input representation")
    return layout


def _replace_student_name(
    raw_members: bytes, run_reference: str
) -> tuple[bytes, list[dict[str, object]]]:
    original = json.loads(raw_members)
    if (
        not isinstance(original, list)
        or len(original) != 3
        or any(not isinstance(member, dict) for member in original)
    ):
        raise ValueError("source member map must contain exactly three members")
    if tuple(member.get("weights") for member in original) != EXPECTED_MEMBERS:
        raise ValueError("source member order differs from the pinned TL3 package")
    expected_coefficients = (0.34, 0.33, 0.33)
    for member, coefficient in zip(original, expected_coefficients, strict=True):
        if not isinstance(member, dict) or not math.isclose(
            float(member.get("coef", math.nan)), coefficient, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("source member coefficients differ from the fixed TL3 blend")
    updated = json.loads(json.dumps(original))
    updated[0]["name"] = run_reference
    for old, new in zip(original, updated, strict=True):
        candidate = dict(new)
        if old["weights"] == SOURCE_MEMBER:
            candidate["name"] = old["name"]
        if candidate != old:
            raise ValueError("only the student name may change in members.json")
    return (json.dumps(updated, indent=2) + "\n").encode(), updated


def _write_export(path: Path, state: dict[str, np.ndarray]) -> dict[str, object]:
    with path.open("xb") as stream:
        np.savez(stream, **state)
    tensor_hashes: dict[str, str] = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(state):
            raise ValueError("serialized NPZ tensor keys differ from the checkpoint")
        for name in sorted(archive.files):
            value = archive[name]
            if value.dtype != np.float32 or not np.array_equal(value, state[name]):
                raise ValueError(f"serialized NPZ tensor differs from FP32 checkpoint: {name}")
            tensor_hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "tensor_sha256": tensor_hashes,
    }


def _build_candidate_zip(
    source_archive: Path,
    destination: Path,
    entries: dict[str, dict[str, object]],
    export_path: Path,
    members_json: bytes,
) -> dict[str, dict[str, object]]:
    with (
        source_archive.open("rb") as source_stream,
        destination.open("xb") as output_stream,
        ZipFile(source_stream) as source,
        ZipFile(output_stream, "w") as output,
    ):
        for info in source.infolist():
            if info.filename == SOURCE_MEMBER:
                data = export_path.read_bytes()
            elif info.filename == "members.json":
                data = members_json
            else:
                data = source.read(info)
                record = entries[info.filename]
                if (
                    len(data) != record["bytes"]
                    or hashlib.sha256(data).hexdigest() != record["sha256"]
                ):
                    raise ValueError(f"unchanged source member identity failed: {info.filename}")
            output.writestr(info, data)

    if destination.stat().st_size >= MAX_ZIP_BYTES:
        raise ValueError("candidate ZIP must be smaller than 20,000,000 bytes")
    output_entries: dict[str, dict[str, object]] = {}
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("candidate ZIP failed its CRC check")
        infos = archive.infolist()
        if len(infos) != len(entries) or set(archive.namelist()) != set(entries):
            raise ValueError("candidate ZIP entry names differ from the pinned source package")
        for info in infos:
            digest = hashlib.sha256()
            size = 0
            with archive.open(info) as member:
                for block in iter(lambda: member.read(1024 * 1024), b""):
                    size += len(block)
                    digest.update(block)
            output_entries[info.filename] = {
                "bytes": size,
                "sha256": digest.hexdigest(),
                "crc32": f"{info.CRC:08x}",
            }
            if info.filename not in (SOURCE_MEMBER, "members.json") and (
                size != entries[info.filename]["bytes"]
                or digest.hexdigest() != entries[info.filename]["sha256"]
            ):
                raise ValueError(f"candidate changed a protected source member: {info.filename}")
    return output_entries


def build_package(
    run_reference: str, runs_dir: Path, requested_output_dir: Path, artifact_role: str
) -> dict[str, object]:
    """Build and hash a TL3 archive with one H512 student member replacement."""
    if artifact_role not in {"production_candidate", "engineering_fixture"}:
        raise ValueError(f"unsupported artifact role: {artifact_role}")
    checkpoint = _assert_fp32_checkpoint(run_reference, runs_dir)
    state, source_record = load_checkpoint_member(run_reference, runs_dir, require_provenance=True)
    input_layout = _validate_member(state, source_record)
    checkpoint_sha256 = str(source_record["checkpoint_sha256"])
    provenance = source_record.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("sha256"):
        raise ValueError("checkpoint provenance sidecar identity is required")
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint changed while it was being exported")

    source_sha256, source_receipt_sha256, entries = verify_source_archive(SOURCE_ARCHIVE)
    output_dir = resolve_create_only_output_dir(requested_output_dir)
    output_dir.mkdir(exist_ok=False)
    extracted = extract_verified_entries(SOURCE_ARCHIVE, output_dir, entries)
    source_export = output_dir / "h512_weights_s5.npz"
    export_record = _write_export(source_export, state)
    with ZipFile(SOURCE_ARCHIVE) as source:
        original_members = source.read("members.json")
        if hashlib.sha256(original_members).hexdigest() != entries["members.json"]["sha256"]:
            raise ValueError("source member map changed after archive verification")
        member_map_bytes, updated_members = _replace_student_name(original_members, run_reference)
    candidate_zip = output_dir / "submission.zip"
    output_entries = _build_candidate_zip(
        SOURCE_ARCHIVE, candidate_zip, entries, source_export, member_map_bytes
    )
    if output_entries[SOURCE_MEMBER]["sha256"] != export_record["sha256"]:
        raise ValueError("candidate student entry differs from the verified NPZ export")
    if output_entries["members.json"]["sha256"] != hashlib.sha256(member_map_bytes).hexdigest():
        raise ValueError("candidate member map differs from its validated update")
    if (
        file_sha256(checkpoint) != checkpoint_sha256
        or file_sha256(Path(str(provenance["path"]))) != provenance["sha256"]
    ):
        raise ValueError("checkpoint or provenance sidecar changed during package creation")

    code_paths = {
        "packager": Path(__file__).resolve(),
        "checkpoint_exporter": ROOT / "scripts/package_submission.py",
        "source_archive_helper": ROOT / "scripts/tl3_source_archive.py",
        "architecture_helper": ROOT / "scripts/team_member_export_lib.py",
        "native_input_contract": ROOT / "src/wnn/native_representation.py",
    }
    receipt: dict[str, object] = {
        "schema": "wnn.tl3-h512-member-package.v1",
        "status": "archive_built; independent_runtime_and_quality_gates_pending",
        "artifact_role": artifact_role,
        "source_archive": {
            "path": str(SOURCE_ARCHIVE),
            "sha256": source_sha256,
            "receipt_sha256": source_receipt_sha256,
            "entries": extracted,
        },
        "new_member": {
            "run_reference": run_reference,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "provenance_path": provenance["path"],
            "provenance_sha256": provenance["sha256"],
            "input_layout": input_layout,
            "export": export_record,
        },
        "member_map": {
            "updated_student_name": run_reference,
            "members": updated_members,
        },
        "artifact": {
            "path": str(candidate_zip.resolve()),
            "bytes": candidate_zip.stat().st_size,
            "sha256": file_sha256(candidate_zip),
            "zip_entries": output_entries,
        },
        "code_sha256": {name: file_sha256(path) for name, path in code_paths.items()},
        "validation": {
            "source_zip_and_every_entry_verified": True,
            "student_architecture": "canonical 211-input H512x1 GRU, two outputs",
            "export_fp32_and_finite": True,
            "native_import_or_prediction_run": False,
            "submission_quality_or_runtime_claim": False,
        },
    }
    receipt_path = output_dir / "BUILD_RECEIPT.json"
    write_json_create_only(receipt_path, receipt)
    return receipt


def main() -> None:
    """Create a new candidate directory and report its immutable receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-reference", required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-role", choices=("production_candidate", "engineering_fixture"), required=True
    )
    args = parser.parse_args()
    try:
        receipt = build_package(
            args.run_reference,
            args.runs_dir.resolve(strict=True),
            args.output_dir,
            args.artifact_role,
        )
    except (BadZipFile, FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
