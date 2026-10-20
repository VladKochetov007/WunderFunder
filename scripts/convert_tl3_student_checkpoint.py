"""Convert the pinned TL3 plain student NPZ to a provenance-bound CPU checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import BadZipFile

from team_member_export_lib import MemberArchitecture, infer_architecture, load_fp32_model
from tl3_source_archive import (
    CHECKPOINT_NAME,
    EXTRACTED_MEMBER,
    SOURCE_ARCHIVE,
    SOURCE_MEMBER,
    SOURCE_RECEIPT,
    extract_verified_entries,
    file_sha256,
    resolve_create_only_output_dir,
    verify_source_archive,
)

if TYPE_CHECKING:
    import torch


RECEIPT_NAME = "conversion_receipt.json"
EXPECTED_ARCHITECTURE = MemberArchitecture(211, 256, 2, 256, 2)
ARCHITECTURE_ARGS: dict[str, object] = {
    "arch": "gru",
    "cell": "gru",
    "enc": "linear",
    "hidden": 256,
    "layers": 2,
    "dropout": 0.0,
    "head_layers": 1,
    "readout_activation": "gelu",
    "numeric_map": "none",
    "ple_intervals": 8,
}


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    gru_shapes = {
        f"gru.{kind}_l{layer}": shape
        for layer in range(2)
        for kind, shape in (
            ("weight_ih", (768, 256)),
            ("weight_hh", (768, 256)),
            ("bias_ih", (768,)),
            ("bias_hh", (768,)),
        )
    }
    return {
        "mean": (211,),
        "std": (211,),
        "inp.weight": (256, 211),
        "inp.bias": (256,),
        "head.0.weight": (256, 512),
        "head.0.bias": (256,),
        "head.2.weight": (2, 256),
        "head.2.bias": (2,),
        **gru_shapes,
    }


def _validated_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    import torch

    state = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    shapes = _expected_shapes()
    if set(state) != set(shapes):
        raise ValueError("loaded state keys differ from the exact TL3 plain student")
    for name, tensor in state.items():
        if tuple(tensor.shape) != shapes[name] or tensor.dtype != torch.float32:
            raise ValueError(f"loaded tensor has an unexpected shape/dtype: {name}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"loaded tensor is nonfinite: {name}")
    if not (state["std"] > 0).all().item():
        raise ValueError("input normalization std must be strictly positive")
    return state


def _state_hashes(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    hashes = {}
    for name, tensor in sorted(state.items()):
        contiguous = tensor.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(list(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def _write_json_create_only(path: Path, record: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def convert_tl3_student(archive_path: Path, requested_output_dir: Path) -> dict[str, str]:
    """Create a strict canonical checkpoint from the pinned TL3 student member.

    Args:
        archive_path: Path to the exact source ZIP whose SHA is pinned in the archive helper.
        requested_output_dir: New absolute directory under physical research scratch.

    Returns:
        Relative names and path of the created checkpoint and receipt.

    Raises:
        FileExistsError: If the create-only output directory already exists.
        OSError: If source or output files cannot be read or written.
        ValueError: If source hashes, ZIP metadata, tensors, or round-trip checks fail.
    """
    import torch

    archive_path = archive_path.resolve(strict=True)
    archive_sha256, receipt_sha256, source_entries = verify_source_archive(archive_path)
    output_dir = resolve_create_only_output_dir(requested_output_dir)
    output_dir.mkdir(exist_ok=False)
    succeeded = False
    try:
        entry_hashes = extract_verified_entries(archive_path, output_dir, source_entries)
        architecture = infer_architecture(_expected_shapes())
        if architecture != EXPECTED_ARCHITECTURE:
            raise ValueError("pinned TL3 architecture specification changed")
        weight_path = output_dir / EXTRACTED_MEMBER
        if (
            weight_path.stat().st_size != source_entries[SOURCE_MEMBER]["bytes"]
            or file_sha256(weight_path) != source_entries[SOURCE_MEMBER]["sha256"]
        ):
            raise ValueError("extracted student member differs from its pinned source hash")
        model = load_fp32_model(weight_path, architecture, "cpu")
        state = _validated_state(model)
        tensor_hashes = _state_hashes(state)
        if not torch.equal(state["mean"], model.mean.detach().cpu()) or not torch.equal(
            state["std"], model.std.detach().cpu()
        ):
            raise ValueError("normalization buffers changed during canonical loading")

        code_root = Path(__file__).resolve().parents[1]
        code_hashes = {
            "converter": file_sha256(Path(__file__).resolve()),
            "archive_helper": file_sha256(code_root / "scripts/tl3_source_archive.py"),
            "export_helper": file_sha256(code_root / "scripts/team_member_export_lib.py"),
            "canonical_model": file_sha256(code_root / "src/wnn/nn.py"),
        }
        source_asset = {
            "kind": "official_team_archive_member",
            "archive_path": str(archive_path),
            "archive_sha256": archive_sha256,
            "archive_receipt_path": str(SOURCE_RECEIPT),
            "archive_receipt_sha256": receipt_sha256,
            "member": SOURCE_MEMBER,
            "member_sha256": source_entries[SOURCE_MEMBER]["sha256"],
        }
        provenance = {
            "original_training_args": None,
            "training_membership": "unknown",
            "teacher_ancestry": "unknown",
            "initializer_ancestry": "unknown",
            "claim_scope": "production-only",
            "heldout_claim": False,
            "training_admission": False,
            "conversion_code_sha256": code_hashes,
            "source_tensor_sha256": tensor_hashes,
        }
        checkpoint = {
            "format": "wnn.tl3-student-conversion.v1",
            "model": state,
            "n_in": 211,
            "args": dict(ARCHITECTURE_ARGS),
            "source_asset": source_asset,
            "provenance": provenance,
        }
        checkpoint_path = output_dir / CHECKPOINT_NAME
        with checkpoint_path.open("xb") as stream:
            torch.save(checkpoint, stream)
        reloaded: object = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(reloaded, dict) or reloaded.get("n_in") != 211:
            raise ValueError("checkpoint round-trip lost its input-width metadata")
        if reloaded.get("args") != ARCHITECTURE_ARGS:
            raise ValueError("checkpoint round-trip changed architecture-only arguments")
        loaded_state = reloaded.get("model")
        if not isinstance(loaded_state, Mapping) or set(loaded_state) != set(state):
            raise ValueError("checkpoint round-trip changed model-state keys")
        for name, tensor in state.items():
            if not isinstance(loaded_state.get(name), torch.Tensor) or not torch.equal(
                tensor, loaded_state[name]
            ):
                raise ValueError(f"checkpoint tensor round-trip mismatch: {name}")
        model.load_state_dict(loaded_state, strict=True)
        output_tensor_hashes = _state_hashes(loaded_state)
        if output_tensor_hashes != tensor_hashes:
            raise ValueError("checkpoint tensor hashes changed after round-trip")

        receipt: dict[str, object] = {
            "schema": "wnn.tl3-student-conversion-receipt.v1",
            "status": "converted_initializer_only; training_admission=false",
            "source_archive": {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "sha256": archive_sha256,
                "receipt_sha256": receipt_sha256,
                "entries": entry_hashes,
                "extracted_source_member": {
                    "path": EXTRACTED_MEMBER,
                    "bytes": weight_path.stat().st_size,
                    "sha256": file_sha256(weight_path),
                },
            },
            "checkpoint": {
                "path": CHECKPOINT_NAME,
                "bytes": checkpoint_path.stat().st_size,
                "sha256": file_sha256(checkpoint_path),
                "tensor_sha256": output_tensor_hashes,
            },
            "code_sha256": code_hashes,
            "validation": {
                "strict_load": True,
                "cpu_fp32": True,
                "finite_tensors": True,
                "positive_std": True,
                "exact_tensor_roundtrip": True,
                "forward_test_run": False,
                "training_or_fit_run": False,
            },
        }
        _write_json_create_only(output_dir / RECEIPT_NAME, receipt)
        succeeded = True
        return {
            "output_dir": str(output_dir),
            "checkpoint": CHECKPOINT_NAME,
            "receipt": RECEIPT_NAME,
        }
    finally:
        if not succeeded:
            shutil.rmtree(output_dir)


def main() -> None:
    """Run the create-only conversion CLI and print its result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=SOURCE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = convert_tl3_student(args.archive, args.output_dir)
    except (BadZipFile, FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
