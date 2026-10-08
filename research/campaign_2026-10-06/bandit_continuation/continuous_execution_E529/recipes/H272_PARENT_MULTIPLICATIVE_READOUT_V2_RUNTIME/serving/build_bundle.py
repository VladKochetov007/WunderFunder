"""Build a local, hash-receipted native ZIP from an admitted fixed parent/head pair.

The builder has no upload path. It accepts only a recipe-bound parent checkpoint,
the matching fixed-epoch-1 head export, and a locally built native extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[5]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import train_parent_readout as runner  # noqa: E402
import train_rnn as trainer  # noqa: E402
from wnn.conditional_readout import ConditionalReadoutHeads, freeze_parent  # noqa: E402

SOLUTION_SOURCE = PACKAGE_ROOT / "serving/solution_h272.py"
EXPECTED_READOUT_SHAPES = {
    "hidden_projection.weight": (32, 512),
    "context_projection.weight": (32, 99),
    "fusion.weight": (32, 96),
    "fusion.bias": (32,),
    "output.weight": (2, 32),
    "output.bias": (2,),
}
MAX_ZIP_BYTES = 20_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_from_tensor(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().contiguous().numpy()
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("native FP32 parent weights must be finite")
    return np.ascontiguousarray(array)


def _parent_native_arrays(
    checkpoint: dict[str, Any], expected_outputs: int
) -> dict[str, np.ndarray]:
    """Convert the exact PyTorch checkpoint state to FastRnn's NPZ key contract."""
    if expected_outputs not in (23, 31) or checkpoint.get("n_in") != 211:
        raise ValueError("native H272 parent requires 211 inputs and a bound 23/31 output count")
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise ValueError("parent checkpoint is missing its model state dictionary")
    output_weight = model.get("head.2.weight")
    if output_weight is None:
        raise ValueError("parent checkpoint is missing head.2.weight")
    actual_outputs = int(output_weight.shape[0])
    runner._validate_parent_output_count(actual_outputs, expected_outputs)
    if tuple(model["mean"].shape) != (211,) or tuple(model["std"].shape) != (211,):
        raise ValueError("native parent normalization must have exactly 211 columns")
    if tuple(model["inp.weight"].shape) != (256, 211):
        raise ValueError("native parent input projection differs from GRU256x2")
    if tuple(model["head.0.weight"].shape) != (256, 512):
        raise ValueError("native parent readout must consume the 512-wide concatenation")
    if tuple(output_weight.shape) != (expected_outputs, 256):
        raise ValueError("native parent output tensor differs from the bound 23/31 contract")

    required = {
        "mean",
        "std",
        "inp.weight",
        "inp.bias",
        "head.0.weight",
        "head.0.bias",
        "head.2.weight",
        "head.2.bias",
    }
    for layer in range(2):
        required.update(
            {
                f"gru.weight_ih_l{layer}",
                f"gru.weight_hh_l{layer}",
                f"gru.bias_ih_l{layer}",
                f"gru.bias_hh_l{layer}",
            }
        )
    missing = required - model.keys()
    if missing:
        raise ValueError(f"parent model state is missing native weights: {sorted(missing)}")
    unexpected_gru = {key for key in model if key.startswith("gru.") and key not in required}
    if unexpected_gru:
        raise ValueError(
            f"parent model has unsupported recurrent weights: {sorted(unexpected_gru)}"
        )

    arrays = {name: _array_from_tensor(model[name]) for name in sorted(required)}
    expected_shapes = {
        "mean": (211,),
        "std": (211,),
        "inp.weight": (256, 211),
        "inp.bias": (256,),
        "head.0.weight": (256, 512),
        "head.0.bias": (256,),
        "head.2.weight": (expected_outputs, 256),
        "head.2.bias": (expected_outputs,),
    }
    for layer in range(2):
        expected_shapes.update(
            {
                f"gru.weight_ih_l{layer}": (768, 256),
                f"gru.weight_hh_l{layer}": (768, 256),
                f"gru.bias_ih_l{layer}": (768,),
                f"gru.bias_hh_l{layer}": (768,),
            }
        )
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"native parent tensor {name} shape {arrays[name].shape} != {shape}")
    return arrays


def _readout_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(EXPECTED_READOUT_SHAPES):
            raise ValueError("readout archive tensor-name set is not the exact native contract")
        arrays = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in EXPECTED_READOUT_SHAPES
        }
    for name, shape in EXPECTED_READOUT_SHAPES.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"readout tensor {name} is nonfinite or has the wrong shape")
    return arrays


def _validate_head_export(
    data_root: Path, recipe_path: Path, recipe: dict[str, Any], head_dir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    expected_dir = data_root / recipe["paths"]["serving_head_export"]
    if head_dir.resolve() != expected_dir.resolve():
        raise ValueError(f"head export path is source-bound to {expected_dir}")
    receipt_path = head_dir / "head_export_receipt.json"
    weights_path = head_dir / "multiplicative_head.npz"
    receipt = json.loads(receipt_path.read_text())
    expected_receipt = {
        "status": "head_weights_exported_no_submission_zip",
        "recipe_id": recipe["recipe_id"],
        "selected_epoch": runner.CHECKPOINT_EPOCH,
        "recipe_sha256": runner.file_sha256(recipe_path),
        "source_lock_sha256": runner.file_sha256(runner.SOURCE_LOCK),
        "parent_checkpoint_sha256": recipe["parent"]["checkpoint_sha256"],
        "parent_source_identity_sha256": recipe["parent"]["source_identity_sha256"],
        "mode": "multiplicative",
        "weight_file": weights_path.name,
        "precision": "float32",
        "public_zip_created": False,
    }
    if not isinstance(receipt.get("authorization_sha256"), str):
        raise ValueError("head export receipt does not bind its root authorization")
    for name, wanted in expected_receipt.items():
        if receipt.get(name) != wanted:
            raise ValueError(f"head export receipt field {name} is not bound to this recipe")
    if _sha256(weights_path) != receipt.get("weight_file_sha256"):
        raise ValueError("head export weight archive differs from its receipt hash")
    if receipt.get("weight_shapes") != {
        name: list(shape) for name, shape in EXPECTED_READOUT_SHAPES.items()
    }:
        raise ValueError("head export receipt shape map differs from the native contract")
    return receipt_path, weights_path, receipt


def _verify_selected_checkpoint(
    data_root: Path,
    recipe_path: Path,
    recipe: dict[str, Any],
    receipt: dict[str, Any],
    parent_checkpoint: dict[str, Any],
) -> None:
    parent = trainer.make_model(
        parent_checkpoint["args"],
        parent_checkpoint["n_in"],
        parent_checkpoint["model"]["head.2.weight"].shape[0],
    )
    parent.load_state_dict(parent_checkpoint["model"], strict=True)
    freeze_parent(parent)
    heads = ConditionalReadoutHeads(512, runner.RANK)
    selected_sha = receipt.get("selected_checkpoint_sha256")
    if not isinstance(selected_sha, str):
        raise ValueError("head receipt does not identify its fixed epoch-1 checkpoint")
    selected_path = runner._load_fixed_heads(
        recipe_path,
        data_root,
        torch.device("cpu"),
        parent,
        heads,
        selected_sha,
    )
    if _sha256(selected_path) != selected_sha:
        raise ValueError("selected readout checkpoint bytes changed after receipt creation")
    if recipe["training"]["selected_checkpoint_epoch"] != runner.CHECKPOINT_EPOCH:
        raise ValueError("recipe no longer selects the fixed epoch-1 readout")
    expected = {
        name: _array_from_tensor(value)
        for name, value in heads.controls["multiplicative"].state_dict().items()
    }
    observed = _readout_arrays(
        data_root / recipe["paths"]["serving_head_export"] / "multiplicative_head.npz"
    )
    if any(not np.array_equal(expected[name], observed[name]) for name in EXPECTED_READOUT_SHAPES):
        raise ValueError("exported multiplicative tensors differ from the selected checkpoint")


def _write_zip(package_dir: Path, output: Path) -> None:
    entries = sorted(path for path in package_dir.rglob("*") if path.is_file())
    if not entries:
        raise ValueError("refusing to create an empty serving archive")
    temporary = output.with_name(output.name + ".partial")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in entries:
            member = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    os.replace(temporary, output)


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    recipe_path = args.recipe.resolve()
    recipe = runner._load_recipe(recipe_path)
    data_root = args.data_root.resolve()
    trainer.ROOT = data_root
    parent_record = recipe["parent"]
    parent_path = data_root / parent_record["checkpoint"]
    if _sha256(parent_path) != parent_record["checkpoint_sha256"]:
        raise ValueError("parent checkpoint bytes differ from the frozen recipe")
    parent_checkpoint = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent_checkpoint.get("epoch") != 3:
        raise ValueError("native bundle parent must be the exact fixed epoch-3 checkpoint")
    parent_arrays = _parent_native_arrays(parent_checkpoint, parent_record["expected_n_out"])

    head_receipt_path, head_weights_path, head_receipt = _validate_head_export(
        data_root, recipe_path, recipe, args.head_dir.resolve()
    )
    _readout_arrays(head_weights_path)
    _verify_selected_checkpoint(data_root, recipe_path, recipe, head_receipt, parent_checkpoint)

    native_extension = args.native_extension.resolve()
    if not native_extension.is_file() or native_extension.stat().st_size == 0:
        raise FileNotFoundError(f"built native extension is missing: {native_extension}")
    output = args.output.resolve()
    receipt_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite native bundle or receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="h272-native-bundle-") as temporary:
        package_dir = Path(temporary)
        np.savez(package_dir / "parent_model.npz", **parent_arrays)
        shutil.copyfile(head_weights_path, package_dir / "multiplicative_head.npz")
        shutil.copyfile(SOLUTION_SOURCE, package_dir / "solution.py")
        native_dir = package_dir / "native"
        native_dir.mkdir()
        shutil.copyfile(native_extension, native_dir / "wnn_native.so")
        artifact_paths = {
            "parent_model.npz": package_dir / "parent_model.npz",
            "multiplicative_head.npz": package_dir / "multiplicative_head.npz",
            "solution.py": package_dir / "solution.py",
            "native/wnn_native.so": native_dir / "wnn_native.so",
        }
        manifest = {
            "schema": "h272-native-serving-bundle-v1",
            "recipe_id": recipe["recipe_id"],
            "recipe_sha256": runner.file_sha256(recipe_path),
            "source_lock_sha256": runner.file_sha256(runner.SOURCE_LOCK),
            "parent_run": parent_record["run"],
            "parent_checkpoint_sha256": parent_record["checkpoint_sha256"],
            "parent_source_identity_sha256": parent_record["source_identity_sha256"],
            "parent_output_count": parent_record["expected_n_out"],
            "selected_readout_checkpoint_sha256": head_receipt["selected_checkpoint_sha256"],
            "head_export_receipt_sha256": _sha256(head_receipt_path),
            "mode": "multiplicative",
            "precision": "float32",
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "public_upload_performed": False,
            "member_sha256": {name: _sha256(path) for name, path in artifact_paths.items()},
        }
        (package_dir / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        _write_zip(package_dir, output)

    size = output.stat().st_size
    if size > MAX_ZIP_BYTES:
        output.unlink()
        raise ValueError(f"serving ZIP is {size} bytes, above the {MAX_ZIP_BYTES}-byte envelope")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            output.unlink()
            raise ValueError("serving ZIP failed its CRC check")
        names = set(archive.namelist())
        expected_names = {
            "parent_model.npz",
            "multiplicative_head.npz",
            "solution.py",
            "native/wnn_native.so",
            "bundle_manifest.json",
        }
        if names != expected_names:
            output.unlink()
            raise ValueError(f"serving ZIP member set changed: {sorted(names)}")
        embedded = json.loads(archive.read("bundle_manifest.json"))
        for name, digest in embedded["member_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                output.unlink()
                raise ValueError(f"serving ZIP member hash mismatch: {name}")

    receipt = {
        "status": "native_bundle_built_locally_no_upload",
        "zip_path": str(output),
        "zip_sha256": _sha256(output),
        "zip_bytes": size,
        "within_20mb_envelope": True,
        "recipe_id": recipe["recipe_id"],
        "parent_checkpoint_sha256": parent_record["checkpoint_sha256"],
        "selected_readout_checkpoint_sha256": head_receipt["selected_checkpoint_sha256"],
        "native_extension_sha256": _sha256(native_extension),
        "member_sha256": embedded["member_sha256"],
        "public_upload_performed": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--native-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(build_bundle(_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
