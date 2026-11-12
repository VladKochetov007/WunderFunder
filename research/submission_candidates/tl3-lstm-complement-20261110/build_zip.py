"""Export pinned tensor data and assemble a CPU candidate; never execute a model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE = "wnn_cpp.cpython-311-x86_64-linux-gnu.so"


def sha256(path: Path) -> str:
    """Hash bytes without deserializing executable objects."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Create a JSON artifact; preserve earlier attempts."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_fitting(checkpoint: dict, name: str, split_sha: str) -> set[tuple[str, int]]:
    """Verify the supplied-data fit boundary from the hash-pinned checkpoint."""
    if name == "lstm":
        data = checkpoint["data"]
        observed_split = data["split_sha256"]
        fit = {tuple(member) for member in data["fit_members"]}
        evaluation = {tuple(member) for member in data["evaluation_members"]}
        config = checkpoint["config"]
        if (
            config["model"]["architecture"] != "lstm"
            or config["model"]["input_feature_mode"] != "raw112"
        ):
            raise ValueError("LSTM model configuration differs")
        if checkpoint["completed_epochs"] != 8 or config["seed"] != 1302:
            raise ValueError("LSTM endpoint differs")
    else:
        provenance = checkpoint["provenance"]
        observed_split = provenance["split_manifest_sha256"]
        fit = {
            (source, row) for source, rows in provenance["fit_row_groups"].items() for row in rows
        }
        args = checkpoint["args"]
        # These pinned PR49 checkpoints encode official-valid groups as a list;
        # `vfold` is their legacy evaluator mode, not a mixed train/valid set.
        if args["eval_source"] != "vfold":
            raise ValueError("legacy evaluation source differs")
        evaluation = {("valid", row) for row in provenance["evaluation_row_groups"]}
        if args["cell"] != "gru" or args["enc"] != ("linear" if name == "w18" else "tiedlead"):
            raise ValueError("legacy model configuration differs")
        if checkpoint.get("readout") is not None or checkpoint.get("readout_tilted"):
            raise ValueError("unsupported transformed target readout")
    if observed_split != split_sha or len(fit) != 10325 or {s for s, _ in fit} != {"train"}:
        raise ValueError("checkpoint fitting boundary differs")
    if evaluation != {("valid", row) for row in range(1873)} or fit & evaluation:
        raise ValueError("checkpoint evaluation boundary differs")
    return fit


def export_tensor_data(root: Path, stage: Path, records: list[dict], split_sha: str) -> list[dict]:
    """Restricted data-only Torch load; no model construction, forward, or inference.

    Original checkpoint bytes are pinned before restricted deserialization. Only
    dense FP32 arrays are exported. Auxiliary outputs are removed, never averaged.
    """
    import numpy as np
    import torch
    from torch.torch_version import TorchVersion

    exported = []
    common_fit = None
    for record in records:
        path = root / record["file"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"checkpoint differs: {path.name}")
        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        fit = verify_fitting(checkpoint, record["name"], split_sha)
        if common_fit is not None and common_fit != fit:
            raise ValueError("members use different training groups")
        common_fit = fit
        state = checkpoint["model"]
        if checkpoint["epoch"] != record["epoch"] or set(state) != set(record["tensor_shapes"]):
            raise ValueError("checkpoint epoch or tensor keys differ")
        arrays, inventory = {}, {}
        for key, shape in record["tensor_shapes"].items():
            tensor = state[key]
            if tensor.dtype != torch.float32 or list(tensor.shape) != shape:
                raise ValueError(f"unexpected tensor: {record['name']}/{key}")
            array = np.ascontiguousarray(tensor.detach().cpu().numpy())
            if not np.isfinite(array).all():
                raise ValueError("nonfinite weights")
            if key == "std" and np.any(array <= 0):
                raise ValueError("nonpositive normalization scale")
            if key in {"head.2.weight", "head.2.bias"}:
                array = np.ascontiguousarray(array[:2])
            arrays[key] = array
            inventory[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        destination = stage / f"{record['name']}.npz"
        with destination.open("xb") as stream:
            np.savez(stream, **arrays)
        metadata = {key: value for key, value in checkpoint.items() if key != "model"}
        write_json(stage.parent / f"{record['name']}.checkpoint-metadata.json", metadata)
        exported.append(
            {
                "name": record["name"],
                "checkpoint_sha256": record["sha256"],
                "export_sha256": sha256(destination),
                "tensors": inventory,
            }
        )
    return exported


def build(inputs_dir: Path, native_archive: Path, output_dir: Path) -> None:
    """Produce a deterministic flat-root ZIP and a byte-bound preparation receipt."""
    inputs = json.loads((HERE / "INPUTS.json").read_text())
    recipe = json.loads((HERE / "recipe.json").read_text())
    for archive in inputs["archives"]:
        if sha256(inputs_dir / archive["file"]) != archive["sha256"]:
            raise ValueError("source release archive differs")
    producer = inputs["producer_manifest"]
    if sha256(HERE / producer["file"]) != producer["sha256"]:
        raise ValueError("producer manifest differs")
    manifest = json.loads((HERE / producer["file"]).read_text())
    lstm_record = next(
        m for m in manifest["models"] if m["fit_id"] == "primary-refit-03-s1t000000-lstm"
    )
    if lstm_record["source_sha256"] != producer["lstm_training_source_sha256"]:
        raise ValueError("LSTM source identity differs")
    if sha256(HERE / "PR53_FIXED_MIXTURES_REPORT.json") != inputs["base_source_report_sha256"]:
        raise ValueError("fixed-mixture source report differs")
    output_dir.mkdir(parents=False, exist_ok=False)
    stage = output_dir / "package"
    stage.mkdir()
    for name in ("solution.py", "lstm_cpu.py"):
        ast.parse((HERE / name).read_text(), feature_version=(3, 11))
        shutil.copyfile(HERE / name, stage / name)
    for dependency in inputs["bundled_dependency"]["files"]:
        path = inputs_dir / dependency["file"]
        if sha256(path) != dependency["sha256"] or path.stat().st_size != dependency["bytes"]:
            raise ValueError("bundled dependency differs")
        shutil.copyfile(path, stage / path.name)
    exported = export_tensor_data(inputs_dir, stage, inputs["checkpoints"], inputs["split_sha256"])
    native_contract = inputs["native_runtime"]
    if sha256(native_archive) != native_contract["sha256"]:
        raise ValueError("native source archive differs")
    with zipfile.ZipFile(native_archive) as archive:
        for level, digest in native_contract["libraries"].items():
            name = f"{level}/{NATIVE}"
            if archive.namelist().count(name) != 1 or archive.getinfo(name).file_size > 1_000_000:
                raise ValueError("invalid native member")
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != digest or data[:4] != b"\x7fELF":
                raise ValueError("native library differs")
            destination = stage / name
            destination.parent.mkdir()
            destination.write_bytes(data)
    write_json(
        stage / "members.json",
        {
            "schema": "tl3c-native-members-v1",
            "mode": recipe["mode"],
            "members": recipe["baseline_members"],
        },
    )
    shutil.copyfile(HERE / "recipe.json", stage / "recipe.json")
    destination = output_dir / "kotiki-tl3-w18-lstm10-cpu-20261110.zip"
    with zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(stage).as_posix(), (2026, 11, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    if destination.stat().st_size >= 20_000_000:
        raise ValueError("candidate does not fit strict <20MB ZIP limit")
    with zipfile.ZipFile(destination) as archive:
        for name in archive.namelist():
            # Reading verifies CRC only; no submitted Python or native code runs.
            archive.read(name)
    write_json(
        output_dir / "BUILD_RECEIPT.json",
        {
            "schema": "wnn.cpu-submit-build.v1",
            "status": "PACKAGED_NOT_EXECUTED",
            "zip": {
                "file": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            },
            "inputs_sha256": sha256(HERE / "INPUTS.json"),
            "recipe_sha256": sha256(HERE / "recipe.json"),
            "source_sha256": {
                name: sha256(HERE / name) for name in ("build_zip.py", "solution.py", "lstm_cpu.py")
            },
            "members": exported,
            "local_operations": [
                "pinned tensor deserialization",
                "tensor serialization",
                "ZIP byte/CRC audit",
                "source AST parsing",
            ],
            "tests_run": False,
            "inference_run": False,
            "training_run": False,
            "native_library_loaded": False,
            "submission_ready": False,
            "quality_score": None,
            "server_acceptance": "PENDING",
            "uploaded": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", required=True, type=Path)
    parser.add_argument("--native-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    build(args.inputs_dir, args.native_archive, args.output_dir)
