"""Package verified best checkpoints and fixed TL3 endpoints without changing fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import torch

try:
    from scripts.checkpoint_export_contract import validate_inventory, verify_selection
except ModuleNotFoundError:
    from checkpoint_export_contract import validate_inventory, verify_selection

SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})")
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


def sha256(path: Path) -> str:
    """Hash a file in bounded chunks."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def json_default(value: object) -> object:
    """Keep configuration JSON portable while leaving tensors in the checkpoint."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "in_checkpoint": True}
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def encoded(value: object) -> bytes:
    """Serialize metadata, refusing apparent embedded credential values."""
    text = json.dumps(value, indent=2, sort_keys=True, default=json_default, allow_nan=False)
    if SECRET.search(text):
        raise ValueError("Metadata contains an apparent credential; publication stopped")
    return (text + "\n").encode("utf-8")


def verified_file(path: Path, expected: str, run: Path) -> None:
    """Require an original regular checkpoint inside the exact run directory."""
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(run.resolve()):
        raise ValueError(f"Unsafe or missing checkpoint: {path.name}")
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f"Checkpoint hash mismatch: {path.name}")


def checkpoint_metadata(path: Path, epoch: int) -> dict[str, object]:
    """CPU-load an already hash-verified owned checkpoint and check its tensor payload."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != epoch - 1:
        raise ValueError(f"Checkpoint epoch differs from its selection: {path.name}")
    model = checkpoint.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError("Checkpoint contains no model state")
    tensors = [value for value in model.values() if isinstance(value, torch.Tensor)]
    if not tensors or any(not torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("Checkpoint model state is empty or nonfinite")
    return {
        "checkpoint_schema": checkpoint.get("checkpoint_schema", "legacy-rnn"),
        "source_epoch_index": checkpoint["epoch"],
        "epoch_one_based": epoch,
        "state_tensor_count": len(tensors),
        "state_tensor_elements": sum(value.numel() for value in tensors),
        "configuration": checkpoint.get("args", checkpoint.get("config")),
        "input_width": checkpoint.get("n_in"),
    }


def add_file(archive: zipfile.ZipFile, source: Path, destination: str) -> dict[str, object]:
    """Append an immutable payload and record its portable entry identity."""
    archive.write(source, destination)
    return {"path": destination, "bytes": source.stat().st_size, "sha256": sha256(source)}


def package_run(
    archive: zipfile.ZipFile, record: dict[str, object], run_root: Path
) -> dict[str, object]:
    """Keep best-observed and frozen scientific roles separate, sharing identical paths."""
    run_name = str(record["run"])
    selected = Path(str(record["best_path"]))
    run = run_root / run_name
    sidecars = (
        ["history.json", "provenance.json"]
        if int(record["pr"]) == 49
        else [
            "checkpoint_pointers.json",
            "epoch_index.json",
            "run_metadata.json",
            "metrics.jsonl",
            "monitor_evaluations.jsonl",
        ]
    )
    captured = {}
    for name in sidecars:
        path = run / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20:
            raise ValueError(f"Missing or unsafe metadata: {name}")
        captured[name] = path.read_bytes()
        if SECRET.search(captured[name].decode("utf-8")):
            raise ValueError("Apparent credential in metadata; publication stopped")
    verify_selection(record, captured)
    best_epoch = int(record["best_epoch"])
    verified_file(selected, str(record["expected_sha256"]), run)
    metadata = checkpoint_metadata(selected, best_epoch)
    files: list[dict[str, object]] = []
    roles = {"best_observed": {"epoch": best_epoch, "valid_wp": record["best_wp"]}}
    destinations: dict[Path, str] = {}

    def checkpoint(path: Path, expected: str) -> str:
        verified_file(path, expected, run)
        if path not in destinations:
            destination = f"{run_name}/checkpoints/{path.name}"
            files.append(add_file(archive, path, destination))
            destinations[path] = destination
        return destinations[path]

    roles["best_observed"]["path"] = checkpoint(selected, str(record["expected_sha256"]))
    if int(record["pr"]) == 49:
        fixed = Path(str(record["fixed_path"]))
        fixed_epoch = int(record["fixed_epoch"])
        verified_file(fixed, str(record["fixed_expected_sha256"]), run)
        checkpoint_metadata(fixed, fixed_epoch)
        roles["fixed_endpoint"] = {
            "epoch": fixed_epoch,
            "valid_wp": record["fixed_wp"],
            "path": checkpoint(fixed, str(record["fixed_expected_sha256"])),
        }
        runtime = Path(str(record["selected_path"]))
        verified_file(runtime, str(record["selected_expected_sha256"]), run)
        checkpoint_metadata(runtime, fixed_epoch)
        roles["selected_runtime"] = {
            "epoch": fixed_epoch,
            "valid_wp": record["fixed_wp"],
            "path": checkpoint(runtime, str(record["selected_expected_sha256"])),
        }
    for name, data in captured.items():
        destination = f"{run_name}/metadata/{name}"
        archive.writestr(destination, data)
        files.append(
            {"path": destination, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    metadata_path = f"{run_name}/metadata/checkpoint_configuration.json"
    archive.writestr(metadata_path, encoded(metadata))
    return {
        "run": run_name,
        "status_at_inventory": record["status"],
        "completed_epochs_at_inventory": record["epochs"],
        "roles": roles,
        "files": files,
        "configuration_path": metadata_path,
        "cpu_checkpoint_load_verified": True,
        "model_tensors_finite": True,
    }


def build(inventory: Path, split: Path, output: Path) -> dict[str, object]:
    """Create exactly two archives; never train, overwrite a prior archive, or delete inputs."""
    if output.exists():
        raise FileExistsError("Preserve prior export; use a new output directory")
    if sha256(split) != SPLIT_SHA:
        raise ValueError("Primary split identity changed")
    source = json.loads(inventory.read_text())
    rows = source["records"]
    present = validate_inventory(rows)
    roots = {
        46: inventory.parent.parents[2] / "runs/quality8-full-20261028/runs",
        49: inventory.parent / "artifacts/runs",
    }
    output.mkdir(parents=True, mode=0o700)
    result = {"created_utc": datetime.now(UTC).isoformat(), "archives": []}
    for pr, expected in ((46, 8), (49, 16)):
        members = [row for row in present if row["pr"] == pr]
        if len(members) != expected:
            raise ValueError("PR checkpoint count differs from the recorded panel")
        name = f"tabdata-pr{pr}-checkpoints-20261029-r1.zip"
        path = output / name
        manifest = {
            "schema": "wnn.experiment-checkpoints.v1",
            "pr": pr,
            "inventory_asof_utc": source["asof_utc"],
            "split_sha256": SPLIT_SHA,
            "fit_train_sequences": 10325,
            "fit_valid_sequences": 0,
            "evaluation_valid_sequences": 1873,
            "official_scored_rows": 3528273,
            "selection_note": "best_observed uses exposed valid; fixed_endpoint preserves the TL3 recipe",
            "absent_runs": [
                row["run"] for row in rows if row["pr"] == pr and not row.get("best_exists")
            ],
            "runs": [],
        }
        with zipfile.ZipFile(
            path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.write(split, "shared/chronological_primary.json")
            manifest["runs"] = [package_run(archive, row, roots[pr]) for row in members]
            archive.writestr("MANIFEST.json", encoded(manifest))
            archive.writestr(
                "README.txt",
                "Original unquantized checkpoints with embedded model state and configuration.\n"
                "MANIFEST.json maps each run, score, epoch, role and SHA256.\n"
                "best_observed is exposed valid selection; fixed_endpoint is the frozen TL3 endpoint.\n"
                "selected_runtime preserves original model.pt hashes referenced by prediction exports.\n"
                "Epochs in roles are one-based; original checkpoint epoch fields are zero-based.\n"
                "Full original metadata and the official primary split are included.\n"
                "Optimizer/resume files and dataset/prediction-array payloads are not included.\n",
            )
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC verification failed")
        manifest_path = output / name.replace(".zip", ".manifest.json")
        manifest_path.write_bytes(encoded(manifest))
        checksums = output / name.replace(".zip", ".sha256.txt")
        checksums.write_text(
            f"{sha256(path)}  {path.name}\n{sha256(manifest_path)}  {manifest_path.name}\n",
            encoding="ascii",
        )
        result["archives"].append(
            {
                "pr": pr,
                "zip": str(path),
                "zip_sha256": sha256(path),
                "bytes": path.stat().st_size,
                "manifest": str(manifest_path),
                "checksums": str(checksums),
                "model_count": expected,
            }
        )
    (output / "EXPORT_RECEIPT.json").write_bytes(encoded(result))
    return result


def main() -> None:
    """Run the bounded checkpoint publication preparation on the receiving node."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(build(args.inventory, args.split, args.output)))


if __name__ == "__main__":
    main()
