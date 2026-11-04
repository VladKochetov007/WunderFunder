"""Archive exact fixed/best checkpoint evidence and separate replay code, stdlib only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heterogeneous_checkpoint_archive_contract import (  # noqa: E402
    NEW_RUNS,
    SPLIT_SHA256,
    W18,
    FileEntry,
    JsonObject,
    capture,
    encoded,
    fit_receipt,
    replay_files,
    sha256,
    validate_run,
)


def write_zip(
    path: Path, entries: list[FileEntry], manifest: JsonObject, readme: str
) -> JsonObject:
    """Stream unchanged bytes into a create-only ZIP, then verify every archived hash."""
    if path.exists() or len({entry.destination for entry in entries}) != len(entries):
        raise ValueError("archive exists or has duplicate entry names")
    partial = path.with_suffix(".zip.partial")
    with zipfile.ZipFile(
        partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for entry in entries:
            digest = hashlib.sha256()
            if entry.raw is not None:
                archive.writestr(entry.destination, entry.raw)
                continue
            with (
                entry.source.open("rb") as source,
                archive.open(entry.destination, "w") as destination,
            ):
                for block in iter(lambda: source.read(1 << 20), b""):
                    destination.write(block)
                    digest.update(block)
            after = entry.source.stat()
            key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if digest.hexdigest() != entry.sha256 or key != entry.stat_key:
                raise ValueError("original payload changed during archive streaming")
        archive.writestr("MANIFEST.json", encoded(manifest))
        archive.writestr("README.txt", readme.encode())
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive CRC verification failed")
        for entry in entries:
            digest = hashlib.sha256()
            with archive.open(entry.destination) as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            if (
                archive.getinfo(entry.destination).file_size != entry.size
                or digest.hexdigest() != entry.sha256
            ):
                raise ValueError("archive payload roundtrip SHA256 mismatch")
    os.replace(partial, path)
    manifest_path = path.with_suffix(".manifest.json")
    with manifest_path.open("xb") as stream:
        stream.write(encoded(manifest))
    return {
        "zip": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "payload_files": len(entries),
        "payload_bytes": sum(item.size for item in entries),
        "payload_roundtrip_verified": True,
    }


def collect_run(
    root: Path, runtime: Path, run: str, epochs: int, split: JsonObject
) -> tuple[JsonObject, list[FileEntry]]:
    """Collect inference snapshots only after the exact owned fitting command terminates."""
    owned = fit_receipt(root, run)
    directory = runtime / "artifacts/runs" / run
    if owned is None:
        return {
            "run": run,
            "state": "pending",
            "reason": "owned fit has no successful terminal receipt",
        }, []
    evidence, files = owned
    if directory.is_symlink() or not directory.resolve(strict=True).is_relative_to(
        runtime.resolve(strict=True)
    ):
        raise ValueError("owned checkpoint directory escapes its named runtime")
    environment = evidence["request"].get("env", {})
    if Path(str(environment.get("WNN_PROJECT_ROOT", ""))).resolve(strict=True) != runtime.resolve(
        strict=True
    ):
        raise ValueError("fit request belongs to another artifact runtime")
    provenance_file = capture(
        directory / "provenance.json", f"{run}/metadata/provenance.json", text=True
    )
    history_file = capture(directory / "history.json", f"{run}/metadata/history.json", text=True)
    provenance, history = json.loads(provenance_file.raw), json.loads(history_file.raw)
    if not isinstance(provenance, dict) or not isinstance(history, list):
        raise ValueError("checkpoint metadata has an unsupported shape")
    source_root = Path(evidence["source_root"])
    validate_run(run, epochs, provenance, history, split, source_root)
    files.extend([provenance_file, history_file])
    hashes = provenance["checkpoint_sha256"]
    for name in ["model.pt", *(f"model_ep{epoch}.pt" for epoch in range(epochs))]:
        expected = hashes.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"producer omits an epoch checkpoint hash: {run}/{name}")
        files.append(capture(directory / name, f"{run}/checkpoints/{name}", expected=expected))
    optional = (
        "mean.npy",
        "std.npy",
        "mean.pt",
        "std.pt",
        "config.json",
        "args.json",
        "run_metadata.json",
    )
    for name in optional:
        path = directory / name
        if path.exists():
            files.append(capture(path, f"{run}/metadata/{name}", text=name.endswith(".json")))
    best = max(history, key=lambda row: row["weighted_pearson"])
    fixed = epochs - 1
    roles = {
        "fixed_endpoint": {
            "path": f"{run}/checkpoints/model_ep{fixed}.pt",
            "epoch_zero_based": fixed,
            "valid_wp": history[fixed]["weighted_pearson"],
        },
        "selected_runtime": {
            "path": f"{run}/checkpoints/model.pt",
            "epoch_zero_based": fixed,
            "valid_wp": history[fixed]["weighted_pearson"],
        },
        "best_observed": {
            "path": f"{run}/checkpoints/model_ep{best['epoch']}.pt",
            "epoch_zero_based": best["epoch"],
            "valid_wp": best["weighted_pearson"],
            "differs_from_fixed": best["epoch"] != fixed,
        },
    }
    return {
        "run": run,
        "state": "complete",
        "configured_epochs": epochs,
        "roles": roles,
        "actual_request_source_root": str(source_root),
        "trainer_source_sha256": provenance["trainer_implementation"]["sha256"],
        "configuration": provenance["args"],
        "files": [item.manifest() for item in files],
        "normalization": "original model state retains embedded buffers; standalone files included only when present",
        "checkpoint_tensor_deserialization": "not_performed_stdlib_only",
    }, files


def build(
    root: Path,
    legacy: Path,
    split_path: Path,
    output: Path,
    galam_source: Path | None,
    require_complete: bool,
    budget: int,
) -> JsonObject:
    """Create bounded checkpoint and separate replay archives; never resume or modify fits."""
    if output.exists():
        raise FileExistsError("preserve prior exports; choose a fresh output directory")
    split_entry = capture(
        split_path, "shared/chronological_primary.json", text=True, expected=SPLIT_SHA256
    )
    split = json.loads(split_entry.raw)
    if (
        len(split["fit"]["train"]) != 10325
        or split["fit"]["valid"]
        or len(split["evaluation"]["valid"]) != 1873
    ):
        raise ValueError("checkpoint packet does not use the complete chronological boundary")
    runs, checkpoint_files = [], [split_entry]
    for run in (*NEW_RUNS, W18):
        record, files = collect_run(
            root, legacy if run == W18 else root, run, 2 if run == W18 else 4, split
        )
        runs.append(record)
        checkpoint_files.extend(files)
    completed = [run for run in runs if run["state"] == "complete"]
    if not completed or require_complete and len(completed) != 7:
        raise ValueError("required completed fitting roles are not all available")
    source_roots = [
        path for name in ("source-r1", "source-r2", "source-r3") if (path := root / name).is_dir()
    ]
    source_roots.extend(Path(run["actual_request_source_root"]) for run in completed)
    if galam_source is not None:
        source_roots.append(galam_source)
    replay, sources = replay_files(root, source_roots)
    if sum(item.size for item in checkpoint_files + replay) > budget:
        raise ValueError("uncompressed checkpoint/replay packet exceeds its declared byte budget")
    manifest = {
        "schema": "wnn.heterogeneous-checkpoint-archive.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "state": "complete" if len(completed) == 7 else "partial",
        "completed_models": len(completed),
        "expected_models": 7,
        "split_sha256": SPLIT_SHA256,
        "runs": runs,
        "selection": "fixed endpoint unchanged; best-observed is exposed-valid diagnostic; all epoch snapshots retain original names",
        "excluded": "datasets, prediction banks, optimizer/resume/training-state files; external ancestor weights remain separate",
    }
    output.mkdir(parents=True, mode=0o700)
    checkpoints = write_zip(
        output / "tabdata-heterogeneous-checkpoints.zip",
        checkpoint_files,
        manifest,
        "Original inference checkpoint bytes and producer evidence. Epochs are zero based.\nFixed and best-observed roles differ explicitly. No tensor deserialization or finite-tensor claim.\nNormalization buffers remain embedded; standalone files included only if present.\n",
    )
    source_manifest = {
        "schema": "wnn.heterogeneous-replay-sources.v1",
        "sources": sources,
        "files": [item.manifest() for item in replay],
        "external_weights_required": True,
        "scope": "source/code replay only; semantic trainer hashes are distinct from capsule inventory hashes",
    }
    source_archive = write_zip(
        output / "tabdata-heterogeneous-replay-sources.zip",
        replay,
        source_manifest,
        "Original scoped Python/dependency/tool/receipt bytes, with SHA256.\nNo datasets, prediction banks or optimizer/resume state. Replaying needs source-qualified external data and ancestor weights.\n",
    )
    receipt = {
        "schema": "wnn.heterogeneous-checkpoint-export.v1",
        "state": manifest["state"],
        "completed_models": len(completed),
        "archives": [checkpoints, source_archive],
        "new_fits": 0,
        "stdlib_only": True,
        "model_tensors_deserialized": False,
    }
    with (output / "EXPORT_RECEIPT.json").open("xb") as stream:
        stream.write(encoded(receipt))
    with (output / "SHA256SUMS.txt").open("x", encoding="ascii") as stream:
        for path in sorted(output.iterdir()):
            if path.name != "SHA256SUMS.txt":
                stream.write(f"{sha256(path)}  {path.name}\n")
    return receipt


def main() -> None:
    """Package already completed roles on the node; never upload or alter originals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--galam-source", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--max-uncompressed-mib", type=int, default=512)
    args = parser.parse_args()
    if args.max_uncompressed_mib < 1:
        parser.error("archive byte budget must be positive")
    print(
        json.dumps(
            build(
                args.root.resolve(strict=True),
                args.legacy_root.resolve(strict=True),
                args.split,
                args.output,
                args.galam_source,
                args.require_complete,
                args.max_uncompressed_mib << 20,
            )
        )
    )


if __name__ == "__main__":
    main()
