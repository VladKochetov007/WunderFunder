"""Offline-checkable bootstrap for a root-admitted remote loss-pair consumer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CACHE_MANIFEST_SHA = "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b"
CACHE_KEY = "949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6"
MEMBERSHIP_SHA = "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
REGISTRY_SHA = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
CPU_KERNEL = "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1"
GPU_KERNEL = "vladkochetov07/wnn-loss-pair-gpu-20261009-v1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
# Heterogeneous historical producer records and JSON artifact contracts.
Record = dict[str, Any]


def digest(path: Path) -> str:
    """Hash a bounded stream without loading mounted arrays into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Record:
    """Reject non-object contract artifacts at the boundary."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, record: Record) -> None:
    """Preserve each phase/admission/output identity as strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def discover_pinned(input_root: Path, filename: str, expected: str) -> Path:
    """Discover actual nested mount paths by bytes, never by assumed slug folder."""
    candidates = list(input_root.rglob(filename))
    matched = [path for path in candidates if digest(path) == expected]
    if len(matched) != 1:
        raise ValueError(
            f"{filename}: expected exactly one pinned nested artifact, found {len(matched)}"
        )
    return matched[0]


def verify_shared_contract(path: Path, admission: Record) -> tuple[Record, dict[str, Path]]:
    """Consume the exact root-pinned producer document and its real artifact triples."""
    if digest(path) != admission.get("gpu_training_contract_sha256"):
        raise ValueError("GPU producer training contract differs from root pin")
    contract = read_json(path)
    if (
        contract.get("schema") != "wnn.loss-pair.training-contract.v1"
        or contract.get("format_version") != 1
    ):
        raise ValueError("unsupported actual GPU producer contract")
    expected_cache = {
        "manifest_sha256": CACHE_MANIFEST_SHA,
        "cache_key": CACHE_KEY,
        "membership_sha256": MEMBERSHIP_SHA,
        "split_manifest_sha256": SPLIT_SHA,
        "portable_registry_sha256": REGISTRY_SHA,
        "fold": 0,
    }
    if any(contract.get("cache", {}).get(key) != value for key, value in expected_cache.items()):
        raise ValueError("GPU producer contract binds different cache/membership")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("GPU producer contract lacks artifact role mapping")
    required = {
        "initial_state",
        "normalization",
        "normalization_metadata",
        "sample_orders",
        "onecycle_trace",
        "profile_inputs",
        "profile_inputs_receipt",
    }
    if not required <= set(artifacts):
        raise ValueError(f"GPU producer omits shared roles: {sorted(required - set(artifacts))}")
    resolved = {}
    for name in required:
        item = artifacts[name]
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("GPU shared paths must be relative to out/training_contract.json")
        artifact = path.parent / relative
        artifact.resolve(strict=True).relative_to(path.parent.resolve(strict=True))
        if artifact.stat().st_size != item["size_bytes"] or digest(artifact) != item["sha256"]:
            raise ValueError(f"GPU shared artifact changed: {name}")
        resolved[name] = artifact
    return contract, resolved


def check_admission(admission: Record, source_manifest_sha: str) -> None:
    """Never turn a prepared packet or passing parity into root spending authority."""
    expected = {
        "schema": "wnn.loss-pair-tpu.root-admission.v1",
        "source_manifest_sha256": source_manifest_sha,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA,
        "cache_key": CACHE_KEY,
        "gpu_kernel": GPU_KERNEL,
        "cpu_kernel": CPU_KERNEL,
    }
    if any(admission.get(k) != value for k, value in expected.items()):
        raise ValueError("root admission identity differs from packet source/data contract")
    if admission.get("push_allowed") is not True or not admission.get(
        "gpu_training_contract_sha256"
    ):
        raise ValueError("packet has pending root/provider/producer admission")
    if admission.get("runtime_seconds", 0) > 3300 or admission.get("runtime_seconds", 0) <= 0:
        raise ValueError("worker runtime exceeds finite packet envelope")


def main() -> None:
    """Validate mounted inputs, profile actual TPU arithmetic, then conditional fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/out"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    admission = read_json(args.admission)
    manifest = read_json(args.source_manifest)
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"sealed source changed: {row['path']}")
    if args.offline:
        # Source validation can run without provider admission or imported JAX.
        write_json(
            args.output_root / "offline_source_receipt.json",
            {
                "status": "pass",
                "source_manifest_sha256": digest(args.source_manifest),
                "fit_admitted": False,
            },
        )
        return
    check_admission(admission, digest(args.source_manifest))
    cache_manifest = discover_pinned(args.input_root, "manifest.json", CACHE_MANIFEST_SHA)
    gpu_contract = discover_pinned(
        args.input_root, "training_contract.json", admission["gpu_training_contract_sha256"]
    )
    _, shared = verify_shared_contract(gpu_contract, admission)
    write_json(
        args.output_root / "phase.json",
        {
            "phase": "cache_mount_validation",
            "cache_manifest": str(cache_manifest),
            "gpu_contract": str(gpu_contract),
        },
    )
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.parsed_sequence_cache_reader import verify_parsed_cache

    validation_path = args.output_root / "tpu_mount_reader_validation.json"
    _, validation_sha = verify_parsed_cache(cache_manifest, CACHE_MANIFEST_SHA, validation_path)
    parity_root = args.output_root / "device_parity"
    write_json(
        args.output_root / "phase.json",
        {"phase": "actual_device_parity", "elapsed_seconds": time.monotonic() - start},
    )
    parity_command = [
        sys.executable,
        str(HERE / "run_device_parity.py"),
        "--initial-state",
        str(shared["initial_state"]),
        "--profile-inputs",
        str(shared["profile_inputs"]),
        "--profile-inputs-receipt",
        str(shared["profile_inputs_receipt"]),
        "--output-root",
        str(parity_root),
    ]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "tpu"
    env["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    env["OMP_NUM_THREADS"] = "1"
    with (args.output_root / "device_parity.log").open("w") as log:
        subprocess.run(
            parity_command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=min(1200, admission["runtime_seconds"]),
            check=True,
        )
    parity = read_json(parity_root / "device_parity_receipt.json")
    if parity.get("status") != "pass":
        raise ValueError("actual-device gate did not pass")
    if admission.get("conditional_fit_allowed") is not True:
        write_json(
            args.output_root / "terminal_receipt.json",
            {
                "status": "parity_complete_fit_not_admitted",
                "parity": parity,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        return
    remaining = admission["runtime_seconds"] - (time.monotonic() - start)
    estimated_fit_seconds = sum(
        row["warm_update_seconds"] * 220 for row in parity["profile"].values()
    )
    export_reserve = admission.get("evaluation_export_reserve_seconds", 600)
    if estimated_fit_seconds + export_reserve > remaining:
        write_json(
            args.output_root / "terminal_receipt.json",
            {
                "status": "parity_complete_pair_deferred_runtime",
                "estimated_fit_seconds": estimated_fit_seconds,
                "remaining_seconds": remaining,
                "parity": parity,
            },
        )
        return
    from tpu_worker import EXPECTED_RECIPE

    recipe = {**EXPECTED_RECIPE, "precision": "fp32", "losses": ["wmse", "huber"]}
    artifacts = {}
    bindings = {
        **{name: shared[name] for name in ("initial_state", "sample_orders", "onecycle_trace")},
        "cache_manifest": cache_manifest,
        "cache_validation_receipt": validation_path,
        "device_parity_receipt": parity_root / "device_parity_receipt.json",
    }
    for name, path in bindings.items():
        artifacts[name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    contract = {
        "schema": "wnn.loss-pair-tpu-run-contract.v1",
        "recipe": recipe,
        "identity": {
            "fold": 0,
            "cache_key": CACHE_KEY,
            "split_manifest_sha256": SPLIT_SHA,
            "membership_sha256": MEMBERSHIP_SHA,
            "portable_registry_sha256": REGISTRY_SHA,
        },
        "artifacts": artifacts,
        "ancestry": {
            "gpu_training_contract_sha256": digest(gpu_contract),
            "root_admission_sha256": digest(args.admission),
            "cache_mount_validation_sha256": validation_sha,
        },
    }
    contract_path = args.output_root / "tpu_training_contract.json"
    write_json(contract_path, contract)
    write_json(
        args.output_root / "phase.json",
        {
            "phase": "matched_fit_pair",
            "elapsed_seconds": time.monotonic() - start,
            "estimated_fit_seconds": estimated_fit_seconds,
            "remaining_seconds": remaining,
        },
    )
    with (args.output_root / "fit_pair.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(HERE / "tpu_worker.py"),
                "--contract",
                str(contract_path),
                "--output-root",
                str(args.output_root / "fit_pair"),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=max(1, int(remaining)),
            check=True,
        )
    write_json(
        args.output_root / "terminal_receipt.json",
        {
            "status": "complete",
            "fit_terminal": read_json(args.output_root / "fit_pair/terminal_receipt.json"),
            "parity": parity,
            "elapsed_seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    main()
