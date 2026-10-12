#!/usr/bin/env python3
"""Bridge pinned F0 checkpoint arrays into the same callback benchmark on Python 3.11.

The exporter uses the reviewed Torch-enabled research interpreter. The runner uses the
existing Python 3.11 environment and frozen benchmark functions; it changes only how
already-exported float32 checkpoint arrays are supplied to the weight loader.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

BENCHMARK_SOURCE = Path(__file__).with_name("benchmark_shared_f0.py")
MODULE_NAME = "shared_f0_py311_benchmark"


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_benchmark() -> ModuleType:
    """Load the committed callback benchmark module from this packet."""
    spec = importlib.util.spec_from_file_location(MODULE_NAME, BENCHMARK_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen callback benchmark: {BENCHMARK_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def export_weights(args: argparse.Namespace) -> dict[str, Any]:
    """Export actual pinned model tensors and producer metadata for abi3 replay."""
    if args.weights.exists() or args.producer.exists():
        raise FileExistsError("refusing to overwrite Python 3.11 bridge inputs")
    benchmark = load_frozen_benchmark()
    weights, producer = benchmark.checkpoint_pair_evidence()
    arrays = {
        f"member{member}__{name}": values
        for member, tensors in enumerate(weights)
        for name, values in tensors.items()
    }
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.weights, **arrays)
    receipt = {
        "schema": "wnn.shared-f0-py311-weight-bridge.r1",
        "checkpoint_loader_source_sha256": sha256_file(BENCHMARK_SOURCE),
        "weights_archive": str(args.weights.resolve()),
        "weights_archive_sha256": sha256_file(args.weights),
        "member_tensor_names": [sorted(tensors) for tensors in weights],
        "producer": producer,
        "export_python": sys.version,
        "export_executable": sys.executable,
    }
    args.producer.parent.mkdir(parents=True, exist_ok=True)
    args.producer.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load_weight_arrays(path: Path, receipt: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    """Read only the exact arrays named in the pre-export receipt."""
    if sha256_file(path) != receipt["weights_archive_sha256"]:
        raise ValueError("pre-exported checkpoint array archive hash mismatch")
    members = receipt["member_tensor_names"]
    if len(members) != 2:
        raise ValueError("the frozen benchmark requires exactly two weight members")
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {
            f"member{member}__{name}" for member, names in enumerate(members) for name in names
        }
        if set(archive.files) != expected_keys:
            raise ValueError("weight archive keys do not match the export receipt")
        return [
            {
                name: np.ascontiguousarray(archive[f"member{member}__{name}"], dtype=np.float32)
                for name in names
            }
            for member, names in enumerate(members)
        ]


def run_py311(args: argparse.Namespace) -> dict[str, Any]:
    """Run the frozen baseline/shared comparison with Python 3.11 arrays."""
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Python 3.11 benchmark: {args.output}")
    receipt = json.loads(args.producer.read_text(encoding="utf-8"))
    weights = load_weight_arrays(args.weights, receipt)
    benchmark = load_frozen_benchmark()
    producer = receipt["producer"]
    benchmark.checkpoint_pair_evidence = lambda: (weights, producer)
    options = argparse.Namespace(
        pair="pair1",
        order=args.order,
        output=args.output,
        cpu=args.cpu,
    )
    result = benchmark.benchmark(options)
    result["python311_bridge"] = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "weights_archive_sha256": receipt["weights_archive_sha256"],
        "producer_export_receipt_sha256": sha256_file(args.producer),
        "weight_source": "actual float32 tensors from pinned checkpoints; same frozen loader and native constructor",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    """Parse one weight export or one Python 3.11 callback comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export-weights", action="store_true")
    mode.add_argument("--run-py311", action="store_true")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--order", choices=("separate-shared", "shared-separate"), default="separate-shared"
    )
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.run_py311 and args.output is None:
        parser.error("--output is required with --run-py311")
    return args


def main() -> None:
    """Export pinned arrays or execute the same callback benchmark under Python 3.11."""
    args = parse_args()
    result = export_weights(args) if args.export_weights else run_py311(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
