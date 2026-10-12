from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(kernel_path: Path, input_root: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("R9 local mount validation receipt is create-only")
    wrapper = import_module("r9_wrapper_mount_validation", kernel_path)
    worker_bytes, binding_bytes, _ = wrapper._decode_embedded()
    bindings = json.loads(binding_bytes)
    with tempfile.TemporaryDirectory(prefix="wnn-r9-worker-") as temporary:
        worker_path = Path(temporary) / "kernel_body.py"
        worker_path.write_bytes(worker_bytes)
        worker = import_module("r9_embedded_worker_mount_validation", worker_path)
        source_root = worker.find_source_mount(input_root)
        manifest = worker.validate_checkpoint_input_dataset(source_root, bindings)
        verified = []
        for run_name, expected in bindings["checkpoints"].items():
            checkpoint, provenance = worker.canonical_source_artifacts(source_root, run_name)
            worker.sha_member(checkpoint, expected["model_ep2_bytes"], expected["model_ep2_sha256"])
            worker.sha_member(
                provenance, expected["provenance_bytes"], expected["provenance_sha256"]
            )
            worker.validate_provenance(provenance, run_name, expected)
            verified.extend((checkpoint, provenance))
        rows = [
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(source_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sorted(verified)
        ]
        receipt = {
            "schema": "wnn.r9.local_checkpoint_mount_fixture.r1",
            "status": "PASS_LOCAL_MOUNT_FIXTURE_ONLY",
            "kernel_script_sha256": sha256_file(kernel_path),
            "embedded_worker_sha256": hashlib.sha256(worker_bytes).hexdigest(),
            "embedded_bindings_sha256": hashlib.sha256(binding_bytes).hexdigest(),
            "input_root": str(input_root.resolve()),
            "dataset_mount": str(source_root.resolve()),
            "dataset_id": manifest["dataset_id"],
            "dataset_manifest_sha256": bindings["checkpoint_source_manifest_sha256"],
            "verified_checkpoint_and_provenance_files": rows,
            "cuda_or_model_execution": False,
            "export_workers_started": 0,
            "provider_actions": 0,
            "note": "Local staged-file mount simulation only; no Kaggle mount, worker, CUDA or model inference was started.",
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.kernel, args.input_root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
