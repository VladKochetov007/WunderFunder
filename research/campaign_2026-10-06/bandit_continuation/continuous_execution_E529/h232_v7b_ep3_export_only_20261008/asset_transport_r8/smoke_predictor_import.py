from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash a source file with bounded sequential reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def predictor_environment(source_root: Path, base: dict[str, str]) -> dict[str, str]:
    """Build the worker's child environment while hiding GPUs for `--help`."""
    source_package = source_root / "src"
    if not (source_package / "wnn").is_dir():
        raise FileNotFoundError(f"extracted WNN package is missing: {source_package / 'wnn'}")
    environment = base.copy()
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_package), *([existing_path] if existing_path else [])]
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "WNN_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "WNN_LOADER_WORKERS": "2",
            "WNN_LOADER_RAM_GB": "0.30",
            "WNN_GPU_FRACTION": "0.75",
        }
    )
    return environment


def verify(source_root: Path, source_archive: Path, receipt_path: Path) -> dict[str, object]:
    """Run the real frozen predictor's help path from its extracted source tree."""
    source_root = source_root.resolve(strict=True)
    source_archive = source_archive.resolve(strict=True)
    predictor = source_root / "scripts" / "predict_run.py"
    trainer = source_root / "scripts" / "train_rnn.py"
    package_init = source_root / "src" / "wnn" / "__init__.py"
    if not all(path.is_file() for path in (predictor, trainer, package_init)):
        raise FileNotFoundError("extracted predictor source is missing required files")
    argv = [sys.executable, str(predictor), "--help"]
    environment = predictor_environment(source_root, dict(os.environ))
    completed = subprocess.run(
        argv,
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    receipt: dict[str, object] = {
        "record_id": "H232_R8_ACTUAL_PREDICTOR_IMPORT_SMOKE_R1",
        "status": "passed" if completed.returncode == 0 else "failed",
        "argv": argv,
        "cwd": str(source_root),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "source_archive_sha256": sha256_file(source_archive),
        "source_sha256": {
            "predict_run.py": sha256_file(predictor),
            "train_rnn.py": sha256_file(trainer),
            "wnn/__init__.py": sha256_file(package_init),
        },
        "child_environment": {
            "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
            "WNN_THREADS": environment["WNN_THREADS"],
            "OMP_NUM_THREADS": environment["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": environment["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": environment["OPENBLAS_NUM_THREADS"],
            "WNN_LOADER_WORKERS": environment["WNN_LOADER_WORKERS"],
            "WNN_LOADER_RAM_GB": environment["WNN_LOADER_RAM_GB"],
            "WNN_GPU_FRACTION": environment["WNN_GPU_FRACTION"],
            "PYTHONPATH": environment["PYTHONPATH"],
        },
        "network_access": "none; only the local argparse --help path was executed",
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0 or "usage:" not in completed.stdout.lower():
        raise RuntimeError(f"predict_run.py --help import smoke failed: {completed.stderr}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Run the bounded predictor import smoke test and write its receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    verify(args.source_root, args.source_archive, args.receipt)


if __name__ == "__main__":
    main()
