"""Emit narrowly scoped RNG metadata from one pinned parent training-state file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
CHECKER_SHA256 = "f27091c864287d6a3b7c173619ad1af5db6887e9632c49a8c6d0bb50c3a9c1d1"
MAX_STATE_BYTES = 64_000_000


def _type_name(value: object) -> str:
    """Return a stable module-qualified Python type name."""
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _cardinality(value: object, torch: Any, np: Any) -> int | None:
    """Return container/tensor length only; never inspect element values."""
    if isinstance(value, (tuple, list, dict)):
        return len(value)
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, np.ndarray):
        return int(value.size)
    return None


def _component(value: object, torch: Any, np: Any) -> dict[str, object]:
    """Describe nested RNG structure without exposing any state values."""
    result: dict[str, object] = {"type": _type_name(value)}
    cardinality = _cardinality(value, torch, np)
    if cardinality is not None:
        result["cardinality"] = cardinality
    if isinstance(value, torch.Tensor):
        result["dtype"] = str(value.dtype)
        result["shape"] = list(value.shape)
        result["numel"] = int(value.numel())
        raw = value.detach().cpu().contiguous().numpy().tobytes(order="C")
        result["sha256"] = hashlib.sha256(raw).hexdigest()
    elif isinstance(value, np.ndarray):
        result["dtype"] = str(value.dtype)
        result["shape"] = list(value.shape)
        result["numel"] = int(value.size)
    elif isinstance(value, list):
        result["components"] = [
            _component(item, torch, np)
            if isinstance(item, torch.Tensor)
            else {"type": _type_name(item), "cardinality": _cardinality(item, torch, np)}
            for item in value
        ]
    elif isinstance(value, tuple):
        result["components"] = [
            {"type": _type_name(item), "cardinality": _cardinality(item, torch, np)}
            for item in value
        ]
    return result


def _stat(path: Path) -> dict[str, int]:
    """Read a regular file identity without following a symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size > MAX_STATE_BYTES:
        raise ValueError("training state is not a bounded regular file")
    return {
        "bytes": before.st_size,
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
    }


def _exclusive_json(path: Path, value: dict[str, object]) -> None:
    """Create one durable report, refusing to replace any prior output."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    """Restricted-load only training_state.pt and report RNG metadata only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state-stat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checker_dir = Path(__file__).resolve().parent
    checker_path = checker_dir / "parent_fit_terminal_check_v1.py"
    checker_hash = hashlib.sha256(checker_path.read_bytes()).hexdigest()
    if checker_hash != CHECKER_SHA256:
        raise ValueError("pinned parent checker source differs")
    sys.path.insert(0, str(checker_dir))
    import numpy as np
    import parent_fit_terminal_check_v1 as checker
    import torch

    if Path(checker.__file__).resolve() != checker_path.resolve():
        raise ValueError("restricted loader resolved outside the staged checker")
    expected = json.loads(args.state_stat.read_text(encoding="utf-8"))
    state_path = args.run_dir / "training_state.pt"
    before = _stat(state_path)
    if expected != before:
        raise ValueError("training-state stat differs from the failed verifier's capture")
    state, _digest_unused, loaded_stat = checker._load_checkpoint(state_path, MAX_STATE_BYTES)
    if loaded_stat != expected:
        raise ValueError("training-state identity changed during restricted load")
    if not isinstance(state, dict):
        raise ValueError("training state is not an object")
    format_name = state.get("format")
    next_epoch = state.get("next_epoch")
    if not isinstance(format_name, str) or type(next_epoch) is not int:
        raise ValueError("training-state format metadata is malformed")
    rng = state.get("rng")
    if not isinstance(rng, dict):
        raise ValueError("training-state RNG field is not a mapping")
    rng_report: dict[str, object] = {
        "keys": sorted(rng),
        "entries": {key: _component(value, torch, np) for key, value in sorted(rng.items())},
    }
    report = {
        "schema": "fg19.parent-fit-r2-rng-diagnostic.v2",
        "status": "DIAGNOSTIC_ONLY_NO_ACCEPTANCE",
        "run_name": RUN,
        "state_file_stat_only": expected,
        "format": format_name,
        "next_epoch": next_epoch,
        "torch_version": str(torch.__version__),
        "rng": rng_report,
        "limits": [
            "Only training_state.pt was loaded, through the exact source-pinned f270 weights_only loader with its narrow NumPy/TorchVersion allowlist.",
            "No model or optimizer values were inspected or emitted.",
            "Python/NumPy RNG values were not emitted; only structure/cardinality is reported.",
            "CUDA/CPU RNG tensors are represented only by dtype, shape, numel and SHA-256.",
            "This diagnostic is not a terminal verification or acceptance result.",
        ],
    }
    _exclusive_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
