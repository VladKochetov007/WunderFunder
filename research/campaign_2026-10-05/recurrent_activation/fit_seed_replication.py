"""Replicate all six E471 recipes, overriding both bound RNG seed defaults."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import fit_recurrent_candidate as experiment
import torch


def main() -> None:
    """Change initialization/order seed while preserving the shared FP32 recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(271829, 271830), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment.SEED = args.seed

    def replication_seed(seed: int = args.seed) -> None:
        torch.manual_seed(seed)

    # Merely changing the module constant leaves the original function's
    # definition-time default bound to 271828. Override that default explicitly.
    experiment.set_seed = replication_seed
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    sys.argv = [str(Path(experiment.__file__)), "--device", "cuda", "--output", str(args.output)]
    experiment.main()
    path = args.output / "result.json"
    result = json.loads(path.read_text())
    if result["seed"] != args.seed or len(result["arms"]) != 6:
        raise ValueError("replication has wrong seed or missing scientific arms")
    result["replication_wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["explicit_precision_policy"] = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if result["explicit_precision_policy"] != {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_enabled": True,
        "float32_matmul_precision": "highest",
    }:
        raise ValueError("shared precision policy changed during replication")
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
