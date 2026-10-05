"""Run unchanged E471 sources with one explicit shared full-FP32 CUDA policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    """Fix both affine backends, preserving the numerical checks and recipes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "fit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    if args.mode == "probe":
        import gpu_probe as experiment
    else:
        import fit_recurrent_candidate as experiment
    sys.argv = [str(Path(experiment.__file__)), "--device", "cuda", "--output", str(args.output)]
    experiment.main()
    path = args.output / "result.json"
    result = json.loads(path.read_text())
    result["precision_wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["explicit_precision_policy"] = {
        "mode": args.mode,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "changed_from_E470": "Historical cuDNN TF32 was enabled; E471 uses a common FP32 policy. Backend comparisons are descriptive, not bit-exact fit reproductions.",
    }
    if (
        result["explicit_precision_policy"]["cuda_matmul_allow_tf32"]
        or result["explicit_precision_policy"]["cudnn_allow_tf32"]
    ):
        raise ValueError("precision policy changed during E471 execution")
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
