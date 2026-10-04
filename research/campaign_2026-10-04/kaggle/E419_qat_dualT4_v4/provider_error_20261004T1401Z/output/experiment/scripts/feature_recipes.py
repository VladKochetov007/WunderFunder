from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from feature_cues import CUE_NAMES
from kaggle_common import canonical_sha256, relative_path, sha256_file


def descriptor(root: Path, name: str) -> dict:
    """Bind each code/cache byte before an experiment is admitted."""
    name = relative_path(name)
    path = root / name
    return {"path": name, "size": path.stat().st_size, "sha256": sha256_file(path)}


def head_partition(component: int) -> int:
    """Fixed component assignment independent of labels, predictions and row order."""
    return (
        int.from_bytes(
            hashlib.sha256(f"feature-head-v1:{component}".encode()).digest()[:8], "little"
        )
        % 2
    )


def recipe(root: Path, fit: str, evaluation: str, cue: str, backend: str, name: str) -> dict:
    """Seal an executable fixed feature screen; no cache means no fictitious sealed job."""
    if cue not in CUE_NAMES or backend not in ("cpu", "gpu", "tpu"):
        raise ValueError("Unsupported cue/backend")
    code_paths = [
        "scripts/feature_screen.py",
        "scripts/feature_cues.py",
        "src/wnn/research_features.py",
        "src/wnn/evaluation.py",
        "src/wnn/gmetric.py",
        "src/wnn/__init__.py",
        "src/wnn/resources.py",
    ]
    if backend == "gpu":
        code_paths += [
            "scripts/kaggle_common.py",
            "scripts/kaggle_gpu_runner.py",
            "scripts/kaggle_kernel.py",
        ]
    code = [descriptor(root, path) for path in sorted(code_paths)]
    inputs = [
        descriptor(root, path)
        for path in sorted(
            [
                fit,
                str(Path(fit).with_suffix(".json")),
                evaluation,
                str(Path(evaluation).with_suffix(".json")),
            ]
        )
    ]
    output = f"research/campaign_2026-10-04/feature_results/{relative_path(name)}"
    return {
        "version": 1,
        "name": name,
        "kind": backend,
        "argv": [
            "{python}",
            "scripts/feature_screen.py",
            "--fit",
            fit,
            "--evaluation",
            evaluation,
            "--cue",
            cue,
            "--backend",
            {"cpu": "numpy", "gpu": "torch", "tpu": "jax"}[backend],
            "--ridge",
            "0.01",
            "--output",
            output,
        ],
        "code": code,
        "code_sha256": canonical_sha256(code),
        "inputs": inputs,
        "artifacts": [
            f"{output}/{filename}"
            for filename in ("report.json", "coefficients.npz", "transforms.npz", "predictions.npz")
        ],
        "limits": {
            "cpus": 1,
            "ram_bytes": 4 * 1024**3,
            "timeout_seconds": 900,
            "gpu_memory_bytes": 0 if backend == "cpu" else 1024**3,
            "gpu_stop_bytes": 0 if backend == "cpu" else 2 * 1024**3,
        },
        "environment": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"} if backend == "tpu" else {},
    }


def main() -> None:
    """Generate a content-sealed draft for independent controller review, without fitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fit", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--cue", choices=CUE_NAMES, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu", "tpu"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = recipe(args.root, args.fit, args.evaluation, args.cue, args.backend, args.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
