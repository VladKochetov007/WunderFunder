"""Explicit input closures for separately admitted exact feature experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_recipes import descriptor
from feature_v2_screen import FAMILIES
from kaggle_common import canonical_sha256, relative_path


def recipe(
    root: Path,
    *,
    name: str,
    operation: str,
    backend: str = "cpu",
    manifest: str = "",
    base: str = "",
    router: str = "",
    fit: str = "",
    evaluation: str = "",
    fit_histories: str = "",
    eval_histories: str = "",
    family: str = "",
    seal: bool = False,
) -> dict:
    """Prepare an explicit draft, optionally sealing only when every input exists."""
    if backend not in ("cpu", "gpu", "tpu") or operation not in ("router", "histories", "screen"):
        raise ValueError("Unknown operation/backend")
    if operation != "screen" and backend != "cpu":
        raise ValueError("Router and history preparation are bounded CPU operations")
    relative_path(name)
    code = [path.relative_to(root).as_posix() for path in (root / "src/wnn").rglob("*.py")]
    code += [
        f"scripts/{path}.py"
        for path in (
            "feature_v2_recipes",
            "feature_v2_cache",
            "feature_v2_filters",
            "feature_v2_screen",
            "feature_cache",
            "feature_screen",
            "feature_cues",
            "feature_recipes",
            "kaggle_common",
        )
    ]
    if backend == "gpu":
        code += ["scripts/kaggle_gpu_runner.py", "scripts/kaggle_kernel.py"]
    output = f"research/campaign_2026-10-04/feature_v2_results/{name}"
    inputs: set[str] = set()

    def cache(path: str) -> None:
        inputs.update((relative_path(path), str(Path(path).with_suffix(".json"))))

    if operation == "screen":
        if family not in FAMILIES:
            raise ValueError("Unknown exact feature family")
        for path in (fit, evaluation, fit_histories, eval_histories):
            cache(path)
        argv = [
            "scripts/feature_v2_screen.py",
            "--fit",
            fit,
            "--evaluation",
            evaluation,
            "--fit-histories",
            fit_histories,
            "--eval-histories",
            eval_histories,
            "--family",
            family,
            "--backend",
            {"cpu": "numpy", "gpu": "torch", "tpu": "jax"}[backend],
            "--output",
            output,
        ]
        artifacts = [
            f"{output}/{file}"
            for file in ("report.json", "coefficients.npz", "transforms.npz", "predictions.npz")
        ]
    else:
        inputs.add(relative_path(manifest))
        selection = json.loads((root / manifest).read_text())
        inputs.add(selection["campaign_split"]["path"])
        # The canonical split loader validates these identities and graph caches.
        inputs.update(
            (
                "datasets/train.parquet",
                "datasets/valid.parquet",
                "datasets/valid_mask.parquet",
                "cache/chains.npz",
                "cache/chain_folds.npz",
                "cache/train_targets.npz",
                "cache/valid_targets.npz",
            )
        )
        argv = [
            "scripts/feature_v2_cache.py",
            "--root",
            ".",
            "--manifest",
            manifest,
            "--output",
            output + ".npz",
            operation,
        ]
        if operation == "router":
            argv += ["--labels", "cache/train_targets.npz", "--limit", "1024"]
        else:
            cache(base)
            cache(router)
            argv += ["--base", base, "--router", router]
        artifacts = [output + ".npz", output + ".json"]
    # Local campaign admission owns device visibility, including CPU isolation.
    environment = {}
    if backend == "tpu":
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    result = {
        "version": 1,
        "name": name,
        "kind": backend,
        "argv": ["{python}", *argv],
        "code": sorted(set(code)),
        "inputs": sorted(inputs),
        "artifacts": artifacts,
        "limits": {
            "cpus": 1,
            "ram_bytes": 4 * 1024**3,
            "timeout_seconds": 3600 if operation != "screen" else 900,
            "gpu_memory_bytes": 0 if backend == "cpu" else 1024**3,
            "gpu_stop_bytes": 0 if backend == "cpu" else 2 * 1024**3,
        },
        "environment": environment,
    }
    if seal:
        result["code"] = [descriptor(root, path) for path in result["code"]]
        result["inputs"] = [descriptor(root, path) for path in result["inputs"]]
        result["code_sha256"] = canonical_sha256(result["code"])
    return result


def main() -> None:
    """Write a concrete controller draft; missing inputs prevent sealing, never fitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--name", required=True)
    parser.add_argument("--operation", choices=("router", "histories", "screen"), required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu", "tpu"), default="cpu")
    for option in (
        "manifest",
        "base",
        "router",
        "fit",
        "evaluation",
        "fit-histories",
        "eval-histories",
        "family",
    ):
        parser.add_argument("--" + option, default="")
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    if output.exists():
        raise FileExistsError(output)
    result = recipe(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
