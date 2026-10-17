"""Create inference-only scaled aliases of R2; no training or refit occurs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

_ALLOWED_ALPHAS = (-0.5, -0.25, 0.0, 0.25, 0.5)
_DEFAULT_ALPHAS = (-0.5, -0.25, 0.25, 0.5)
_INPUT_START = 211
_INPUT_END = 271
_RUN_PREFIX = "geometry_adapter_f0_s3_r2_scale_a_"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def create_aliases(
    checkpoint_path: Path,
    runs_root: Path,
    manifest_path: Path,
    alphas: tuple[float, ...],
) -> dict[str, Any]:
    """Write one `predict_run.py`-compatible checkpoint alias per scale."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("scales must be nonempty and unique")
    if any(alpha not in _ALLOWED_ALPHAS for alpha in alphas):
        raise ValueError(f"scales must be selected from {_ALLOWED_ALPHAS}")
    tokens = [f"{'m' if alpha < 0 else 'p'}{abs(alpha):.2f}".replace(".", "p") for alpha in alphas]
    run_names = [f"{_RUN_PREFIX}{token}" for token in tokens]
    output_dirs = [runs_root / name for name in run_names]
    existing = [str(path) for path in output_dirs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace existing run directories: {existing}")

    source_sha256 = sha256_file(checkpoint_path)
    import torch

    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict) or not isinstance(source.get("model"), dict):
        raise ValueError("checkpoint must contain a model state dictionary")
    source_state = source["model"]
    source_weight = source_state.get("inp.weight")
    if not isinstance(source_weight, torch.Tensor) or source_weight.ndim != 2:
        raise ValueError("checkpoint model must contain a two-dimensional inp.weight")
    if source_weight.shape[1] != _INPUT_END or not source_weight.is_floating_point():
        raise ValueError("expected a floating-point 271-input geometry checkpoint")
    geometry_weight = source_weight[:, _INPUT_START:_INPUT_END]
    if not bool(torch.isfinite(geometry_weight).all()) or not bool(
        torch.count_nonzero(geometry_weight)
    ):
        raise ValueError("source geometry weights must be finite and nonzero")

    aliases: list[dict[str, Any]] = []
    for alpha, run_name, output_dir in zip(alphas, run_names, output_dirs, strict=True):
        alias = copy.deepcopy(source)
        alias_state = alias["model"]
        alias_weight = alias_state["inp.weight"]
        if alpha == 0.0:
            alias_weight[:, _INPUT_START:_INPUT_END].zero_()
        else:
            alias_weight[:, _INPUT_START:_INPUT_END] = geometry_weight * alpha
        if not torch.equal(alias_weight[:, :_INPUT_START], source_weight[:, :_INPUT_START]):
            raise RuntimeError("alias changed a parent input column")
        if not torch.equal(alias_weight[:, _INPUT_START:_INPUT_END], geometry_weight * alpha):
            raise RuntimeError("alias geometry columns do not match the requested scale")
        for key, source_value in source_state.items():
            if (
                key != "inp.weight"
                and isinstance(source_value, torch.Tensor)
                and not torch.equal(alias_state[key], source_value)
            ):
                raise RuntimeError(f"alias changed non-input model tensor {key}")

        output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_out = output_dir / "model_ep0.pt"
        torch.save(alias, checkpoint_out)
        aliases.append(
            {
                "alpha": alpha,
                "run_name": run_name,
                "checkpoint_path": str(checkpoint_out),
                "checkpoint_bytes": checkpoint_out.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint_out),
                "geometry_input_columns": [_INPUT_START, _INPUT_END],
                "geometry_weight_nonzero_count": int(
                    torch.count_nonzero(alias_weight[:, _INPUT_START:_INPUT_END])
                ),
            }
        )

    if sha256_file(checkpoint_path) != source_sha256:
        raise RuntimeError("source checkpoint changed while creating aliases")
    manifest = {
        "schema": "wnn.geometry-alpha-inference-aliases.v1",
        "scope": "inference-only aliases; no new fit, refit, or replication",
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_bytes": checkpoint_path.stat().st_size,
        "source_checkpoint_sha256": source_sha256,
        "parent_geometry_columns": [_INPUT_START, _INPUT_END],
        "mutation": "scale only model['inp.weight'][:, 211:271]; all other checkpoint fields and model tensors are retained",
        "training_membership_and_ancestry": "copied unchanged from the R2 source checkpoint",
        "aliases": aliases,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    """Create selected alpha aliases without modifying the source checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        action="append",
        dest="alphas",
        help="repeat for selected scales; defaults to four nonzero scales (zero only when requested)",
    )
    args = parser.parse_args()
    alphas = tuple(_DEFAULT_ALPHAS if args.alphas is None else args.alphas)
    manifest = create_aliases(
        args.checkpoint.resolve(strict=True),
        args.runs_root.resolve(),
        args.manifest.resolve(),
        alphas,
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
