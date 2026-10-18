"""Create inference-only scaled aliases of R2; no training or refit occurs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_ALPHAS = (-0.5, -0.25, 0.0, 0.25, 0.5)
_DEFAULT_ALPHAS = (-0.5, -0.25, 0.25, 0.5)
_INPUT = (211, 271)
_PREFIX = "geometry_adapter_f0_s3_r2_scale_a_"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def create_aliases(
    checkpoint: Path,
    runs: Path,
    manifest: Path,
    scales: tuple[float, ...],
    expected_sha: str,
) -> dict[str, object]:
    """Write predictor-compatible aliases and provenance sidecars."""
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if manifest.exists():
        raise FileExistsError(manifest)
    if not scales or len(set(scales)) != len(scales) or any(a not in _ALPHAS for a in scales):
        raise ValueError(f"provide unique scales selected from {_ALPHAS}")
    names = [f"{_PREFIX}{'m' if a < 0 else 'p'}{abs(a):.2f}".replace(".", "p") for a in scales]
    if any((runs / name).exists() for name in names):
        raise FileExistsError("one or more alias run directories already exist")
    source_sha = sha256_file(checkpoint)
    if source_sha != expected_sha:
        raise ValueError("source checkpoint hash differs from the approved plan")
    source_provenance = checkpoint.with_name("provenance.json")
    if not source_provenance.is_file():
        raise FileNotFoundError(source_provenance)
    provenance_sha = sha256_file(source_provenance)

    import torch

    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source, dict) or not all(
        isinstance(source.get(key), dict) for key in ("model", "args")
    ):
        raise ValueError("checkpoint must contain model state and args mappings")
    model, args = source["model"], source["args"]
    weight = model.get("inp.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[1] != _INPUT[1]:
        raise ValueError("expected a 271-input checkpoint with inp.weight")
    geometry = weight[:, _INPUT[0] : _INPUT[1]]
    valid_geometry = (
        weight.is_floating_point()
        and torch.isfinite(geometry).all()
        and torch.count_nonzero(geometry)
    )
    if not valid_geometry:
        raise ValueError("source geometry weights must be finite and nonzero")

    aliases: list[dict[str, object]] = []
    for alpha, name in zip(scales, names, strict=True):
        alias_model = model.copy()
        alias_weight = weight.clone()
        alias_model["inp.weight"] = alias_weight
        alias = source.copy()
        alias["model"] = alias_model
        columns = alias_weight[:, _INPUT[0] : _INPUT[1]]
        columns.zero_() if alpha == 0.0 else columns.copy_(geometry * alpha)
        if not torch.equal(alias_weight[:, : _INPUT[0]], weight[:, : _INPUT[0]]) or not torch.equal(
            columns, geometry * alpha
        ):
            raise RuntimeError("alias input columns do not match the requested transform")
        run_dir = runs / name
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_out = run_dir / "model_ep0.pt"
        torch.save(alias, checkpoint_out)
        alias_sha = sha256_file(checkpoint_out)
        provenance = {
            "schema": "wnn.inference-checkpoint-alias-provenance.v1",
            "artifact_kind": "inference_only_checkpoint_alias",
            "run": name,
            "fit_action": "none; source args/provenance are historical, not a new fit",
            "source_checkpoint": {"path": str(checkpoint), "sha256": source_sha},
            "source_fit_ancestry": {"run": args.get("name"), "initializer": args.get("init")},
            "source_provenance": {"path": str(source_provenance), "sha256": provenance_sha},
            "inference_transform": {
                "alpha": alpha,
                "columns": list(_INPUT),
                "op": "scale inp.weight columns",
            },
            "alias_checkpoint_sha256": alias_sha,
        }
        sidecar = run_dir / "provenance.json"
        sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        aliases.append(
            {
                "alpha": alpha,
                "run": name,
                "checkpoint_sha256": alias_sha,
                "sidecar_sha256": sha256_file(sidecar),
            }
        )

    if sha256_file(checkpoint) != source_sha:
        raise RuntimeError("source checkpoint changed while creating aliases")
    record = {
        "schema": "wnn.geometry-alpha-inference-aliases.v1",
        "scope": "inference-only; no new fit, refit, or replication",
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": source_sha,
            "expected_sha256": expected_sha,
        },
        "checkpoint_mutation": "only inp.weight[:, 211:271] differs in checkpoint; each run has a separate inference-alias provenance sidecar",
        "aliases": aliases,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    """Create selected aliases without modifying the R2 checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--alpha", type=float, choices=_ALPHAS, action="append")
    args = parser.parse_args()
    scales = tuple(args.alpha if args.alpha is not None else _DEFAULT_ALPHAS)
    record = create_aliases(
        args.checkpoint.resolve(strict=True),
        args.runs_root.resolve(),
        args.manifest.resolve(),
        scales,
        args.expected_source_sha256,
    )
    print(json.dumps(record, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
