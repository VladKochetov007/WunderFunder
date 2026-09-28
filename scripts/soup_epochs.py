"""Average fixed epoch checkpoints from one training trajectory into one run.

Example:
    .venv/bin/python scripts/soup_epochs.py ab68_ema6_since_f0 \
        ab68_ema6_since_soup_f0 3,4,5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Create a single checkpoint from prespecified epochs of one run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("epochs", help="comma-separated checkpoint epochs, fixed before scoring")
    args = parser.parse_args()
    epochs = [int(value) for value in args.epochs.split(",")]
    if len(epochs) < 2 or len(set(epochs)) != len(epochs):
        parser.error("provide at least two distinct epochs")
    source = ROOT / "artifacts" / "runs" / args.source
    checkpoints = [
        torch.load(source / f"model_ep{epoch}.pt", map_location="cpu", weights_only=False)
        for epoch in epochs
    ]
    reference = checkpoints[-1]
    keys = reference["model"].keys()
    for checkpoint in checkpoints[:-1]:
        if checkpoint["model"].keys() != keys or checkpoint["n_in"] != reference["n_in"]:
            raise ValueError("incompatible epoch checkpoints")
        for name in ("mean", "std"):
            if not torch.equal(checkpoint["model"][name], reference["model"][name]):
                raise ValueError(f"normalization buffer {name} changed across epochs")
    model = {}
    for name in keys:
        values = [checkpoint["model"][name] for checkpoint in checkpoints]
        if values[-1].is_floating_point() and name not in ("mean", "std"):
            model[name] = (
                torch.stack([value.float() for value in values]).mean(0).to(values[-1].dtype)
            )
        else:
            model[name] = values[-1]
    output = ROOT / "artifacts" / "runs" / args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.save({**reference, "model": model}, output / "model.pt")
    print(f"wrote {output / 'model.pt'} from {args.source} epochs {epochs}")


if __name__ == "__main__":
    main()
