"""Export a compact-GRU checkpoint as arrays accepted by the Rust stepper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    """Copy the checkpoint's model state and normalization into one NPZ file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arrays = {
        "mean": np.asarray(checkpoint["mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["std"], dtype=np.float32),
    }
    arrays.update(
        {
            name: tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            for name, tensor in checkpoint["state_dict"].items()
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
