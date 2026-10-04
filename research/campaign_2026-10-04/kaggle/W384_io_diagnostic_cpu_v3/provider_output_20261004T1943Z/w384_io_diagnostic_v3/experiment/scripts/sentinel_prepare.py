"""Freeze one feature-only calibration and initialize all eight H081 control arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wnn.resources import configure_torch
from wnn.sentinel import SENTINEL_ARMS, canonical_hash, initialize_sentinel_checkpoint
from wnn.sentinel_calibration import calibrate

ROOT = Path(__file__).resolve().parents[1]


def prepare(
    parent: Path,
    directory: Path,
    root: Path,
    sequence_count: int = 64,
    row_stride: int = 7,
    seed: int = 81081,
    split_manifest: Path | None = None,
) -> None:
    """Create immutable sibling initializers; no optimizer updates or labels are consumed."""
    if directory.exists():
        raise FileExistsError(f"refusing to replace sentinel initialization directory: {directory}")
    checkpoint, calibration = calibrate(
        parent, root, sequence_count, row_stride, seed, split_manifest
    )
    directory.mkdir(parents=True)
    (directory / "calibration.json").write_text(
        json.dumps(calibration, indent=2, allow_nan=False) + "\n"
    )
    for values, indicators in SENTINEL_ARMS:
        name = f"{values}_{indicators}"
        output = directory / name
        output.mkdir()
        torch.save(
            initialize_sentinel_checkpoint(checkpoint, calibration, values, indicators),
            output / "model.pt",
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "calibration_sha256": canonical_hash(calibration),
                "training_updates": 0,
                "arms": [f"{values}_{indicators}" for values, indicators in SENTINEL_ARMS],
                "counterfactuals": "replacement, explicit missingness and safe scaling are distinct contrasts",
                "native_status": "streaming transform implemented; native packaging requires a dedicated port",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--row-stride", type=int, default=7)
    parser.add_argument("--seed", type=int, default=81081)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    configure_torch()
    prepare(
        args.parent,
        args.directory,
        ROOT,
        args.sequences,
        args.row_stride,
        args.seed,
        args.split_manifest,
    )


if __name__ == "__main__":
    main()
