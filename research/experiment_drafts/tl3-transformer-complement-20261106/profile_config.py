"""Materialize explicit diagnostic configs without altering production recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Write an idempotent one-rank or peak-LR diagnostic config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile-one", "profile-peak"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == "profile-one":
        config["required_world_size"] = 1
    else:
        config["warmup_fraction"] = 0.0
    if args.output.exists():
        if json.loads(args.output.read_text()) != config:
            raise ValueError("existing diagnostic config differs")
        return
    with args.output.open("x") as stream:
        stream.write(json.dumps(config, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
