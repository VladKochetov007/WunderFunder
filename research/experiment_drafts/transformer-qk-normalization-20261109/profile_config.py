"""Create a separate peak-LR engineering recipe; never change scientific configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Disable only warmup in an immutable diagnostic config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["warmup_fraction"] = 0.0
    with args.output.open("x") as stream:
        stream.write(json.dumps(config, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
