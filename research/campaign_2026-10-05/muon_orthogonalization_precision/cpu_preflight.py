"""Fit-only CPU checks for the E487 one-step precision diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from precision_replay import run_fit_only_cpu_preflight


def main() -> None:
    """Run one actual fit-only B4×256 batch and write an immutable receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preflight output: {args.output}")
    torch.set_num_threads(1)
    with torch.autocast(device_type="cpu", enabled=False):
        result = run_fit_only_cpu_preflight()
    args.output.mkdir(parents=True)
    (args.output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
