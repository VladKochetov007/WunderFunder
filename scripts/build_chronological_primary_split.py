"""Freeze the official train-only fit and full-valid primary evaluation manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from wnn.chronological_manifest import build_chronological_primary_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Build one source-pinned chronological primary manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_chronological_primary_manifest(args.root, args.output)
    print(output)


if __name__ == "__main__":
    main()
