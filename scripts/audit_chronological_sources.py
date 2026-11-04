"""Create the root-issued identity receipt for the official train and valid sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from wnn.chronological_source_audit import audit_sources
from wnn.chronological_source_index import SOURCE_AUDIT_PATH

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Audit the official sources under ``--root`` and publish the one-shot receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output is not None else root / SOURCE_AUDIT_PATH
    audit_sources(root, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
