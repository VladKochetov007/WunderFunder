from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.multistage_delivery_core import run_delivery  # noqa: E402


def main() -> None:
    """Run the durable, deadline-bounded successor postprocess and Stage4 driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-root", "run-dir", "operator-root", "data-root", "auxiliary-split"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--revision", choices=("r1", "r2"), default="r1")
    args = parser.parse_args()
    paths = [
        args.runtime_root,
        args.run_dir,
        args.operator_root,
        args.data_root,
        args.auxiliary_split,
    ]
    result = run_delivery(
        *(path.resolve(strict=True) for path in paths),
        poll_seconds=args.poll_seconds,
        revision=args.revision,
    )
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
