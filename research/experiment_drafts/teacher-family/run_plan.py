from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Print the frozen per-arm matrix, without launching training."""
    packet = Path(__file__).resolve().parent
    root = packet.parents[2]
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(root / "scripts/experiment_draft_plan.py"),
                str(packet / "recipe.json"),
            ]
        )
    )


if __name__ == "__main__":
    main()
