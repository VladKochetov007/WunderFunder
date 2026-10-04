"""Run only the staged event-clock implementation checks with its import overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGING = ROOT / "research/campaign_2026-10-04/staging/e426"
sys.path[0:0] = [str(STAGING / "src"), str(ROOT / "src"), str(ROOT / "scripts")]


def main() -> int:
    """Import pytest after installing the isolated module overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junitxml", type=Path, required=True)
    args = parser.parse_args()
    if args.junitxml.exists():
        raise FileExistsError(f"Preserve previous check evidence: {args.junitxml}")
    args.junitxml.parent.mkdir(parents=True, exist_ok=True)
    import pytest

    return pytest.main(
        [
            str(STAGING / "tests/test_e426_event_ema.py"),
            str(STAGING / "tests/test_e426_trainer_resume.py"),
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={args.junitxml}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
