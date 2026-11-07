from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]

def main() -> int:
    """Run the fixed-eight launcher CLI after adding repository import paths."""
    from galam_l4_quality8.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
