#!/usr/bin/env python3
"""Optional Harness² research-workflow entry point; never dispatches ML fits."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn.harness2_workflow.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
