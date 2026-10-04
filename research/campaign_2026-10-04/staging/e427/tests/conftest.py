from __future__ import annotations

import sys
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[1] / "src"
ROOT_SRC = Path(__file__).resolve().parents[5] / "src"
sys.path[:0] = [str(OVERLAY), str(ROOT_SRC)]
