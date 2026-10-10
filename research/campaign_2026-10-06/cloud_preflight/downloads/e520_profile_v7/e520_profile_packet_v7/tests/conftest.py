"""Load the isolated profile packet modules and pinned repository package."""

from __future__ import annotations

import sys
from pathlib import Path

candidate_root = Path(__file__).resolve().parents[1]
repository_root = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(repository_root / "src"))
sys.path.insert(0, str(candidate_root))
