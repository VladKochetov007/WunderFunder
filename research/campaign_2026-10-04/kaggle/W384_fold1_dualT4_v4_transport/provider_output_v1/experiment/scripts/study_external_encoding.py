from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    study = Path(__file__).resolve().parents[1] / "research" / "campaign_2026-09-29" / "external"
    sys.path.insert(0, str(study))
    runpy.run_path(str(study / "encoding_study.py"), run_name="__main__")
