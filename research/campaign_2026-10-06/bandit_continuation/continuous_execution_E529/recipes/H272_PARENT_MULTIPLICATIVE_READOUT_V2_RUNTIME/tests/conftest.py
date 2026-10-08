from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for path in (PACKAGE_ROOT, PACKAGE_ROOT / "scripts", PACKAGE_ROOT / "src"):
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)
