from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
PACKET = ROOT / "runtime/short4h-20261022-r1"
STATE = ROOT / "state/short4h-20261022-r1"
SOURCE = ROOT / "runtime/source-a266-lf-subset-v1"
EXPERIMENT = "TABDATA-SHORT4H-20261022-R1"
sys.path[:0] = [str(ROOT / "runtime/parallel-expansion-v1/code"), str(SOURCE / "src")]
