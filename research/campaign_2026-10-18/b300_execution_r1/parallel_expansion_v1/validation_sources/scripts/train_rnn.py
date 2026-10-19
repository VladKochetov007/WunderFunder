from __future__ import annotations

import os
from pathlib import Path

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
# Historical helpers expose heterogeneous model and checkpoint schemas.
from typing import Any

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS as CORRUPT_TRAIN_ROW_GROUPS
from wnn.data import N_FEATURES
from wnn.data import SEQUENCE_LENGTH as SEQUENCE_LENGTH
from wnn.data import WARMUP as WARMUP
from wnn.data import SequenceFile as SequenceFile
from wnn.features import FEATURE_NAMES
from wnn.propensity import propensity as propensity
from wnn.splits import dev_halves as dev_halves
from wnn.splits import valid_row_groups as valid_row_groups
from wnn.training.cli import build_parser as build_parser
from wnn.training.compatibility import LegacyTrainerAPI

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", SOURCE_ROOT)).resolve()
SEQ_BYTES = 20000 * (N_FEATURES + 3) * 4
EV_I0PX = N_FEATURES + FEATURE_NAMES.index("ev_i0px")
INPUT_PROPENSITY = None
ONLINE_NORM = False
READOUT = None
READOUT_TILTED = False
HEURISTICS = False
RAW_ONLY = False
RESEARCH_FEATURES = "none"
RESEARCH_FEATURE_PRIOR = None
RESEARCH_FEATURE_CONTROL = "none"
PRIVILEGED_MODE = "none"
PRIVILEGED_HORIZON = 8
SENTINEL_PREPROCESSOR = None
_api = LegacyTrainerAPI(globals())


def __getattr__(name: str) -> Any:
    """Adapt heterogeneous legacy exports; core training uses explicit imports."""
    return _api.attribute(name)


def main() -> None:
    """Run stateful recurrent training through reusable semantic components."""
    _api.main()


if __name__ == "__main__":
    main()
