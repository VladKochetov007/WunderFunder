from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

SHARED_PACKET = Path(__file__).resolve().parent
DEFAULT_STUDY = "source-holdout-transformer-kd-20261028"
QUALITY_STUDY = "transformer-width-context-20261028"
STUDY_ENV = "WNN_SEQUENCE_STUDY"


class Study(NamedTuple):
    """One explicitly registered frozen study, selected before module imports."""

    slug: str
    packet: Path
    cells: tuple[str, ...]
    allow_cache: bool


def select_study(slug: str | None = None) -> Study:
    """Resolve a closed study allowlist; never load arbitrary external configs."""
    selected = DEFAULT_STUDY if slug is None else slug
    if selected == DEFAULT_STUDY:
        roles = ("teacher", "labels", "kd")
    elif selected == QUALITY_STUDY:
        roles = ("baseline", "wide", "context")
    else:
        raise ValueError(f"Unregistered sequence study: {selected}")
    cells = tuple(f"{role}_s{seed}" for seed in (1001, 1002) for role in roles)
    return Study(selected, SHARED_PACKET.parent / selected, cells, selected == DEFAULT_STUDY)


STUDY = select_study(os.environ.get(STUDY_ENV))
