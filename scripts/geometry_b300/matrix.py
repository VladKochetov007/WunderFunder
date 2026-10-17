"""Validate the explicit geometry B300 cell and wave assignments."""

from __future__ import annotations

import json
from pathlib import Path

from common import pair_identity


def validate_frozen_matrix(root: Path) -> None:
    """Require the 120-fit matrix to match the paired-cell launch arithmetic."""
    path = root / "scripts/geometry_b300/matrix.json"
    matrix = json.loads(path.read_text())
    arms = ["baseline", "moments", "dictionary", "combined"]
    cells = [
        {"pair_index": index, "fold": pair_identity(index)[0], "seed": pair_identity(index)[1]}
        for index in range(30)
    ]
    waves = []
    for wave in range(15):
        pair_a, pair_b = 2 * wave, 2 * wave + 1
        assignments = []
        for gpu in range(8):
            pair_index = pair_a if gpu < 4 else pair_b
            fold, seed = pair_identity(pair_index)
            assignments.append(
                {
                    "gpu_slot": gpu,
                    "pair_index": pair_index,
                    "fold": fold,
                    "seed": seed,
                    "arm": arms[gpu % 4],
                }
            )
        waves.append(
            {"wave_index": wave, "pair_indices": [pair_a, pair_b], "assignments": assignments}
        )
    expected = {
        "schema": "causal_geometry_b300_matrix_v1",
        "arms": arms,
        "cells": cells,
        "waves": waves,
    }
    if matrix != expected:
        raise ValueError(f"frozen matrix does not match the 30-pair/15-wave launcher: {path}")
