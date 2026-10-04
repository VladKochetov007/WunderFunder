from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# utils.py is the organizers' reference scorer; it lives at the repository root.
sys.path.insert(0, str(ROOT))

DATASETS = ROOT / "datasets"


def pytest_collection_modifyitems(config, items):
    if (DATASETS / "valid.parquet").exists():
        return
    skip = pytest.mark.skip(reason="datasets/*.parquet not present")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
