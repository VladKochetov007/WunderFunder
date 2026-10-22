from __future__ import annotations

import sys
from collections.abc import Iterator
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


@pytest.fixture(autouse=True)
def strict_sequence_precision(request: pytest.FixtureRequest) -> Iterator[None]:
    """Give sequence-runtime receiver tests the same explicit precision baseline as fitting."""
    filename = request.node.path.name
    if not (
        filename.startswith("test_b300_")
        or filename
        in {
            "test_sequence_numeric.py",
            "test_numeric_sequence_integration.py",
        }
    ):
        yield
        return
    import torch

    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn
