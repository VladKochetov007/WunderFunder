from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def test_native_prediction_file_includes_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept the full NPY and reject malformed payload boundaries and coverage."""
    packet = (
        Path(__file__).resolve().parents[1]
        / "research/campaign_2026-10-20/submission_delivery_r1/tl3_h512_endpoint_screen_r1"
    )
    monkeypatch.syspath_prepend(str(packet))
    spec = importlib.util.spec_from_file_location(
        "native_encoding_test", packet / "replay_h512_tl3_endpoint.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    path = tmp_path / "prediction.npy"
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(468, 20000, 2))
    assert values.offset > 0
    del values
    module.validate_prediction_encoding(path)
    with path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="encoding or coverage"):
        module.validate_prediction_encoding(path)
    path.unlink()
    np.save(path, np.zeros((1, 20000, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="encoding or coverage"):
        module.validate_prediction_encoding(path)
