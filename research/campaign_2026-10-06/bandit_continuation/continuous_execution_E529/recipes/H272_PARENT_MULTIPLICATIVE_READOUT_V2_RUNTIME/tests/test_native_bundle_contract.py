from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from serving.build_bundle import (  # noqa: E402
    EXPECTED_READOUT_SHAPES,
    _parent_native_arrays,
    _readout_arrays,
    _write_zip,
)


def _parent_state(n_out: int) -> dict[str, torch.Tensor]:
    state = {
        "mean": torch.zeros(211),
        "std": torch.ones(211),
        "inp.weight": torch.zeros(256, 211),
        "inp.bias": torch.zeros(256),
        "head.0.weight": torch.zeros(256, 512),
        "head.0.bias": torch.zeros(256),
        "head.2.weight": torch.zeros(n_out, 256),
        "head.2.bias": torch.zeros(n_out),
    }
    for layer in range(2):
        state[f"gru.weight_ih_l{layer}"] = torch.zeros(768, 256)
        state[f"gru.weight_hh_l{layer}"] = torch.zeros(768, 256)
        state[f"gru.bias_ih_l{layer}"] = torch.zeros(768)
        state[f"gru.bias_hh_l{layer}"] = torch.zeros(768)
    return state


def test_parent_npz_projection_accepts_only_the_bound_23_or_31_outputs() -> None:
    for width in (23, 31):
        arrays = _parent_native_arrays({"n_in": 211, "model": _parent_state(width)}, width)
        assert arrays["head.2.weight"].shape == (width, 256)
        assert all(
            value.dtype == np.float32 and value.flags.c_contiguous for value in arrays.values()
        )

    with pytest.raises(ValueError, match="bound 23/31 output count"):
        _parent_native_arrays({"n_in": 211, "model": _parent_state(24)}, 24)
    with pytest.raises(ValueError, match="bound 23/31 contract"):
        _parent_native_arrays({"n_in": 211, "model": _parent_state(23)}, 31)
    with pytest.raises(ValueError, match="missing native weights"):
        state = _parent_state(23)
        del state["gru.weight_hh_l1"]
        _parent_native_arrays({"n_in": 211, "model": state}, 23)


def test_readout_npz_requires_exact_finite_native_tensor_set(tmp_path: Path) -> None:
    archive_path = tmp_path / "head.npz"
    np.savez(
        archive_path,
        **{
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in EXPECTED_READOUT_SHAPES.items()
        },
    )
    arrays = _readout_arrays(archive_path)
    assert set(arrays) == set(EXPECTED_READOUT_SHAPES)

    incomplete = tmp_path / "incomplete.npz"
    np.savez(incomplete, **{"output.weight": np.zeros((2, 32), dtype=np.float32)})
    with pytest.raises(ValueError, match="tensor-name set"):
        _readout_arrays(incomplete)

    nonfinite = tmp_path / "nonfinite.npz"
    bad = {
        name: np.zeros(shape, dtype=np.float32) for name, shape in EXPECTED_READOUT_SHAPES.items()
    }
    bad["output.bias"][0] = np.nan
    np.savez(nonfinite, **bad)
    with pytest.raises(ValueError, match="nonfinite"):
        _readout_arrays(nonfinite)


def test_zip_manifest_assembly_is_closed_and_byte_reproducible(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "native").mkdir(parents=True)
    (package / "solution.py").write_text("class PredictionModel: pass\n")
    (package / "native/wnn_native.so").write_bytes(b"fixture native extension")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _write_zip(package, first)
    _write_zip(package, second)

    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
    with zipfile.ZipFile(first) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ["native/wnn_native.so", "solution.py"]
        assert archive.read("solution.py") == b"class PredictionModel: pass\n"
