from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.probe_factorized_native import FACTOR_NAMES, factor_arrays, load_native, metrics


def test_factor_arrays_preserve_exact_rank128_pairs() -> None:
    factors = {
        name: {
            "left": torch.from_numpy(np.zeros((1152, 128), dtype=np.float32)),
            "right": torch.from_numpy(np.zeros((128, 384), dtype=np.float32)),
        }
        for name in FACTOR_NAMES
    }
    result = factor_arrays(
        {"factorization": {"rank": 128, "training_updates": 0, "factors": factors}}
    )
    assert set(result) == set(FACTOR_NAMES)
    for name, pair in factors.items():
        for side, tensor in pair.items():
            assert result[name][side].dtype == np.float32
            assert result[name][side].tobytes() == tensor.numpy().tobytes()


def test_metric_gate_checks_full_row_error_and_pooled_wp() -> None:
    rng = np.random.default_rng(20261004)
    targets = rng.normal(size=(4096, 2)).astype(np.float32)
    reference = rng.normal(size=(4096, 23)).astype(np.float32)
    mask = np.arange(len(targets)) % 3 == 0
    stream = {"targets": targets, "mask": mask}

    exact = metrics([reference], [reference.copy()], [stream])
    assert exact["passed"]
    assert exact["max_abs_first_two_all_rows"] == 0.0

    changed = reference.copy()
    changed[:, 0] += np.float32(0.001)
    failed = metrics([changed], [reference], [stream])
    assert failed["max_abs_first_two_all_rows"] > 3e-4
    assert not failed["passed"]


def test_factor_arrays_reject_nonzero_update_control() -> None:
    with pytest.raises(ValueError, match="zero-update SVD export"):
        factor_arrays({"factorization": {"rank": 128, "training_updates": 1}})


@pytest.mark.parametrize("level", ["v2", "v3"])
def test_native_builds_load_independently(level: str) -> None:
    root = os.environ.get("WNN_FACTOR_LIBRARY_ROOT")
    if root is None:
        pytest.skip("set WNN_FACTOR_LIBRARY_ROOT to the new scorer builds")
    module = load_native(Path(root) / level / "wnn_native")
    assert module.ModelStepper.__name__ == "ModelStepper"
