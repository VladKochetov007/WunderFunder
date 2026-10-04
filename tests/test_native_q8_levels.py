from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_separate_v2_v3_processes_have_identical_q8_predictions(tmp_path: Path) -> None:
    libraries = os.environ.get("WNN_Q8_TEST_LIBRARIES")
    if not libraries:
        pytest.skip("both newly built native library levels required")
    code = """
import sys
import numpy as np
import torch
sys.path.insert(0, sys.argv[1])
import wnn_native as native
from wnn.nn import RNNModel
from wnn.quantization import quantize_model
torch.set_num_threads(1)
torch.manual_seed(617)
model = RNNModel(211, 33, 2).eval()
weights = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
features = np.random.default_rng(71).normal(size=(200, 112)).astype(np.float32)
result = {}
for preset, precision in [('recurrent', 'q8r'), ('recurrent_head', 'q8x')]:
    represented, codes, _ = quantize_model(weights, preset)
    predictor = native.Predictor(represented, precision=precision, q8_codes=codes)
    result[precision] = np.stack([predictor.step(row) for row in features])
np.savez(sys.argv[2], **result)
"""
    for level in ("v2", "v3"):
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(Path(libraries) / level),
                str(tmp_path / f"{level}.npz"),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    with np.load(tmp_path / "v2.npz") as left, np.load(tmp_path / "v3.npz") as right:
        for precision in ("q8r", "q8x"):
            np.testing.assert_array_equal(left[precision], right[precision])
