"""Test exact NumPy prestate identity, validation and restoration without edits."""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OWN.parent / "live_capture_v2"))
import live_capture as capture

LOGGER = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pre = torch.load(OWN / "cpu_reproduction/normal/pre.pt", weights_only=False)
    module = capture.trainer_module()
    results = {"original_rng_array_dtype": str(pre["rng"]["numpy"][1].dtype), "dtype_aliases": {}}
    for dtype in (np.uint64, np.float64):
        altered = copy.deepcopy(pre)
        state = list(altered["rng"]["numpy"])
        state[1] = state[1].astype(dtype)
        altered["rng"]["numpy"] = tuple(state)
        entry = {"values_equal": bool(np.array_equal(state[1], pre["rng"]["numpy"][1]))}
        try:
            capture.validate(altered)
            entry["validate_accepted"] = True
        except (ValueError, AssertionError, TypeError) as error:
            entry.update(validate_accepted=False, validation_reason=str(error))
        try:
            capture.restore(altered, module)
            actual = np.random.get_state()  # noqa: NPY002 -- inspect actual global source RNG
            entry["restore_accepted"] = True
            entry["restored_dtype"] = str(actual[1].dtype)
            entry["restored_values_equal_original"] = bool(np.array_equal(actual[1], pre["rng"]["numpy"][1]))
            entry["silently_cast_dtype"] = actual[1].dtype != state[1].dtype
        except (ValueError, AssertionError, TypeError) as error:
            entry.update(restore_accepted=False, restore_reason=str(error))
        try:
            capture.compare(pre["rng"], altered["rng"])
            entry["rng_compare_accepted"] = True
        except AssertionError:
            entry["rng_compare_accepted"] = False
        results["dtype_aliases"][str(np.dtype(dtype))] = entry
    results["same_shape_numeric_aliases"] = []
    for a, b in ((np.array([1], dtype=np.bool_), np.array([1], dtype=np.uint32)),
                 (np.array([1], dtype=np.uint32), np.array([1], dtype=np.float64)),
                 (np.array([0.0], dtype=np.float32), np.array([-0.0], dtype=np.float32))):
        try:
            capture.compare(a, b)
            accepted = True
        except AssertionError:
            accepted = False
        results["same_shape_numeric_aliases"].append({"left_dtype": str(a.dtype), "right_dtype": str(b.dtype), "accepted": accepted})
    (OWN / "rng_prestate_results.json").write_text(json.dumps(results, indent=2) + "\n")
    LOGGER.info("RNG_PRESTATE %s", json.dumps(results))


if __name__ == "__main__":
    main()
