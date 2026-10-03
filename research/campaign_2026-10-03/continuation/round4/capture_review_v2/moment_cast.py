"""Independent exact warm-moment dtype check at the restore boundary."""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

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
    results = {}
    for field in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
        altered = copy.deepcopy(pre)
        identifier = next(i for i, state in altered["optimizer"]["state"].items() if field in state)
        original = altered["optimizer"]["state"][identifier][field]
        altered["optimizer"]["state"][identifier][field] = original.to(torch.float64)
        entry = {"original_dtype": str(original.dtype), "altered_dtype": "torch.float64"}
        try:
            capture.validate(altered)
            entry["validate_accepted"] = True
        except (ValueError, AssertionError, TypeError) as error:
            entry.update(validate_accepted=False, validation_reason=str(error))
        try:
            env = capture.restore(altered, module)
            restored = env["opt"].state_dict()["state"][identifier][field]
            entry["restore_accepted"] = True
            entry["restored_dtype"] = str(restored.dtype)
            entry["silently_cast_dtype"] = restored.dtype != torch.float64
            entry["restored_values_equal_original"] = torch.equal(restored, original)
            del env
        except (ValueError, AssertionError, TypeError) as error:
            entry.update(restore_accepted=False, restore_reason=str(error))
        results[field] = entry
    (OWN / "moment_cast_results.json").write_text(json.dumps(results, indent=2) + "\n")
    LOGGER.info("MOMENT_CAST %s", json.dumps(results))


if __name__ == "__main__":
    main()
