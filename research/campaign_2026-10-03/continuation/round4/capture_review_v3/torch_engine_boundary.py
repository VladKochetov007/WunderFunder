from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
sys.path[:0] = [str(OWN.parent / "live_capture_v3"), str(ROOT / "src")]
import live_capture as capture


def main() -> None:
    """Check a few opaque state-byte boundaries on a private CPU engine only."""
    pre = torch.load(OWN / "cpu_reproduction/normal/pre.pt", weights_only=False)
    checks = []
    for offset in (0, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 5048, 5052):
        altered = copy.deepcopy(pre["rng"])
        altered["torch_cpu"][offset] ^= 1
        engine = torch.Generator(device="cpu")
        row = {"offset": offset}
        try:
            capture.validate_rng(altered, "cpu")
            row["validate_accepted"] = True
            engine.set_state(altered["torch_cpu"])
            row["private_roundtrip_exact"] = torch.equal(engine.get_state(), altered["torch_cpu"])
        except ValueError as error:
            row.update(validate_accepted=False, reason=str(error))
        checks.append(row)
    altered = copy.deepcopy(pre)
    altered["rng"]["torch_cpu"][28] ^= 1
    initial = capture.get_rng("cpu")
    witness = {"offset": 28, "metadata_identical": True}
    capture.validate(altered)
    witness["full_prestate_validate_accepted"] = True
    env = capture.restore(altered, capture.trainer_module())
    witness["restore_accepted"] = True
    actual = capture.get_rng("cpu")
    witness["restored_byte"] = int(actual["torch_cpu"][28])
    witness["declared_byte"] = int(altered["rng"]["torch_cpu"][28])
    try:
        capture.compare(altered["rng"], actual)
        witness["restored_RNG_compare_rejected"] = False
    except AssertionError:
        witness["restored_RNG_compare_rejected"] = True
    del env
    capture.set_rng(initial, "cpu")
    capture.compare(initial, capture.get_rng("cpu"))
    result = {
        "checks": checks,
        "actual_restore_witness": witness,
        "accepted_normalized": [
            r for r in checks if r.get("validate_accepted") and not r["private_roundtrip_exact"]
        ],
        "global_RNG_untouched": True,
        "CUDA_evidence": False,
    }
    (OWN / "torch_engine_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
