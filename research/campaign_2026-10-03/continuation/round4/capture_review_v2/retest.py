"""Independent targeted v2 gate regressions and one pristine CPU reproduction."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import resource
import sys
import time
from pathlib import Path

import torch

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
BUILDER = OWN.parent / "live_capture_v2"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BUILDER))
import baselines
import live_capture as capture
import runner

LOGGER = logging.getLogger(__name__)


def rejects(function, *args, **kwargs) -> dict:
    try:
        function(*args, **kwargs)
    except (AssertionError, ValueError) as error:
        return {"rejected": True, "reason": str(error)}
    return {"rejected": False}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.monotonic()
    torch.set_num_threads(1)
    capture.verify_generated()
    output = OWN / "cpu_reproduction"
    output.mkdir(exist_ok=False)
    smoke = runner.run_probe(output, "cpu")
    (output / "results.json").write_text(json.dumps(smoke, indent=2) + "\n")
    runner.output_hashes(output)
    pre = torch.load(output / "normal/pre.pt", weights_only=False)
    module = capture.trainer_module()
    checks = {}
    for forged in (False, True):
        corrupt = copy.deepcopy(pre)
        ids = corrupt["optimizer"]["param_groups"][1]["params"]
        a, b = ids[:2]
        ids[0], ids[1] = b, a
        if forged:
            association = corrupt["optimizer_id_names"]
            association[a], association[b] = association[b], association[a]
        checks[f"optimizer_swap_forged_{forged}_validate"] = rejects(capture.validate, corrupt)
        checks[f"optimizer_swap_forged_{forged}_restore"] = rejects(capture.restore, corrupt, module)
    for tag, a, b in (
        ("integer_100000_plus1", torch.tensor(100000), torch.tensor(100001)),
        ("bool_tensor", torch.tensor(True), torch.tensor(False)),
        ("bool_scalar_alias", True, 1),
        ("integer_scalar_alias", 1, 1.0),
        ("signed_zero", torch.tensor(0.0), torch.tensor(-0.0)),
    ):
        checks[tag] = rejects(capture.compare, a, b, 1e-5, 1e-4)
    rng = pre["rng"]["torch_cpu"]
    altered_rng = rng.clone()
    altered_rng[0] ^= 1
    checks["RNG_byte_change"] = rejects(capture.compare, rng, altered_rng, 1e-5, 1e-4)
    checks["naked_float_tolerance"] = rejects(capture.compare, torch.tensor(1.0), torch.tensor(1.000005), 1e-5, 1e-4)
    scalar = {"x": torch.tensor([1.0], dtype=torch.float32)}
    zero = baselines.derive_policy([copy.deepcopy(scalar) for _ in range(6)], "cpu")
    changed = {"x": torch.nextafter(scalar["x"], torch.tensor([2.0]))}
    checks["zero_floor_ulp_change"] = rejects(baselines.compare_policy, scalar, changed, zero)
    samples = [{"x": torch.tensor([1 + i * 1e-6])} for i in range(6)]
    # CPU scalar simulation of the device-labelled calibration rule, NOT CUDA.
    nonzero = baselines.derive_policy(samples, "cuda")
    limit = nonzero["limits"]["root.x"]
    within = {"x": scalar["x"] + limit * 0.75}
    outside = {"x": scalar["x"] + limit * 1.5}
    baselines.compare_policy(scalar, within, nonzero)
    checks["nonzero_floor_outside_rejected"] = rejects(baselines.compare_policy, scalar, outside, nonzero)
    checks["CPU_nonzero_floor_rejected"] = rejects(baselines.derive_policy, samples, "cpu")
    ceiling_samples = [{"x": torch.tensor([1.0 + i * 0.1])} for i in range(6)]
    checks["over_ceiling_rejected"] = rejects(baselines.derive_policy, ceiling_samples, "cuda")
    nonfinite = [{"x": torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0 + i * 1e-6])} for i in range(6)]
    nf_policy = baselines.derive_policy(nonfinite, "cuda")
    nf_changed = copy.deepcopy(nonfinite[0])
    nf_changed["x"][1] = -float("inf")
    checks["nonzero_floor_inf_sign_change"] = rejects(baselines.compare_policy, nonfinite[0], nf_changed, nf_policy)
    bad_samples = copy.deepcopy(nonfinite)
    bad_samples[3]["x"][0] = 0
    checks["unstable_baseline_nan_mask"] = rejects(baselines.derive_policy, bad_samples, "cuda")
    checks["exact_six_required"] = rejects(baselines.derive_policy, samples[:5], "cuda")
    checks["pending_receipt_rejected"] = rejects(runner.verify_review, BUILDER / "review_request.json")
    floor_files = [output / name / "frozen_floor.json" for name in ("normal_baselines", "skip_baselines")]
    actual_floors = [json.loads(path.read_text()) for path in floor_files]
    result = {
        "checks": checks,
        "all_adversaries_rejected": all(value["rejected"] for value in checks.values()),
        "within_nonzero_floor_control_passed": True,
        "simulated_nonzero_floor": nonzero,
        "CPU_reproduction": {key: smoke[key] for key in ("normal_max_delta", "repeat_max_delta", "skip_max_delta", "uninstrumented_default_max_delta", "independent_floor", "skip_clock")},
        "original_runs_preserved": sum(len(list((output / name).glob("original_default_*.pt"))) for name in ("normal_baselines", "skip_baselines")),
        "each_floor_has_six_sample_hashes": all(len(p["sample_hashes"]) == 6 for p in actual_floors),
        "predictors_target_clipnorm_saved": all(k in torch.load(output / "normal/post.pt", weights_only=False) for k in ("out", "out_ema", "target", "clip_norm")),
        "cost": {"wall_seconds": time.monotonic() - start, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "CUDA_evidence": False,
    }
    (OWN / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("RESULTS %s", json.dumps(result))


if __name__ == "__main__":
    main()
