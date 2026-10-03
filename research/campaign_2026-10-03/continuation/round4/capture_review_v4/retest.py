from __future__ import annotations

import copy
import json
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any  # The bound capture contains heterogeneous nested state.

import numpy as np
import torch

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
BUILDER = OWN.parent / "live_capture_v4"
sys.path[:0] = [str(BUILDER), str(ROOT / "src")]
import live_capture as capture
import runner


def main() -> None:
    """Run only the known normalization witness and cheapest prior regressions."""
    started = time.monotonic()
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

    def reject(name: str, function: Any, *args: Any) -> None:
        before = capture.get_rng("cpu")
        try:
            function(*args)
        except (AssertionError, ValueError, TypeError) as error:
            checks[name] = {"pass": True, "reason": str(error)}
        else:
            checks[name] = {"pass": False, "reason": "accepted corruption"}
        capture.compare(before, capture.get_rng("cpu"))
        checks[name]["global_RNG_unchanged"] = True

    for offset in (28, 20, 5048):
        altered = copy.deepcopy(pre)
        altered["rng"]["torch_cpu"][offset] ^= 1
        for boundary, function, args in (
            ("validate_rng", capture.validate_rng, (altered["rng"], "cpu")),
            ("validate", capture.validate, (altered,)),
            ("restore", capture.restore, (altered, module)),
            ("set_rng", capture.set_rng, (altered["rng"], "cpu")),
        ):
            reject(f"offset{offset}_{boundary}", function, *args)
    initial = capture.get_rng("cpu")
    for seed in (23, 817):
        np.random.seed(seed)
        np.random.normal(size=3)
        random.seed(seed)
        random.gauss(0, 1)
        torch.manual_seed(seed)
        torch.rand(17)
        torch.randn(3)
        state = capture.get_rng("cpu")
        capture.validate_rng(state, "cpu")
        capture.compare(state, capture.get_rng("cpu"))
        private = torch.Generator(device="cpu")
        capture.validate_torch_rng(state["torch_cpu"], private, "cpu_private")
        capture.compare(state["torch_cpu"], private.get_state())
        capture.compare(state, capture.get_rng("cpu"))
        capture.set_rng(state, "cpu")
        capture.compare(state, capture.get_rng("cpu"))
        checks[f"seed{seed}_advanced_cached_no_draw_roundtrip"] = {"pass": True}
    capture.set_rng(initial, "cpu")
    for dtype in (np.uint64, np.float64, ">u4" if np.little_endian else "<u4"):
        altered = copy.deepcopy(pre)
        state = list(altered["rng"]["numpy"])
        state[1] = state[1].astype(dtype)
        altered["rng"]["numpy"] = tuple(state)
        reject(f"numpy_{np.dtype(dtype)}_restore", capture.restore, altered, module)
        reject(f"numpy_{np.dtype(dtype)}_compare", capture.compare, pre["rng"], altered["rng"])
    for field in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
        altered = copy.deepcopy(pre)
        identifier = next(i for i, state in altered["optimizer"]["state"].items() if field in state)
        altered["optimizer"]["state"][identifier][field] = altered["optimizer"]["state"][
            identifier
        ][field].double()
        reject(field + "_float64_restore", capture.restore, altered, module)
    altered = copy.deepcopy(pre)
    ids = altered["optimizer"]["param_groups"][1]["params"]
    a, b = ids[:2]
    mapping = altered["optimizer_id_names"]
    assert pre["student"][mapping[a]].shape == pre["student"][mapping[b]].shape
    ids[0], ids[1] = b, a
    mapping[a], mapping[b] = mapping[b], mapping[a]
    reject("same_shape_id_swap_forged", capture.restore, altered, module)
    altered = copy.deepcopy(pre)
    altered["clock"]["step_count"] = True
    reject("bool_clock_restore", capture.restore, altered, module)
    for name, a, b in (
        ("integer_clock", torch.tensor(100000), torch.tensor(100001)),
        ("scalar_bool_int", True, 1),
        ("array_bool_int", np.array([True]), np.array([1], dtype=np.uint32)),
        ("signed_zero", np.array([0.0]), np.array([-0.0])),
    ):
        reject(name, capture.compare, a, b)
    reject("pending_receipt", runner.verify_review, BUILDER / "review_request.json")
    reject("runtime_outside_GPU_chain", runner.runtime_guard)
    env = capture.restore(pre, module)
    capture.compare(pre["rng"], capture.get_rng("cpu"))
    capture.compare(pre["optimizer"], capture.cloned(env["opt"].state_dict()))
    checks["pristine_restore_rng_moments"] = {"pass": True}
    del env
    result = {
        "checks": checks,
        "count": len(checks),
        "all_pass": all(row["pass"] for row in checks.values()),
        "CPU_reproduction": {
            key: smoke[key]
            for key in (
                "normal_max_delta",
                "repeat_max_delta",
                "skip_max_delta",
                "uninstrumented_default_max_delta",
                "independent_floor",
                "skip_clock",
            )
        },
        "wall_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "GPU_launched": False,
        "CUDA_evidence": False,
    }
    (OWN / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}))
    assert result["all_pass"], "Known integrity regression remains"


if __name__ == "__main__":
    main()
