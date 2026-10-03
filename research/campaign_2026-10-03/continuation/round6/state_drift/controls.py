"""Tiny actual-RNN controls, never opening a real capture or CUDA engine.

Any represents heterogeneous source-adapter dictionaries.
"""

from __future__ import annotations

import argparse
import copy
import json
import resource
import time
from typing import Any

import torch
from protocol import HERE, V4, cap, execute, local_update, prefix_state, require_runtime
from run_real import authorize, trajectory_check, warm_counts
from runner import synthetic_environment


def sequential_fixed(env: dict[str, Any]) -> torch.Tensor:
    """Original actual-RNN sequential carry starting None, fixed weights and four calls."""
    h = None
    with torch.no_grad():
        for start in (0, 10, 20, 30):
            _, h = env["forward_chunk"](env["model"], env["inputs"], slice(start, start + 10), h)
            h = env["detach_state"](h)
    return h


def main() -> None:
    """Run fixed-weight identity, planted weight drift and malformed-interface controls."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-label", default="cpu_controls")
    options = parser.parse_args()
    if not options.output_label.isidentifier():
        raise ValueError("CPU output label must be one directory identifier inside state_drift")
    started, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    torch.manual_seed(713601)
    source_before = {
        name: cap.sha(V4 / name)
        for name in (
            "live_capture.py",
            "runner.py",
            "baselines.py",
            "fit_panel.py",
            "export_schema.json",
        )
    }
    env = synthetic_environment("cpu")  # four actual updates, genuine separate EMA/h/moments
    observer = cap.Capture(True)
    env["_capture"] = observer
    exec(cap.compiled_chunk(True), env)
    stale, online = observer.pre, observer.post
    checks: dict[str, bool] = {}
    reference = prefix_state(stale)
    checks["planted_changing_weights_hidden_drift"] = not torch.equal(reference, stale["h"])
    checks["actual_CPU_warm_counts"] = warm_counts(stale) == (4, 0)
    # Schema-only planted skip count, not an AMP skip simulation or scientific capture.
    schema_skip = copy.deepcopy(stale)
    schema_skip["clock"]["skips"] = torch.tensor(2, dtype=torch.int64)
    for state in schema_skip["optimizer"]["state"].values():
        if "step" in state:
            state["step"] = 2
    checks["schema_counts_attempts_vs_completions"] = warm_counts(schema_skip) == (2, 2)
    next(state for state in schema_skip["optimizer"]["state"].values() if "step" in state)[
        "step"
    ] = 4
    try:
        warm_counts(schema_skip)
    except ValueError:
        checks["inconsistent_warm_moment_clock_rejected"] = True
    else:
        checks["inconsistent_warm_moment_clock_rejected"] = False
    fixed = copy.deepcopy(stale)
    fixed["h"] = reference
    fixed_env = cap.restore(fixed, cap.trainer_module())
    checks["fixed_prefix_vs_sequential_carry_bitexact"] = torch.equal(
        reference, sequential_fixed(fixed_env)
    )
    fixed_post, _ = local_update(fixed, reference)
    output = HERE / options.output_label
    null = execute(fixed, fixed_post, output / "fixed_null")
    planted = execute(stale, online, output / "planted_changing_weights")
    checks["unchanged_weight_null_no_difference"] = null[
        "outcome"
    ] == "UNRESOLVED_AT_NUMERIC_FLOOR" and all(
        v["rms_difference"] == 0 for v in null["effects"].values()
    )
    checks["repeat_floors_independent_and_exact"] = all(
        v == 0 for r in (null, planted) for v in r["independent_floors"]["floors_rms"].values()
    )
    checks["planted_drift_affects_prediction_gradient_update"] = all(
        planted["effects"][k]["rms_difference"] > 0
        for k in ("hidden", "prediction", "gradient", "update")
    )
    try:
        authorize(None, HERE / "ABSENT_pre.pt", HERE / "ABSENT_post.pt", HERE / "ABSENT_trace.json")
    except ValueError as error:
        checks["no_receipt_fails_before_capture_load_or_cuda_validation"] = "UNAPPROVED" in str(
            error
        )
    else:
        checks["no_receipt_fails_before_capture_load_or_cuda_validation"] = False
    fake_cuda = {**stale, "execution": {**stale["execution"], "device": "cuda"}}
    try:
        require_runtime(fake_cuda, None)
    except ValueError as error:
        checks["cuda_API_guard_before_private_validation"] = "UNAPPROVED" in str(error)
    else:
        checks["cuda_API_guard_before_private_validation"] = False
    try:
        trajectory_check(stale, {})
    except ValueError as error:
        checks["unknown_trajectory_rejected"] = "UNKNOWN" in str(error)
    else:
        checks["unknown_trajectory_rejected"] = False
    damaged = copy.deepcopy(stale)
    damaged["rng"]["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
    rng_before = torch.get_rng_state().clone()
    try:
        cap.validate_rng(damaged["rng"], "cpu")
    except ValueError:
        checks["private_rng_rejects_before_global_setter"] = torch.equal(
            rng_before, torch.get_rng_state()
        )
    else:
        checks["private_rng_rejects_before_global_setter"] = False
    checks["source_adapter_preserved"] = source_before == {
        name: cap.sha(V4 / name) for name in source_before
    }
    torch.save({"synthetic_pre": stale, "synthetic_post": online}, HERE / "synthetic_control.pt")
    results = {
        "passed": all(checks.values()),
        "checks": checks,
        "null": null,
        "planted": planted,
        "input_status": "no actual capture/data/score; CPU-protocol tiny RNN input3hidden4B2T60",
        "controls_output": options.output_label,
        "adapter_hashes": source_before,
        "source_hashes": cap.identities(),
        "cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": results["passed"], "checks": checks, "cost": results["cost"]}, indent=2
        )
    )
    assert results["passed"]


if __name__ == "__main__":
    main()
