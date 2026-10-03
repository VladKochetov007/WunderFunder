"""Blind scalar and metadata falsifiers; never loads models, data, or captures."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def recurrence(weights: list[float], inputs: list[float]) -> float:
    """Carry the state across the four calls without updating a reference weight."""
    state = 0.0
    for weight, value in zip(weights, inputs, strict=True):
        state = weight * state + value
    return state


def capture_gate(metadata: dict[str, Any], expected: dict[str, Any]) -> str:
    """Reject prospective metadata traps; actual evidence is not examined here."""
    return "READY_FOR_REPLAY_CHECK" if metadata == expected else "BLOCK"


def numeric_gate(policy: dict[str, Any] | None, residual: float) -> str:
    """Require a separately accepted, operation-specific unchanged-replay bound."""
    if not math.isfinite(residual) or residual < 0:
        return "BLOCK"
    if policy is None:
        return "UNKNOWN"
    if (policy.get("operation") != "unchanged_local_step"
            or policy.get("independently_accepted") is not True
            or policy.get("calibrated_from_effect") is not False):
        return "BLOCK"
    bound = policy.get("bound")
    if not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound < 0:
        return "BLOCK"
    return "PASS_CONTROL_ONLY" if residual <= bound else "BLOCK"


def main() -> None:
    start = time.perf_counter()
    prereg_path = HERE / "PREREG.json"
    prereg_bytes = prereg_path.read_bytes()
    prereg = json.loads(prereg_bytes)
    spec = prereg["toy"]
    inputs, current = spec["prefix_inputs"], spec["current_weight"]
    stale = recurrence(spec["warm_weights"], inputs)
    fresh = recurrence([current] * 4, inputs)
    identity = recurrence([current] * 4, inputs)
    # Differentiate the recurrence analytically, without a model/autograd import.
    h, dh = 0.0, 0.0
    for value in inputs:
        h, dh = current * h + value, h + current * dh
    residual = current * fresh + 0.25 - 1.0
    local_gradient = 2 * residual * fresh
    full_prefix_gradient = 2 * residual * (fresh + current * dh)
    assert stale != fresh
    assert identity == fresh
    assert local_gradient != full_prefix_gradient
    assert h == fresh
    expected = prereg["actual_capture_gate"]
    assert capture_gate(expected, expected) == "READY_FOR_REPLAY_CHECK"
    changes: dict[str, tuple[str, Any]] = {
        "one_4000_call": ("reference_calls", [[0, 4000]]),
        "ema_weights": ("reference_weight_source", "EMA student"),
        "mixed_reference_weights": ("current_weights_constant_through_reference", False),
        "bptt": ("reference_detached_for_local_step", False),
        "skipped_update": ("successful_warm_optimizer_updates", 3),
        "initial_cached_state": ("sequence_initial_state", "cached"),
        "prefix_consumes_local_rng": ("prefix_rng_side_effects_restored_before_local_comparison", False),
        "teacher_recomputed": ("held_fixed", ["student theta and buffers"]),
    }
    schema_results = {}
    for name, (key, value) in changes.items():
        bad = copy.deepcopy(expected)
        bad[key] = value
        schema_results[name] = capture_gate(bad, expected)
        assert schema_results[name] == "BLOCK"
    policy = {"operation": "unchanged_local_step", "independently_accepted": True,
              "calibrated_from_effect": False, "bound": 0.0}
    numeric_results = {
        "missing": numeric_gate(None, 0),
        "identity_zero_floor": numeric_gate(policy, 0),
        "zero_floor_relaxation": numeric_gate(policy, 1e-12),
        "wrong_operation": numeric_gate({**policy, "operation": "component_autograd_sum"}, 0),
        "effect_calibration": numeric_gate({**policy, "calibrated_from_effect": True}, 0),
        "nan_bound": numeric_gate({**policy, "bound": float("nan")}, 0),
        "nonfinite_residual": numeric_gate(policy, float("inf")),
    }
    assert numeric_results["missing"] == "UNKNOWN"
    assert numeric_results["identity_zero_floor"] == "PASS_CONTROL_ONLY"
    assert all(v == "BLOCK" for k, v in numeric_results.items()
               if k not in {"missing", "identity_zero_floor"})
    sources = [ROOT / "scripts/train_rnn.py", ROOT / "src/wnn/nn.py",
               ROOT / "research/campaign_2026-10-03/continuation/round5/mechanism_review/STATE.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources}
    results = {
        "scope": "scalar/schema positive and adversarial controls, not actual capture validation",
        "prereg_sha256_before_tests": hashlib.sha256(prereg_bytes).hexdigest(),
        "stale_state": stale, "fixed_current_state": fresh,
        "identity_state_residual": identity - fresh,
        "local_detached_gradient": local_gradient,
        "full_prefix_gradient": full_prefix_gradient,
        "reference_state_derivative": dh,
        "schema_negative_controls": schema_results,
        "numeric_controls": numeric_results,
        "source_hashes": hashes,
        "compute_seconds": time.perf_counter() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_calls": 0, "capture_reads": 0, "data_reads": 0,
        "candidate_code_reads": 0, "fits": 0,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
