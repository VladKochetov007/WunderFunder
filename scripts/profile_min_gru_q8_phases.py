#!/usr/bin/env python3
"""Measure minGRU Q8 callback phases against its sealed uninstrumented baseline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from profile_min_gru_cost_pair import (
    RAW_WIDTH,
    ROWS,
    SHAPES,
    callback_class,
    cpu_model,
    hidden_state,
    load_inputs,
    load_native,
    max_rss_bytes,
    resident_bytes,
    sha256_array,
    sha256_file,
    synthesize_weights,
    write_json,
)

DEFAULT_LOCAL_DIR = Path(
    "/home/vlad/development/wunder/worktrees/min-gru-real-costprofile-20261014/"
    "cache/min_gru_cost_profile_r1"
)
DEFAULT_BASELINE_NATIVE = Path(
    "/home/vlad/development/wunder/worktrees/min-gru-real-costprofile-20261014/"
    "native/target/release/libwnn_native.so"
)
BASELINE_NATIVE_SHA256 = "5e29328c940aeaada3f38c86a75a824585fedae78bf68d6a1dd5d78b98996f13"
BASELINE_SOURCE_COMMIT = "7364a452e0c72c2600a27e21cb66f6deb9e714ce"
BASELINE_OUTPUT_SHA256 = "0feb54067b2d40f207c4d5e2df46aabd17d76ff2b45b84df866dd61bc82ec6cc"
BASELINE_STATE_SHA256 = "4d1f832b5dce5d46f8f1025ea65dd5762fc1d2d2f95576b535808abb0bede009"
PROFILE_SLOTS = {
    13: "normalization",
    14: "q8_activation_quantize_all_linears",
    15: "q8_matvec_all_linears",
    16: "fp32_matvec_all_linears",
    17: "encoder_activation",
    18: "candidate_gate_and_state_update_all_layers",
    19: "skip_state_projection_concat",
    20: "head_activation",
    21: "model_step_inclusive",
    22: "raw_sanitize_and_causal_feature_builder",
    23: "raw_callback_body_inclusive",
}


def process_snapshot() -> dict[str, int | None]:
    """Return process RSS values needed to bound the local measurement."""
    return {
        "resident_bytes": resident_bytes(),
        "max_rss_bytes": max_rss_bytes(),
    }


def run_callback(predictor: Any, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a complete raw sequence through the same per-row callback as the baseline."""
    predictions = np.empty((len(raw), 2), dtype=np.float32)
    started = time.perf_counter()
    for index, row in enumerate(raw):
        predictions[index] = predictor.step(row)
    elapsed = time.perf_counter() - started
    return predictions, hidden_state(predictor), elapsed


def phase_worker(
    local_dir: Path,
    baseline_native_path: Path,
    baseline_native_sha256: str,
    profile_native_path: Path,
    profile_native_sha256: str,
    profile_source_commit: str,
    repetitions: int,
    output_path: Path,
) -> None:
    """Compare the profile build with the immutable Q8 output and collect phase counters."""
    before = process_snapshot()
    raw, mean, std, input_manifest = load_inputs(local_dir)
    weights, model_manifest = synthesize_weights("min_gru", mean, std)
    baseline_module = load_native(baseline_native_path, baseline_native_sha256)
    profile_module = load_native(profile_native_path, profile_native_sha256)
    if not hasattr(profile_module, "profile_reset") or not hasattr(profile_module, "profile_cycles"):
        raise AssertionError("profile extension does not expose the feature-gated counter API")

    predictor_type = callback_class(profile_module, "min_gru")
    constructor_started = time.perf_counter()
    predictor = predictor_type(weights, fast_activations=True, precision="q8vnni")
    constructor_seconds = time.perf_counter() - constructor_started
    if predictor.precision != "q8vnni" or predictor.fast_activations is not True:
        raise AssertionError("profile callback did not select the baseline Q8/fast policy")

    baseline_type = callback_class(baseline_module, "min_gru")
    baseline = baseline_type(weights, fast_activations=True, precision="q8vnni")
    baseline.reset()
    baseline_state_before = hidden_state(baseline)
    if baseline_state_before.shape != (2, SHAPES["min_gru"]["recurrent"]) or np.any(
        baseline_state_before != 0.0
    ):
        raise AssertionError("baseline callback did not reset to two zero recurrent states")
    baseline_output, baseline_state, baseline_seconds = run_callback(baseline, raw)
    baseline_output_hash = sha256_array(baseline_output)
    baseline_state_hash = sha256_array(baseline_state)
    if baseline_output_hash != BASELINE_OUTPUT_SHA256 or baseline_state_hash != BASELINE_STATE_SHA256:
        raise AssertionError(
            "current fixture/legacy binary no longer reproduces the sealed Q8 output and state hashes"
        )

    measured: list[dict[str, Any]] = []
    phase_cycles: list[list[int]] = []
    for repetition in range(repetitions):
        predictor.reset()
        state_before = hidden_state(predictor)
        if state_before.shape != baseline_state_before.shape or np.any(state_before != 0.0):
            raise AssertionError("profile callback did not reset to the baseline zero-state shape")
        profile_module.profile_reset()
        predictions, final_state, elapsed = run_callback(predictor, raw)
        counters = [int(value) for value in profile_module.profile_cycles()]
        if len(counters) != 32:
            raise AssertionError(f"profile API returned {len(counters)} counters, expected 32")
        if not np.isfinite(predictions).all() or not np.isfinite(final_state).all():
            raise FloatingPointError("profile callback emitted nonfinite predictions or state")
        output_hash = sha256_array(predictions)
        state_hash = sha256_array(final_state)
        if not np.array_equal(predictions, baseline_output) or not np.array_equal(
            final_state, baseline_state
        ):
            raise AssertionError("profile instrumentation changed complete-sequence Q8 outputs/state")
        if counters[21] == 0 or counters[23] == 0:
            raise AssertionError("inclusive model/callback counters were not populated")
        phase_cycles.append(counters)
        measured.append(
            {
                "index": repetition,
                "seconds": elapsed,
                "microseconds_per_raw_row": elapsed * 1_000_000 / len(raw),
                "rows": len(raw),
                "output_sha256": output_hash,
                "final_state_sha256": state_hash,
                "bitwise_matches_sealed_uninstrumented_baseline": True,
                "all_predictions_and_state_finite": True,
                "reset_state_all_zero": True,
            }
        )
        del predictions, final_state
        gc.collect()

    mean_cycles = {
        str(slot): sum(run[slot] for run in phase_cycles) / repetitions / ROWS
        for slot in PROFILE_SLOTS
    }
    total_callback = mean_cycles["23"]
    if total_callback <= 0 or mean_cycles["14"] <= 0 or mean_cycles["15"] <= 0:
        raise AssertionError("callback, Q8 quantization, or Q8 matvec counters are empty")
    stage_shares = {
        PROFILE_SLOTS[slot]: {
            "tsc_ticks_per_raw_row": mean_cycles[str(slot)],
            "fraction_of_raw_callback_body_inclusive_ticks": mean_cycles[str(slot)] / total_callback,
        }
        for slot in PROFILE_SLOTS
        if slot != 23
    }
    after = process_snapshot()
    result = {
        "format": "wnn.min-gru-q8-phase-profile",
        "version": 1,
        "method": {
            "callback": "Python-to-native step for each of 20,000 source-qualified raw112 rows; native path sanitizes/builds causal 211 features and steps minGRU",
            "precision": "q8vnni asserted",
            "activation_policy": "fast activations asserted, matching sealed paired cost profile",
            "source_commit": profile_source_commit,
            "phase_slots": PROFILE_SLOTS,
            "counter_semantics": "x86 RDTSC deltas from the existing native profile API; phase slots are cumulative exclusive spans except model_step_inclusive and raw_callback_body_inclusive, which intentionally include nested spans",
            "candidate_and_gate_semantics": "unchanged: candidate x>=0 ? x+0.5 : sigmoid(x); write=sigmoid(gate); h=(1-write)*old+write*candidate",
        },
        "comparison": {
            "baseline_binary": str(baseline_native_path),
            "baseline_binary_sha256": sha256_file(baseline_native_path),
            "baseline_source_commit": BASELINE_SOURCE_COMMIT,
            "baseline_sealed_output_sha256": BASELINE_OUTPUT_SHA256,
            "baseline_sealed_state_sha256": BASELINE_STATE_SHA256,
            "profile_binary": str(profile_native_path),
            "profile_binary_sha256": sha256_file(profile_native_path),
            "baseline_callback_seconds": baseline_seconds,
            "profile_constructor_seconds_including_q8_pack": constructor_seconds,
            "profile_output_bitwise_equal_to_baseline": True,
            "profile_state_bitwise_equal_to_baseline": True,
        },
        "model": model_manifest,
        "input": {
            "source": input_manifest["source"],
            "normalization": input_manifest["normalization"],
            "rows": ROWS,
            "raw_width": RAW_WIDTH,
        },
        "baseline_reset": {
            "state_shape": list(baseline_state_before.shape),
            "state_all_zero": True,
        },
        "profile_repetitions": measured,
        "phase_cycles_mean_per_raw_row": mean_cycles,
        "phase_timing_summary": stage_shares,
        "profile_counter_consistency": {
            "q8_quantize_ticks_positive": mean_cycles["14"] > 0,
            "q8_matvec_ticks_positive": mean_cycles["15"] > 0,
            "model_inclusive_ticks_positive": mean_cycles["21"] > 0,
            "callback_inclusive_ticks_positive": mean_cycles["23"] > 0,
        },
        "resources": {
            "cpu_model": cpu_model(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "process_before": before,
            "process_after": after,
            "process_max_rss_bytes": max_rss_bytes(),
        },
        "quality_claim": False,
        "limitations": "One deterministic synthetic-weight minGRU, one fitting sequence and one Intel CPU. Counters are sampled RDTSC ticks, and atomic/counter instrumentation perturbs this build; this is hotspot evidence only, not an optimized-path result, score, capacity or competition-host runtime claim.",
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite phase-profile evidence: {output_path}")
    write_json(output_path, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--baseline-native", type=Path, default=DEFAULT_BASELINE_NATIVE)
    parser.add_argument("--baseline-sha256", default=BASELINE_NATIVE_SHA256)
    parser.add_argument("--profile-native", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--profile-source-commit", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    phase_worker(
        args.local_dir,
        args.baseline_native,
        args.baseline_sha256,
        args.profile_native,
        args.profile_sha256,
        args.profile_source_commit,
        args.repetitions,
        args.output,
    )


if __name__ == "__main__":
    main()
