#!/usr/bin/env python3
"""Compare the AVX2 minGRU path with the sealed scalar Q8 callback."""

from __future__ import annotations

import argparse
import gc
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
    cgroup_snapshot,
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

BASELINE_NATIVE = Path(
    "/home/vlad/development/wunder/worktrees/min-gru-real-costprofile-20261014/"
    "native/target/release/libwnn_native.so"
)
BASELINE_NATIVE_SHA256 = "5e29328c940aeaada3f38c86a75a824585fedae78bf68d6a1dd5d78b98996f13"
BASELINE_OUTPUT_SHA256 = "0feb54067b2d40f207c4d5e2df46aabd17d76ff2b45b84df866dd61bc82ec6cc"
BASELINE_STATE_SHA256 = "4d1f832b5dce5d46f8f1025ea65dd5762fc1d2d2f95576b535808abb0bede009"


def run_callback(predictor: Any, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Run every raw row through the callback and return predictions, state and time."""
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
    optimized_native_path: Path,
    optimized_native_sha256: str,
    source_commit: str,
    repetitions: int,
    output_path: Path,
) -> None:
    """Compare full-sequence time and bitwise results for scalar and AVX2 paths."""
    if repetitions < 2:
        raise ValueError("at least two paired measured replays are required")
    before = {
        "resident_bytes": resident_bytes(),
        "process_max_rss_bytes": max_rss_bytes(),
        "cgroup": cgroup_snapshot(),
    }
    raw, mean, std, input_manifest = load_inputs(local_dir)
    weights, model_manifest = synthesize_weights("min_gru", mean, std)
    baseline_module = load_native(baseline_native_path, baseline_native_sha256)
    optimized_module = load_native(optimized_native_path, optimized_native_sha256)
    callback_type = callback_class(optimized_module, "min_gru")
    baseline_type = callback_class(baseline_module, "min_gru")

    baseline_started = time.perf_counter()
    baseline = baseline_type(weights, fast_activations=True, precision="q8vnni")
    baseline_constructor_seconds = time.perf_counter() - baseline_started
    optimized_started = time.perf_counter()
    optimized = callback_type(weights, fast_activations=True, precision="q8vnni")
    optimized_constructor_seconds = time.perf_counter() - optimized_started
    for name, predictor in (("baseline", baseline), ("optimized", optimized)):
        if predictor.precision != "q8vnni" or predictor.fast_activations is not True:
            raise AssertionError(f"{name} did not select Q8 VNNI and fast activations")
    if optimized.avx2_fastpath_enabled is not True:
        raise AssertionError("optimized callback did not assert the AVX2/FMA fast path")

    expected_state_shape = (2, SHAPES["min_gru"]["recurrent"])
    zero_state = np.zeros(expected_state_shape, dtype=np.float32)
    baseline.reset()
    optimized.reset()
    if not np.array_equal(hidden_state(baseline), zero_state):
        raise AssertionError("baseline reset did not restore zero recurrent state")
    if not np.array_equal(hidden_state(optimized), zero_state):
        raise AssertionError("optimized reset did not restore zero recurrent state")

    baseline_output, baseline_state, baseline_warmup_seconds = run_callback(baseline, raw)
    optimized_output, optimized_state, optimized_warmup_seconds = run_callback(optimized, raw)
    if not np.array_equal(optimized_output, baseline_output) or not np.array_equal(
        optimized_state, baseline_state
    ):
        raise AssertionError("warmup replay differs from baseline full-sequence predictions/state")
    baseline_output_sha256 = sha256_array(baseline_output)
    baseline_state_sha256 = sha256_array(baseline_state)
    if baseline_output_sha256 != BASELINE_OUTPUT_SHA256:
        raise AssertionError("sealed scalar baseline output hash changed")
    if baseline_state_sha256 != BASELINE_STATE_SHA256:
        raise AssertionError("sealed scalar baseline final-state hash changed")

    measured: dict[str, list[dict[str, Any]]] = {"baseline": [], "optimized": []}
    reference_output = baseline_output.copy()
    reference_state = baseline_state.copy()
    for pair_index in range(repetitions):
        order = [
            ("baseline", baseline),
            ("optimized", optimized),
        ]
        if pair_index % 2:
            order.reverse()
        for name, predictor in order:
            predictor.reset()
            state_before = hidden_state(predictor)
            if not np.array_equal(state_before, zero_state):
                raise AssertionError(f"{name} reset did not restore the expected zero state")
            predictions, final_state, elapsed = run_callback(predictor, raw)
            if not np.isfinite(predictions).all() or not np.isfinite(final_state).all():
                raise FloatingPointError(f"{name} emitted nonfinite predictions or state")
            if not np.array_equal(predictions, reference_output) or not np.array_equal(
                final_state, reference_state
            ):
                raise AssertionError(f"{name} full callback differs bitwise from the reference")
            measured[name].append(
                {
                    "pair_index": pair_index,
                    "seconds": elapsed,
                    "microseconds_per_raw_row": elapsed * 1_000_000 / len(raw),
                    "rows": len(raw),
                    "output_sha256": sha256_array(predictions),
                    "final_state_sha256": sha256_array(final_state),
                    "reset_state_all_zero": True,
                    "finite_predictions_and_state": True,
                    "bitwise_matches_sealed_scalar_baseline": True,
                }
            )
            del predictions, final_state
            gc.collect()

    baseline_times = [row["seconds"] for row in measured["baseline"]]
    optimized_times = [row["seconds"] for row in measured["optimized"]]
    baseline_median = float(np.median(baseline_times))
    optimized_median = float(np.median(optimized_times))
    baseline_storage = {key: int(value) for key, value in baseline.storage_bytes.items()}
    optimized_storage = {key: int(value) for key, value in optimized.storage_bytes.items()}
    after = {
        "resident_bytes": resident_bytes(),
        "process_max_rss_bytes": max_rss_bytes(),
        "cgroup": cgroup_snapshot(),
    }
    result = {
        "format": "wnn.min-gru-q8-fastpath-cost-profile",
        "version": 1,
        "method": {
            "callback": "20,000 sequential Python-to-native raw112 steps, including sanitization and causal feature construction to 211 columns",
            "weights": "same deterministic synthetic Q8 fixture and seed as the sealed matched-matrix profile",
            "reset": "both callbacks reset before each complete-sequence replay",
            "timing_order": "one warmup per callback, then alternating baseline-first and optimized-first paired measured replays",
        },
        "implementation": {
            "baseline_source_commit": "7364a452e0c72c2600a27e21cb66f6deb9e714ce",
            "baseline_binary": str(baseline_native_path),
            "baseline_binary_sha256": sha256_file(baseline_native_path),
            "optimized_source_commit": source_commit,
            "optimized_binary": str(optimized_native_path),
            "optimized_binary_sha256": sha256_file(optimized_native_path),
            "selected_policy": {
                "precision": optimized.precision,
                "fast_activations": optimized.fast_activations,
                "avx2_fma_fastpath_enabled": optimized.avx2_fastpath_enabled,
                "assertions_passed": True,
            },
        },
        "input": {
            "source": input_manifest["source"],
            "normalization": input_manifest["normalization"],
            "rows": ROWS,
            "raw_width": RAW_WIDTH,
        },
        "model": model_manifest,
        "constructor": {
            "baseline_seconds_including_q8_pack": baseline_constructor_seconds,
            "optimized_seconds_including_q8_pack": optimized_constructor_seconds,
        },
        "warmup": {
            "baseline_seconds": baseline_warmup_seconds,
            "optimized_seconds": optimized_warmup_seconds,
            "outputs_and_final_state_bitwise_equal": True,
        },
        "comparison": {
            "baseline_output_sha256": baseline_output_sha256,
            "baseline_final_state_sha256": baseline_state_sha256,
            "baseline_matches_sealed_output_and_state": True,
            "optimized_complete_sequence_bitwise_parity": True,
            "baseline_median_seconds": baseline_median,
            "optimized_median_seconds": optimized_median,
            "median_speedup_factor": baseline_median / optimized_median,
            "baseline_storage_bytes": baseline_storage,
            "optimized_storage_bytes": optimized_storage,
            "storage_breakdowns_equal": baseline_storage == optimized_storage,
            "weight_bytes_equal": int(baseline.weight_bytes) == int(optimized.weight_bytes),
        },
        "measured_replays": measured,
        "resources": {
            "cpu_model": cpu_model(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "before": before,
            "after": after,
        },
        "quality_claim": False,
        "limitations": "One Intel CPU, one fitting sequence of real features, and deterministic synthetic weights. Complete-sequence parity is only for this fixture; timings are local cost evidence, not fit quality, capacity, or competition-host performance.",
    }
    write_json(output_path, result)


def parse_args() -> argparse.Namespace:
    """Parse the source, input and receipt bindings for one bounded comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(
            "/home/vlad/development/wunder/worktrees/min-gru-real-costprofile-20261014/"
            "cache/min_gru_cost_profile_r1"
        ),
    )
    parser.add_argument("--baseline-native", type=Path, default=BASELINE_NATIVE)
    parser.add_argument("--baseline-sha256", default=BASELINE_NATIVE_SHA256)
    parser.add_argument("--optimized-native", type=Path, required=True)
    parser.add_argument("--optimized-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the frozen scalar-versus-AVX2 callback comparison."""
    args = parse_args()
    phase_worker(
        args.local_dir,
        args.baseline_native,
        args.baseline_sha256,
        args.optimized_native,
        args.optimized_sha256,
        args.source_commit,
        args.repetitions,
        args.output,
    )


if __name__ == "__main__":
    main()
