"""Run the deterministic native minGRU callback and write its evidence receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
import wnn_native

from wnn.features import StreamingEventFeatures
from wnn.min_gru import MinGRUModel


def make_inputs(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed raw rows and their Python causal 211-column expansion."""
    raw = np.random.default_rng(seed).normal(0.0, 0.2, (rows, 112)).astype(np.float32)
    stream = StreamingEventFeatures()
    events = np.empty((rows, 99), dtype=np.float32)
    for index, row in enumerate(raw):
        events[index] = stream.update(row)
    return raw, np.concatenate((raw, events), axis=1)


def model_weights(seed: int) -> tuple[MinGRUModel, dict[str, np.ndarray]]:
    """Create a small independent-width two-layer minGRU with bound state-dict bytes."""
    torch.manual_seed(seed)
    model = MinGRUModel(
        211,
        hidden=12,
        layers=2,
        encoder_width=12,
        recurrent_width=8,
        readout_width=6,
    ).eval()
    with torch.no_grad():
        model.mean.copy_(torch.linspace(-0.1, 0.1, 211))
        model.std.copy_(torch.linspace(0.7, 1.3, 211))
    weights = {
        name: np.ascontiguousarray(value.detach().cpu().numpy())
        for name, value in model.state_dict().items()
    }
    return model, weights


def digest_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def digest_weights(weights: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def cpu_model() -> str:
    """Return the Linux CPU model name when available."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right), initial=0.0))


def timed_run(predictor: object, rows: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    output = predictor.run(rows)
    return output, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/campaign_2026-10-14/cpu_student_frontier_r1/"
            "min_gru_native_r1/RUNTIME_PROBE_R1.json"
        ),
    )
    parser.add_argument("--rows", type=int, default=20_000)
    args = parser.parse_args()
    if args.rows < 100:
        raise ValueError("probe requires at least 100 rows for the causal prefix check")
    torch.set_num_threads(1)

    model, weights = model_weights(20261014)
    raw, model_inputs = make_inputs(args.rows, 10062026)
    with torch.inference_mode():
        started = time.perf_counter()
        reference, reference_state = model(torch.from_numpy(model_inputs[None]))
        reference_seconds = time.perf_counter() - started
    reference = reference[0].numpy()
    reference_state = reference_state[:, 0].numpy()

    stepper = wnn_native.MinGruStepper(weights, fast_activations=False, precision="f32")
    started = time.perf_counter()
    direct_output = stepper.run(model_inputs)
    direct_seconds = time.perf_counter() - started
    direct_state = stepper.hidden_state

    exact_construct_started = time.perf_counter()
    exact = wnn_native.MinGruPredictor(
        weights, fast_activations=False, precision="f32"
    )
    exact_construct_seconds = time.perf_counter() - exact_construct_started
    exact_output, exact_run_seconds = timed_run(exact, raw)
    exact_state = exact.hidden_state
    exact.reset()
    replay, replay_seconds = timed_run(exact, raw)
    if not np.array_equal(exact_output, replay):
        raise AssertionError("native exact callback did not replay bitwise after reset")

    chunked = wnn_native.MinGruPredictor(
        weights, fast_activations=False, precision="f32"
    )
    prefix_output = chunked.run(raw[:99])
    suffix_output = chunked.run(raw[99:])
    chunked_output = np.concatenate((prefix_output, suffix_output), axis=0)
    if not np.array_equal(exact_output, chunked_output):
        raise AssertionError("full and split native raw callbacks differ")

    exact_step = wnn_native.MinGruPredictor(
        weights, fast_activations=False, precision="f32"
    )
    started = time.perf_counter()
    per_row_output = np.stack([exact_step.step(row) for row in raw])
    per_row_seconds = time.perf_counter() - started
    if not np.array_equal(exact_output, per_row_output):
        raise AssertionError("one-row Python callback and native batch helper differ")

    fast = wnn_native.MinGruPredictor(
        weights, fast_activations=True, precision="f32"
    )
    fast_output, fast_seconds = timed_run(fast, raw)
    q8 = wnn_native.MinGruPredictor(
        weights, fast_activations=True, precision="q8vnni"
    )
    q8_output, q8_seconds = timed_run(q8, raw)
    if not np.isfinite(exact_output).all() or not np.isfinite(fast_output).all():
        raise AssertionError("FP32 callback emitted nonfinite values")
    if not np.isfinite(q8_output).all() or not np.isfinite(q8_state := q8.hidden_state).all():
        raise AssertionError("Q8 callback emitted nonfinite values")

    out = {
        "format": "wnn.native-min-gru-synthetic-callback-probe",
        "version": 1,
        "model": {
            "class": "wnn.min_gru.MinGRUModel",
            "state_dict_sha256": digest_weights(weights),
            "input_features": 211,
            "raw_features": 112,
            "event_features": 99,
            "encoder_width": 12,
            "recurrent_width": 8,
            "readout_width": 6,
            "layers": 2,
            "outputs": 2,
        },
        "synthetic_input": {
            "rows": args.rows,
            "seed": 10062026,
            "generator": "numpy.default_rng(seed).normal(0, 0.2).astype(float32)",
            "raw_payload_sha256": digest_array(raw),
            "stored_payload": False,
            "claim_limit": "Deterministic synthetic input and random weights; no fitting or grade.",
        },
        "reference": {
            "framework": f"torch {torch.__version__}",
            "python_version": platform.python_version(),
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "full_sequence_seconds": reference_seconds,
        },
        "execution_environment": {
            "cpu_model": cpu_model(),
            "machine": platform.machine(),
            "cpu_quota": os.environ.get("WNN_CPU_QUOTA", "unspecified"),
            "memory_max_bytes": int(os.environ["WNN_MEM_MAX"])
            if os.environ.get("WNN_MEM_MAX", "").isdigit()
            else os.environ.get("WNN_MEM_MAX", "unspecified"),
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
        "size_definition": "weight_bytes = packed or FP32 matrix payloads + FP32 biases + FP32 mean/std; storage_bytes additionally includes recurrent state, model scratch and raw-stream feature-builder state.",
        "native_exact_fp32": {
            "activation": "exact GELU erf and sigmoid",
            "precision": "f32",
            "construction_seconds": exact_construct_seconds,
            "weight_bytes": exact.weight_bytes,
            "storage_bytes": exact.storage_bytes,
            "bulk_callback_seconds": exact_run_seconds,
            "bulk_callback_ns_per_row": exact_run_seconds * 1e9 / args.rows,
            "python_step_callback_seconds": per_row_seconds,
            "python_step_callback_ns_per_row": per_row_seconds * 1e9 / args.rows,
            "split_at_row_99_bitwise_equal": True,
            "reset_replay_bitwise_equal": True,
            "reset_replay_seconds": replay_seconds,
            "max_abs_prediction_vs_python": max_abs(exact_output, reference),
            "max_abs_hidden_vs_python": max_abs(exact_state, reference_state),
            "max_abs_prediction_vs_direct_211_stepper": max_abs(exact_output, direct_output),
            "direct_211_stepper_seconds": direct_seconds,
            "direct_211_stepper_max_abs_prediction_vs_python": max_abs(
                direct_output, reference
            ),
            "direct_211_stepper_max_abs_hidden_vs_python": max_abs(
                direct_state, reference_state
            ),
            "max_abs_prefix_delta_vs_python": max_abs(exact_output[:99], reference[:99]),
            "final_hidden_state_sha256": digest_array(exact_state),
        },
        "native_fast_fp32": {
            "bulk_callback_seconds": fast_seconds,
            "max_abs_prediction_delta_vs_exact_fp32": max_abs(fast_output, exact_output),
            "max_abs_hidden_delta_vs_exact_fp32": max_abs(fast.hidden_state, exact_state),
        },
        "native_fast_q8vnni": {
            "bulk_callback_seconds": q8_seconds,
            "weight_bytes": q8.weight_bytes,
            "storage_bytes": q8.storage_bytes,
            "max_abs_prediction_delta_vs_exact_fp32": max_abs(q8_output, exact_output),
            "max_abs_hidden_delta_vs_exact_fp32": max_abs(q8_state, exact_state),
        },
        "interpretation": "Prototype transport/numerical evidence only; no trained model, quality claim, production replacement, or target-host speed claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
