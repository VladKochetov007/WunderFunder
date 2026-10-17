"""Parity and callback profile for a staged native geometry package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load_solution(package: Path) -> ModuleType:
    """Load the exact solution module from a staged package directory."""
    path = package / "solution.py"
    spec = importlib.util.spec_from_file_location("geometry_solution_profile", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import package solution: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reference_modules():
    """Load canonical batch feature functions without initializing Torch."""
    sys.modules["torch"] = ModuleType("torch")
    from wnn.features import StreamingEventFeatures, event_features
    from wnn.moment_geometry.online import features_numpy

    return StreamingEventFeatures, event_features, features_numpy


def _synthetic_rows(count: int, seed: int) -> np.ndarray:
    """Create a bounded synthetic 112-column replay with discrete geometry support."""
    generator = np.random.default_rng(seed)
    rows = generator.normal(size=(count, 112)).astype(np.float32)
    levels = np.asarray((-1.0, 0.0, 1.0, 2.0), dtype=np.float32)
    for column in (0, 3, 52, 55):
        indices = generator.integers(0, len(levels), size=count)
        hold = generator.random(count) < 0.75
        for index in range(1, count):
            if hold[index]:
                indices[index] = indices[index - 1]
        rows[:, column] = levels[indices]
    return np.ascontiguousarray(rows)


def _load_member(package: Path, precision: str) -> tuple[dict[str, np.ndarray], np.float32]:
    """Load the one fixed 271-input package member."""
    spec = cast(list[dict[str, object]], json.loads((package / "members.json").read_text()))
    if len(spec) != 1:
        raise ValueError("geometry submission check expects exactly one fixed member")
    item = spec[0]
    with np.load(package / str(item["weights"]), allow_pickle=False) as archive:
        weights = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32) for name in archive.files
        }
    if weights.get("mean", np.empty(0)).shape != (271,):
        raise ValueError("staged geometry model must have 271 inputs")
    if precision not in {"f32", "f16", "i16", "q16x", "q8vnni"}:
        raise ValueError(f"unsupported precision {precision!r}")
    return weights, np.float32(item["coef"])


def check_package(
    package: Path, row_count: int, seed: int, total_rows: int | None
) -> dict[str, object]:
    """Compare native streaming features and predictions to the canonical references."""
    from geometry_serving_reference import GeometryReferenceStream

    event_stream_type, event_reference, geometry_reference = _reference_modules()
    solution = _load_solution(package)
    if not hasattr(solution.wnn_native, "GeometryFeatureStepper"):
        raise RuntimeError("native library lacks GeometryFeatureStepper; build the task branch")
    prior_value = json.loads((package / "geometry_prior.json").read_text(encoding="utf-8"))
    prior = solution._load_prior(package / "geometry_prior.json")
    prior_f32 = tuple(np.asarray(value, dtype=np.float32) for value in prior)
    raw = _synthetic_rows(row_count, seed)

    batch_events = event_reference(raw)
    batch_geometry = geometry_reference(raw, prior_value)
    event_stream = event_stream_type()
    geometry_stream = GeometryReferenceStream(prior_value)
    native_features = solution.wnn_native.GeometryFeatureStepper(*prior_f32)
    streamed_events = np.empty_like(batch_events)
    streamed_geometry = np.empty_like(batch_geometry)
    feature_started = time.perf_counter_ns()
    for index, row in enumerate(raw):
        expected_event = event_stream.update(row)
        expected_geometry = geometry_stream.update(row)
        observed = native_features.step(row)
        streamed_events[index] = observed[:99]
        streamed_geometry[index] = observed[99:]
        if not np.array_equal(streamed_events[index], expected_event):
            raise AssertionError(
                f"native event features differ from streaming reference at row {index}"
            )
        if not np.array_equal(streamed_geometry[index], expected_geometry):
            raise AssertionError(
                f"native geometry features differ from streaming reference at row {index}"
            )
    feature_seconds = (time.perf_counter_ns() - feature_started) / 1e9
    event_equal = bool(np.array_equal(streamed_events, batch_events))
    geometry_equal = bool(np.array_equal(streamed_geometry, batch_geometry))
    if not event_equal or not geometry_equal:
        raise AssertionError("native streaming features differ from canonical batch features")

    engineered = np.ascontiguousarray(np.concatenate((raw, batch_events, batch_geometry), axis=1))
    precision_path = package / "precision.txt"
    precision = (
        precision_path.read_text(encoding="utf-8").strip() if precision_path.is_file() else "f32"
    )
    weights, coefficient = _load_member(package, precision)
    expected_model = solution.wnn_native.ModelStepper(
        weights, fast_activations=True, precision=precision
    )
    expected = expected_model.run(engineered)[:, :2] * coefficient

    callback = solution.PredictionModel()
    point = SimpleNamespace(seq_ix=seed, state=None, need_prediction=False)
    need = np.arange(row_count) >= min(row_count, 99)
    for hole in (1_777, 7_011, 14_003):
        if hole < row_count:
            need[hole] = False
    observed_predictions = np.zeros_like(expected)
    callback_started = time.perf_counter_ns()
    for index, row in enumerate(raw):
        point.state = row
        point.need_prediction = bool(need[index])
        value = callback.predict(point)
        if need[index]:
            if value is None:
                raise AssertionError(f"callback omitted requested row {index}")
            observed_predictions[index] = value
        elif value is not None:
            raise AssertionError(f"callback returned a masked row {index}")
    callback_seconds = (time.perf_counter_ns() - callback_started) / 1e9
    prediction_equal = bool(np.array_equal(observed_predictions[need], expected[need]))
    max_error = (
        float(np.max(np.abs(observed_predictions[need] - expected[need]))) if need.any() else 0.0
    )
    if not prediction_equal:
        raise AssertionError(
            f"native callback output differs from expected output; max_abs={max_error}"
        )

    point.seq_ix = seed + 1
    for index, row in enumerate(raw[: min(32, row_count)]):
        point.state = row
        point.need_prediction = True
        value = callback.predict(point)
        if value is None or not np.array_equal(value, expected[index]):
            raise AssertionError(f"sequence reset parity failed at prefix row {index}")

    seconds_per_row = callback_seconds / row_count
    return {
        "status": "pass",
        "synthetic": True,
        "rows": row_count,
        "seed": seed,
        "precision": precision,
        "feature_width": int(engineered.shape[1]),
        "native_event_features_bitwise_equal": event_equal,
        "native_geometry_features_bitwise_equal": geometry_equal,
        "native_predictions_bitwise_equal": prediction_equal,
        "native_max_abs_error": max_error,
        "reference_feature_stream_seconds": feature_seconds,
        "callback_seconds_20k_proxy": callback_seconds if row_count == 20_000 else None,
        "callback_seconds_per_row": seconds_per_row,
        "need_prediction_rows": int(need.sum()),
        "row_cost_runtime_estimate_seconds": (
            seconds_per_row * total_rows if total_rows is not None else None
        ),
        "runtime_estimate_label": "row-cost extrapolation only; complete competition runtime not measured",
    }


def main() -> None:
    """Run the bounded synthetic native parity and callback profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--total-rows", type=int)
    args = parser.parse_args()
    if not 1 <= args.rows <= 20_000:
        raise SystemExit("--rows must be between 1 and 20000")
    if args.total_rows is not None and args.total_rows < args.rows:
        raise SystemExit("--total-rows must be at least --rows")
    result = check_package(
        args.package_root.resolve(strict=True), args.rows, args.seed, args.total_rows
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
