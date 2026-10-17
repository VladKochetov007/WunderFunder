"""Check and profile the standalone 271-channel geometry submission callback."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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


def _reference_modules() -> tuple[Callable[[np.ndarray], np.ndarray], Callable[..., np.ndarray]]:
    """Load canonical NumPy feature references without importing the Torch runtime."""
    # The online module postpones annotations; this stub is sufficient for its NumPy path.
    sys.modules["torch"] = ModuleType("torch")
    from wnn.features import event_features
    from wnn.moment_geometry.online import features_numpy

    return event_features, features_numpy


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


def _member_inputs(package: Path) -> tuple[list[tuple[dict[str, np.ndarray], np.float32]], str]:
    """Load fixed package members and the declared serving precision."""
    spec = cast(list[dict[str, object]], json.loads((package / "members.json").read_text()))
    members: list[tuple[dict[str, np.ndarray], np.float32]] = []
    for item in spec:
        with np.load(package / str(item["weights"]), allow_pickle=False) as archive:
            weights = {
                name: np.ascontiguousarray(archive[name], dtype=np.float32)
                for name in archive.files
            }
        members.append((weights, np.float32(item["coef"])))
    precision_path = package / "precision.txt"
    precision = (
        precision_path.read_text(encoding="utf-8").strip() if precision_path.is_file() else "f32"
    )
    return members, precision


def check_package(
    package: Path, row_count: int, seed: int, total_rows: int | None
) -> dict[str, object]:
    """Compare both streaming feature channels and outputs against canonical references."""
    event_reference, geometry_reference = _reference_modules()
    solution = _load_solution(package)
    prior = solution._load_prior(package / "geometry_prior.json")
    raw = _synthetic_rows(row_count, seed)

    reference_events = event_reference(raw)
    reference_geometry = geometry_reference(
        raw, json.loads((package / "geometry_prior.json").read_text())
    )
    event_stream = solution._EventStream()
    geometry_stream = solution._GeometryStream(prior)
    streamed_events = np.empty_like(reference_events)
    streamed_geometry = np.empty_like(reference_geometry)
    feature_started = time.perf_counter_ns()
    for index, row in enumerate(raw):
        streamed_events[index] = event_stream.update(row)
        streamed_geometry[index] = geometry_stream.update(row)
    feature_seconds = (time.perf_counter_ns() - feature_started) / 1e9
    event_equal = bool(np.array_equal(streamed_events, reference_events))
    geometry_equal = bool(np.array_equal(streamed_geometry, reference_geometry))
    if not event_equal or not geometry_equal:
        raise AssertionError(
            "feature parity failed: "
            f"event_max_abs={float(np.max(np.abs(streamed_events - reference_events)))} "
            f"geometry_max_abs={float(np.max(np.abs(streamed_geometry - reference_geometry)))}"
        )

    engineered = np.ascontiguousarray(
        np.concatenate((raw, reference_events, reference_geometry), axis=1), dtype=np.float32
    )
    members, precision = _member_inputs(package)
    if not members:
        raise ValueError("members.json must contain at least one model")
    expected = np.zeros((row_count, 2), dtype=np.float32)
    native_models = []
    for weights, coefficient in members:
        native = solution.wnn_native.ModelStepper(
            weights, fast_activations=True, precision=precision
        )
        native_models.append(native)
        expected += coefficient * native.run(engineered)[:, :2]

    native_row_models = [
        solution.wnn_native.ModelStepper(weights, fast_activations=True, precision=precision)
        for weights, _ in members
    ]
    native_row = np.empty((1, engineered.shape[1]), dtype=np.float32)
    native_started = time.perf_counter_ns()
    for index in range(row_count):
        native_row[0] = engineered[index]
        for model in native_row_models:
            model.run(native_row)
    native_row_seconds = (time.perf_counter_ns() - native_started) / 1e9

    callback = solution.PredictionModel()
    point = SimpleNamespace(seq_ix=seed, state=None, need_prediction=False)
    need = np.arange(row_count) >= min(row_count, 99)
    for hole in (1_777, 7_011, 14_003):
        if hole < row_count:
            need[hole] = False
    observed = np.zeros_like(expected)
    started = time.perf_counter_ns()
    for index, row in enumerate(raw):
        point.state = row
        point.need_prediction = bool(need[index])
        value = callback.predict(point)
        if need[index]:
            if value is None:
                raise AssertionError(f"callback omitted requested row {index}")
            observed[index] = value
        elif value is not None:
            raise AssertionError(f"callback returned a masked row {index}")
    callback_seconds = (time.perf_counter_ns() - started) / 1e9
    predictions_equal = bool(np.array_equal(observed[need], expected[need]))
    max_prediction_error = (
        float(np.max(np.abs(observed[need] - expected[need]))) if need.any() else 0.0
    )
    if not predictions_equal:
        raise AssertionError(
            f"callback/native sequence parity failed; max_abs={max_prediction_error}"
        )

    # Changing sequence IDs must reset both feature clocks and every recurrent member.
    point.seq_ix = seed + 1
    reset_need = np.ones(min(32, row_count), dtype=np.bool_)
    for index, row in enumerate(raw[: len(reset_need)]):
        point.state = row
        point.need_prediction = bool(reset_need[index])
        value = callback.predict(point)
        if value is None or not np.array_equal(value, expected[index]):
            raise AssertionError(f"sequence reset parity failed at prefix row {index}")

    seconds_per_row = callback_seconds / row_count
    result: dict[str, object] = {
        "status": "pass",
        "synthetic": True,
        "rows": row_count,
        "seed": seed,
        "precision": precision,
        "callback_feature_path": callback.feature_path,
        "callback_model_path": callback.model_path,
        "feature_width": int(engineered.shape[1]),
        "event_features_bitwise_equal": event_equal,
        "geometry_features_bitwise_equal": geometry_equal,
        "native_predictions_bitwise_equal": predictions_equal,
        "native_max_abs_error": max_prediction_error,
        "feature_stream_seconds": feature_seconds,
        "native_one_row_stepper_seconds": native_row_seconds,
        "callback_seconds_20k_proxy": callback_seconds if row_count == 20_000 else None,
        "callback_seconds_per_row": seconds_per_row,
        "feature_stream_seconds_per_row": feature_seconds / row_count,
        "native_one_row_seconds_per_member_row": native_row_seconds / (row_count * len(members)),
        "need_prediction_rows": int(need.sum()),
        "full_dataset_runtime_estimate_seconds": (
            seconds_per_row * total_rows if total_rows is not None else None
        ),
        "runtime_estimate_label": "row-cost extrapolation only; complete competition runtime not measured",
    }
    return result


def main() -> None:
    """Run the bounded synthetic feature parity and callback profile."""
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
