from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _native_level(package: Path) -> str:
    """Select the packaged native library supported by this CPU."""
    flags: set[str] = set()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                break
    except OSError:
        pass
    v3 = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
    avx2 = {"avx", "avx2", "fma"}
    if flags >= v3 and (package / "v3").is_dir():
        return "v3"
    if flags >= avx2 and (package / "avx2").is_dir():
        return "avx2"
    return "v2"


def _load_solution(package: Path):
    """Import the solution module and native library from the extracted ZIP."""
    level = _native_level(package)
    sys.path.insert(0, str(package / level))
    spec = importlib.util.spec_from_file_location(
        "checked_geometry_solution", package / "solution.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("submission ZIP does not contain a loadable solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, level


def _raw_rows(seed: int, count: int) -> np.ndarray:
    """Create deterministic finite 112-column rows with discrete geometry levels."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(count, 112)).astype(np.float32)
    levels = np.asarray((-1.0, 0.0, 1.0, 2.0), dtype=np.float32)
    for column in (0, 3, 52, 55):
        indices = rng.integers(0, len(levels), size=count)
        hold = rng.random(count) < 0.75
        for index in range(1, count):
            if hold[index]:
                indices[index] = indices[index - 1]
        values[:, column] = levels[indices]
    return values


def _replay(model, sequence: int, values: np.ndarray, warmup: int) -> np.ndarray:
    """Replay one callback sequence and validate its warmup and finite outputs."""
    predictions: list[np.ndarray] = []
    for step, row in enumerate(values):
        needs_prediction = step >= warmup
        value = model.predict(
            SimpleNamespace(seq_ix=sequence, state=row, need_prediction=needs_prediction)
        )
        if not needs_prediction:
            if value is not None:
                raise AssertionError(f"unexpected warmup prediction at row {step}")
            continue
        prediction = np.asarray(value, dtype=np.float32)
        if prediction.shape != (2,) or not np.isfinite(prediction).all():
            raise AssertionError(f"invalid requested prediction at row {step}")
        predictions.append(prediction.copy())
    return np.stack(predictions)


def main() -> None:
    """Check exact-package callback reset determinism with an A-B-A replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("/pkg"))
    parser.add_argument("--rows", type=int, default=104)
    parser.add_argument("--warmup", type=int, default=99)
    args = parser.parse_args()
    if not 1 <= args.warmup < args.rows:
        raise SystemExit("require 1 <= warmup < rows")
    solution, level = _load_solution(args.package_root.resolve(strict=True))
    model = solution.PredictionModel()
    first = _raw_rows(20261018, args.rows)
    intervening = _raw_rows(20261019, args.rows)
    first_predictions = _replay(model, 41001, first, args.warmup)
    intervening_predictions = _replay(model, 41002, intervening, args.warmup)
    repeated_predictions = _replay(model, 41001, first, args.warmup)
    np.testing.assert_array_equal(first_predictions, repeated_predictions)
    result = {
        "passed": True,
        "native_level": level,
        "rows_per_sequence": args.rows,
        "warmup_rows": args.warmup,
        "predictions_per_sequence": args.rows - args.warmup,
        "all_predictions_finite": bool(
            np.isfinite(first_predictions).all()
            and np.isfinite(intervening_predictions).all()
            and np.isfinite(repeated_predictions).all()
        ),
        "aba_outputs_bitwise_equal": True,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
