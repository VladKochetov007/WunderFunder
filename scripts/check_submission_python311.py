from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, DataPoint, PredictionModel  # noqa: E402


def replay(
    model: PredictionModel, sequence: int, features: np.ndarray, need: np.ndarray
) -> np.ndarray:
    """Exercise the scorer's row callback, including no-prediction warmup rows."""
    predictions = np.zeros((len(features), 2), dtype=np.float32)
    for step, row in enumerate(features):
        value = model.predict(DataPoint(sequence, step, bool(need[step]), row))
        if not need[step]:
            if value is not None:
                raise ValueError(f"unexpected warmup prediction at row {step}")
            continue
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"invalid required prediction at row {step}")
        predictions[step] = value
    return predictions


def main() -> None:
    """Test actual ZIP bytes in Python 3.11; this is not a runtime benchmark."""
    parser = argparse.ArgumentParser(description="Offline Python 3.11 submission callback gate")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--level", choices=("v2", "avx2", "v3"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("run this gate inside the offline Python 3.11 container")
    if os.environ.get("LD_LIBRARY_PATH") or os.environ.get("PYTHONPATH"):
        raise RuntimeError("compatibility gate requires clean library and Python paths")
    os.environ["WNN_NATIVE_LEVEL"] = args.level
    parquet = pq.ParquetFile(args.data)
    streams = []
    for group in range(2):
        table = parquet.read_row_group(
            group, columns=["seq_ix", "need_prediction", *FEATURE_COLUMNS], use_threads=False
        )
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError("gate requires two complete 20,000-row sequences")
        features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(
            np.float32
        )
        streams.append(
            (int(table["seq_ix"][0].as_py()), features, table["need_prediction"].to_numpy())
        )
    with np.load(args.reference, allow_pickle=False) as reference:
        expected = [
            reference["pred"][i] if "pred" in reference else reference[f"pred_{i}"]
            for i in range(2)
        ]
    with tempfile.TemporaryDirectory(prefix="wnn-py311-gate-") as temporary:
        package = Path(temporary)
        with zipfile.ZipFile(args.zip) as archive:
            if archive.testzip() is not None:
                raise ValueError("corrupt submission ZIP")
            for member in archive.namelist():
                if Path(member).is_absolute() or ".." in Path(member).parts:
                    raise ValueError(f"unsafe ZIP path: {member}")
            archive.extractall(package)
        spec = importlib.util.spec_from_file_location("checked_solution", package / "solution.py")
        if spec is None or spec.loader is None:
            raise ImportError("submission lacks a loadable solution.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.PredictionModel()
        predictions = [replay(model, *stream) for stream in streams]
        maximum_difference = max(
            float(np.max(np.abs(actual - want)))
            for actual, want in zip(predictions, expected, strict=True)
        )
        # This checks a packaging-only rebuild against archived native outputs.
        # It is deliberately stricter than the historical Torch parity threshold.
        if maximum_difference > 1e-6:
            raise ValueError(f"packaging rebuild changed predictions: {maximum_difference}")
        first_sequence, first_features, first_need = streams[0]
        reset = replay(model, first_sequence, first_features[:120], first_need[:120])
        np.testing.assert_array_equal(reset, predictions[0][:120])
        import wnn_native

        native_path = Path(wnn_native.__file__)
        if native_path.parent != package / args.level:
            raise RuntimeError(f"unexpected imported native library: {native_path}")
        report = {
            "passed": True,
            "zip_sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest(),
            "python": sys.version,
            "numpy": np.__version__,
            "level": args.level,
            "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "rows": 2 * SEQUENCE_LENGTH,
            "reset_prefix_rows": 120,
            "max_abs_vs_archived_native": maximum_difference,
            "scope": "offline import, constructor, full-stream callback, reset and archived-output equivalence; no runtime or public-score approval",
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
