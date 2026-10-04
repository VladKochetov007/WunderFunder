"""Run a submission callback, stream official WP, and preserve all predictions.

    python scripts/verify_submission.py SUBMISSION.zip DATA.parquet OUT.npz

NPZ output contains every sequence's ``pred``, ``y`` and ``mask`` arrays. NPY
output stores predictions plus adjacent ``.targets.npy`` and ``.mask.npy`` files.
Arrays are backed by temporary disk mappings while scoring, so memory does not
grow with the number of sequences. Legacy block WP is an optional diagnostic.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import (  # noqa: E402
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    BlockAccumulator,
    DataPoint,
    GlobalAccumulator,
    validate_sequence,
)

_LOG = logging.getLogger(__name__)


def global_wp(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> list[float]:
    """Return both official target scores for compatibility with older callers."""
    accumulator = GlobalAccumulator()
    accumulator.add(y, p, m)
    result = accumulator.result()
    return [float(result["t0"]), float(result["t1"])]


def verify_submission(zpath: Path, data: Path, out: Path) -> dict[str, float | int | str | None]:
    """Verify all required outputs and stream a complete dataset to disk artifacts."""
    if out.suffix not in {".npz", ".npy"}:
        raise ValueError("prediction output must end in .npz or .npy")
    pf = pq.ParquetFile(data)
    if pf.num_row_groups == 0:
        raise ValueError("no selected rows")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        with zipfile.ZipFile(zpath) as archive:
            archive.extractall(tmp)
        sys.path.insert(0, str(tmp))
        try:
            sys.modules.pop("solution", None)
            t0 = time.perf_counter()
            model = importlib.import_module("solution").PredictionModel()
            load_s = time.perf_counter() - t0
            shape = (pf.num_row_groups, SEQUENCE_LENGTH, 2)
            predictions = np.lib.format.open_memmap(
                tmp / "pred.npy", mode="w+", dtype=np.float32, shape=shape
            )
            targets = np.lib.format.open_memmap(
                tmp / "y.npy", mode="w+", dtype=np.float32, shape=shape
            )
            masks = np.lib.format.open_memmap(
                tmp / "mask.npy", mode="w+", dtype=bool, shape=shape[:2]
            )
            accumulator = GlobalAccumulator()
            block_accumulator = BlockAccumulator()
            diagnostic_error = None
            rows, busy = 0, 0.0
            seen = set()
            for group in range(pf.num_row_groups):
                table = pf.read_row_group(group, use_threads=False)
                seq, need = validate_sequence(table)
                if seq in seen:
                    raise ValueError("duplicate sequence ID")
                seen.add(seq)
                features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(
                    np.float32
                )
                y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(
                    np.float32
                )
                p = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
                started = time.perf_counter()
                for step in range(SEQUENCE_LENGTH):
                    value = model.predict(DataPoint(seq, step, bool(need[step]), features[step]))
                    if need[step]:
                        value = np.asarray(value, dtype=np.float32)
                        if value.shape != (2,) or not np.isfinite(value).all():
                            raise ValueError("return two finite scores on every required row")
                        p[step] = value
                    elif value is not None:
                        raise ValueError("warm-up must return None after updating state")
                busy += time.perf_counter() - started
                rows += SEQUENCE_LENGTH
                mask = table["is_scored"].to_numpy().astype(bool) & need
                accumulator.add(y, p, mask)
                if diagnostic_error is None:
                    try:
                        block_accumulator.add(y, p, mask)
                    except ValueError as error:
                        diagnostic_error = str(error)
                predictions[group], targets[group], masks[group] = p, y, mask
            official = accumulator.result()
            block_wp = None
            if diagnostic_error is None:
                if block_accumulator.scores:
                    block_wp = float(block_accumulator.result()["weighted_pearson"])
                else:
                    diagnostic_error = "no eligible blocks"
            for array in (predictions, targets, masks):
                array.flush()
            if out.suffix == ".npz":
                # NumPy writes contiguous memmaps in chunks without stacking arrays.
                np.savez(out, pred=predictions, y=targets, mask=masks)
            else:
                shutil.copyfile(tmp / "pred.npy", out)
                shutil.copyfile(tmp / "y.npy", out.with_suffix(".targets.npy"))
                shutil.copyfile(tmp / "mask.npy", out.with_suffix(".mask.npy"))
            return {
                "metric_id": str(official["metric_id"]),
                "load_s": round(load_s, 2),
                "us_per_row": round(1e6 * busy / rows, 1),
                "rows": rows,
                "sequences": len(seen),
                "selected_rows": int(official["selected_rows"]),
                "block_wp": None if block_wp is None else round(block_wp, 6),
                "block_diagnostic_error": diagnostic_error,
                "global_wp": round(float(official["weighted_pearson"]), 6),
                "global_t0": round(float(official["t0"]), 6),
                "global_t1": round(float(official["t1"]), 6),
                "eta_min_39M_rows": round(39.4e6 * busy / rows / 60, 1),
            }
        finally:
            sys.path.remove(str(tmp))
            sys.modules.pop("solution", None)


def main() -> None:
    """Run the submission verification CLI and log its JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    _LOG.info(
        json.dumps(
            verify_submission(arguments.submission, arguments.data, arguments.output),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
