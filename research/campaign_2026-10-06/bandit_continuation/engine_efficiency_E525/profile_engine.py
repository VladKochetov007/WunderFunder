from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    root = args.root
    sys.path.insert(0, str(root))
    from utils import FEATURE_COLUMNS

    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    source = root / "submissions/c20261007_c1_s3_i16/pkg/weights_0.npz"
    with np.load(source, allow_pickle=False) as archive:
        weights = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if args.case.startswith("cpp_"):
        sys.path.insert(
            0,
            str(
                root / "research/campaign_2026-10-06/bandit_continuation/"
                "engine_efficiency_E525/cpp_reference_build"
            ),
        )
        import wnn_cpp

        backend = args.case.removeprefix("cpp_")
        engine = wnn_cpp.Ensemble([weights], [1.0], dynamic_int8=True, quant_backend=backend)
    else:
        precision = args.case.removeprefix("rust_")
        library = (
            root / "research/campaign_2026-10-04/E462_python311_repair_v1/"
            "q8x/pkg/v3/wnn_native.abi3.so"
            if precision.startswith("q8")
            else root / "submissions/c20261004_combo_s4_i16/pkg/v3/wnn_native.abi3.so"
        )
        spec = importlib.util.spec_from_file_location("wnn_native", library)
        if spec is None or spec.loader is None:
            raise ImportError(str(library))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        engine = module.EnsemblePredictor([weights], [1.0], precision=precision, budget_s=None)
    table = pq.ParquetFile(root / "cache/valid_subset20.parquet").read_row_group(
        0, columns=FEATURE_COLUMNS, use_threads=False
    )
    rows = np.column_stack([column.to_numpy() for column in table.columns]).astype(np.float32)
    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < args.seconds:
        engine.reset()
        for row in rows:
            value = engine.step(row)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite output")
        iterations += 1
    print({"case": args.case, "complete_sequences": iterations, "profile_only": True})


if __name__ == "__main__":
    main()
