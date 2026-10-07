from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(engine: object, rows: np.ndarray) -> np.ndarray:
    """Use the identical row-at-a-time API, resetting each complete sequence."""
    output = np.empty((*rows.shape[:2], 2), dtype=np.float32)
    for group, sequence in enumerate(rows):
        engine.reset()
        for step, row in enumerate(sequence):
            output[group, step] = engine.step(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--quant-only", action="store_true")
    args = parser.parse_args()
    root = args.root
    sys.path.insert(0, str(root))
    from utils import FEATURE_COLUMNS

    folder = root / "research/campaign_2026-10-06/bandit_continuation"
    rust = root / "submissions/c20261004_combo_s4_i16/pkg/v3"
    if args.quant_only:
        rust = root / "research/campaign_2026-10-04/E462_python311_repair_v1/q8x/pkg/v3"
    cpp = folder / "engine_efficiency_E525/cpp_reference_build"
    sys.path[:0] = [str(rust), str(cpp)]
    import wnn_cpp
    import wnn_native

    affinity = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity})
    data = root / "cache/valid_subset20.parquet"
    parquet = pq.ParquetFile(data)
    rows = np.stack(
        [
            np.column_stack(
                [
                    column.to_numpy()
                    for column in parquet.read_row_group(
                        group, columns=FEATURE_COLUMNS, use_threads=False
                    ).columns
                ]
            ).astype(np.float32)
            for group in range(2)
        ]
    )
    weight_paths = [
        root / f"submissions/c20261007_c1_s{seed}_i16/pkg/weights_0.npz" for seed in (3, 4)
    ]
    weights = []
    for path in weight_paths:
        with np.load(path, allow_pickle=False) as archive:
            weights.append({key: np.ascontiguousarray(archive[key]) for key in archive.files})
    with np.load(folder / "submission_C1_E524/torch_cpu_reference_first2.npz") as ref:
        reference = ref["pred"]
    results: list[dict[str, object]] = []
    for members in (1, 2):
        coefficients = [1.0 / members] * members
        cases = {}
        for precision in ("q8r", "q8x") if args.quant_only else ("f32", "f16", "i16"):
            cases[f"rust_{precision}"] = wnn_native.EnsemblePredictor(
                weights[:members], coefficients, precision=precision, budget_s=None
            )
        for precision in ("auto", "avx2", "scalar") if args.quant_only else ("f32", "f16", "int8"):
            cases[f"cpp_{precision}"] = wnn_cpp.Ensemble(
                weights[:members],
                coefficients,
                fast_activations=True,
                half_weights=precision == "f16",
                dynamic_int8=args.quant_only or precision == "int8",
                quant_backend=precision if args.quant_only else "auto",
            )
        measurements = {name: [] for name in cases}
        outputs = {}
        for engine in cases.values():
            replay(engine, rows[:, :128])
        names = list(cases)
        for trial in range(args.trials):
            order = names[trial:] + names[:trial]
            if trial % 2:
                order.reverse()
            for name in order:
                start = time.perf_counter()
                prediction = replay(cases[name], rows)
                elapsed = time.perf_counter() - start
                measurements[name].append(elapsed * 1e6 / 40000)
                if name in outputs and not np.array_equal(outputs[name], prediction):
                    raise ValueError(f"nondeterministic full replay: {name}")
                outputs[name] = prediction
                print(
                    json.dumps(
                        {
                            "members": members,
                            "trial": trial,
                            "case": name,
                            "us_per_row": measurements[name][-1],
                        }
                    ),
                    flush=True,
                )
        target = reference[:members].mean(axis=0)
        for name, prediction in outputs.items():
            if not np.isfinite(prediction).all():
                raise ValueError(f"nonfinite predictions: {name}")
            error = np.abs(prediction[:, 99:] - target[:, 99:])
            results.append(
                {
                    "members": members,
                    "case": name,
                    "trials_us_per_row": measurements[name],
                    "median_us_per_row": statistics.median(measurements[name]),
                    "minimum_us_per_row": min(measurements[name]),
                    "max_abs_vs_cpu_float32": float(error.max()),
                    "mean_abs_vs_cpu_float32": float(error.mean()),
                    "full_repeat_exact": True,
                    "f32_parity_pass": bool(error.max() <= 0.0003)
                    if name.endswith("_f32")
                    else None,
                }
            )
        np.savez_compressed(args.output.parent / f"predictions_members{members}.npz", **outputs)
    report = {
        "scope": "matched own weights/raw streams; native row step API, not full scorer callback; no training/public claim",
        "cpu_affinity": affinity,
        "rows_per_trial": 40000,
        "complete_sequences": 2,
        "trials": args.trials,
        "weights_sha256": {str(p): digest(p) for p in weight_paths},
        "data_sha256": digest(data),
        "benchmark_source_sha256": digest(Path(__file__)),
        "rust_library_sha256": digest(Path(wnn_native.__file__)),
        "cpp_library_sha256": digest(Path(wnn_cpp.__file__)),
        "cpp_quant_backend": wnn_cpp.quant_backend(),
        "python": sys.version,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
