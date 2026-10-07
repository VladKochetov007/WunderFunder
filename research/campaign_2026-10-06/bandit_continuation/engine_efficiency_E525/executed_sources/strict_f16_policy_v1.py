"""Compare identical represented FP16 body weights, FP32 biases/state/tail."""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from benchmark_engines import digest, replay


def main() -> None:
    folder = Path(__file__).resolve().parent
    root = folder.parents[3]
    sys.path[:0] = [str(root), str(root / "submissions/c20261004_combo_s4_i16/pkg/v3"), str(folder / "cpp_reference_build")]
    import wnn_native
    import wnn_cpp
    from utils import FEATURE_COLUMNS

    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    data = root / "cache/valid_subset20.parquet"
    reader = pq.ParquetFile(data)
    rows = np.stack([np.column_stack([column.to_numpy() for column in reader.read_row_group(group, columns=FEATURE_COLUMNS, use_threads=False).columns]).astype(np.float32) for group in range(2)])
    weights = []
    hashes = {}
    for seed in (3, 4):
        source = root / f"submissions/c20261007_c1_s{seed}_i16/pkg/weights_0.npz"
        with np.load(source, allow_pickle=False) as archive:
            member = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
        folded = member["inp.weight"].astype(np.float64) / member["std"].astype(np.float64)
        member["inp.bias"] = (member["inp.bias"].astype(np.float64) - (folded * member["mean"]).sum(axis=1)).astype(np.float32)
        member["inp.weight"] = folded.astype(np.float32)
        member["mean"] = np.zeros_like(member["mean"])
        member["std"] = np.ones_like(member["std"])
        for key in member:
            if key == "inp.weight" or key.startswith("gru.weight_") or key == "head.0.weight":
                member[key] = member[key].astype(np.float16).astype(np.float32)
        path = folder / f"strict_fp16_represented_s{seed}.npz"
        np.savez(path, **member)
        hashes[str(seed)] = {"original_sha256": digest(source), "represented_sha256": digest(path)}
        weights.append(member)
    records = []
    for members in (1, 2):
        engines = {
            "rust_strict_f16": wnn_native.EnsemblePredictor(weights[:members], [1/members]*members, precision="f16", budget_s=None),
            "cpp_strict_f16": wnn_cpp.Ensemble(weights[:members], [1/members]*members, half_weights=True, dynamic_int8=False),
        }
        times = {name: [] for name in engines}
        predictions = {}
        for engine in engines.values():
            replay(engine, rows[:, :128])
        for trial in range(3):
            for name in list(engines)[::(-1 if trial % 2 else 1)]:
                started = time.perf_counter()
                output = replay(engines[name], rows)
                times[name].append((time.perf_counter() - started)*1e6/40000)
                if not np.isfinite(output).all() or (name in predictions and not np.array_equal(predictions[name], output)):
                    raise ValueError("Nonfinite or nondeterministic replay")
                predictions[name] = output
                print(json.dumps({"members": members, "case": name, "trial": trial, "us_per_row": times[name][-1]}), flush=True)
        gap = np.abs(predictions["rust_strict_f16"][:,99:] - predictions["cpp_strict_f16"][:,99:])
        for name in engines:
            records.append({"members":members, "case": name, "median_us_per_row": statistics.median(times[name]), "trials_us_per_row": times[name], "max_abs_rust_vs_cpp": float(gap.max()), "mean_abs_rust_vs_cpp": float(gap.mean()), "repeat_exact":True})
    report = {"scope": "Identical represented FP16 input/GRU/first-head matrices; FP32 bias/state/output-head. Both use float32 arithmetic. No training or public claim.", "weights":hashes, "source_sha256":digest(Path(__file__)), "data_sha256":digest(data), "results":records}
    (folder / "STRICT_FP16_POLICY_COMPARISON_V1.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
