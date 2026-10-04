"""Verify the fixed E412 i16 package against both CPU builds and exported model rows."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils import DataPoint, FEATURE_COLUMNS  # noqa: E402
from wnn.gmetric import reference_global_wp


def main() -> None:
    root = Path.cwd()
    base = root / "research/campaign_2026-10-04/checks"
    output = base / "e412_i16_package_verification_v2.json"
    if output.exists():
        raise FileExistsError(output)
    arrays = {}
    for key in ("i16_v3", "i16_v2", "f32_v3"):
        with np.load(base / f"e412_package_{key}.npz", allow_pickle=False) as stored:
            arrays[key] = {name: stored[name] for name in stored.files}
    native, portable, floating = (arrays[key] for key in ("i16_v3", "i16_v2", "f32_v3"))
    if any(not np.array_equal(native[key], other[key]) for other in (portable, floating)
           for key in ("y", "mask")):
        raise ValueError("Callback runs have different targets or masks")
    if not np.array_equal(native["pred"], portable["pred"]):
        raise ValueError("CPU levels differ")
    required = np.broadcast_to(np.arange(20000) >= 99, native["mask"].shape)
    error = np.abs(native["pred"][required] - floating["pred"][required])
    score = reference_global_wp(native["y"], native["pred"], native["mask"])
    float_score = reference_global_wp(native["y"], floating["pred"], native["mask"])
    precision_gate = bool(error.max() <= 3e-4 and abs(score - float_score) <= 5e-6)

    subset = pq.ParquetFile(root / "cache/valid_subset20.parquet")
    with np.load(root / "cache/valid_targets.npz", allow_pickle=False) as target_cache:
        positions = {int(seq): index for index, seq in enumerate(target_cache["seq_ix"])}
    runs = ("c20261004_combo_s3_final_r1", "c20261004_combo_head_s3_final")
    cached = [np.load(root / f"cache/pred_{run}_valid.npy", mmap_mode="r") for run in runs]
    batch = []
    for group in range(subset.num_row_groups):
        seq = int(subset.read_row_group(group, columns=["seq_ix"]).column(0)[0].as_py())
        batch.append(sum(np.asarray(values[positions[seq]], np.float64) for values in cached) / 2)
    batch = np.stack(batch)
    batch_error = np.abs(native["pred"][required] - batch[required])

    package = root / "submissions/c20261004_combo_s3_i16"
    spec = importlib.util.spec_from_file_location("e412_callback", package / "pkg/solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PredictionModel()
    if str(model.model.precision) != "i16":
        raise ValueError("Callback must use the package i16 arithmetic")
    first = subset.read_row_group(0)
    features = np.column_stack([first[name].to_numpy() for name in FEATURE_COLUMNS]).astype(np.float32)
    repeated = []
    for sequence_id in (1, 2, 1):
        prediction = np.zeros((20000, 2), np.float32)
        for row in range(20000):
            value = model.predict(DataPoint(sequence_id, row, row >= 99, features[row]))
            if row >= 99:
                prediction[row] = value
            elif value is not None:
                raise ValueError("Warmup callback returned a value")
        repeated.append(prediction)
    reset_exact = all(np.array_equal(value, native["pred"][0]) for value in repeated)
    result = {
        "package_sha256": hashlib.sha256((package / "submission.zip").read_bytes()).hexdigest(),
        "package_bytes": (package / "submission.zip").stat().st_size,
        "sequences": len(native["pred"]),
        "v2_v3_bitwise": True,
        "sequence_reset_bitwise": reset_exact,
        "callback_precision": str(model.model.precision),
        "i16_vs_native_f32": {"max_abs": float(error.max()), "wp_delta": score - float_score,
                               "max_abs_limit": 3e-4, "wp_delta_limit": 5e-6, "passed": precision_gate},
        "gpu_fp16_export_diagnostic": {"max_abs": float(batch_error.max()),
            "p999_abs": float(np.quantile(batch_error, .999)),
            "wp_delta": score - reference_global_wp(native["y"], batch, native["mask"])},
        "scope": "20 full validation streams; fixed predictor; verification subset is not score confirmation",
        "passed": precision_gate and reset_exact,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise ValueError("Package verification gate failed")


if __name__ == "__main__":
    main()
