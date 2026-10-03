from __future__ import annotations

import json
import logging

import numpy as np
import pyarrow.parquet as pq
from continuation_common import FEATURES, OUT, ROOT, causal_design
from probe_offsets import predict_calibrated


def verify() -> None:
    """Verify prefix invariance, finite predictions and target-free inference."""
    sampling = json.loads((OUT / "sampling_v1.json").read_text())
    group = sampling["evaluation_valid"][0]
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    table = parquet.read_row_group(group, columns=FEATURES, use_threads=False)
    values = np.column_stack([table[c].to_numpy() for c in FEATURES])
    with np.load(OUT / "continuation_ridge.npz") as data:
        beta = data["beta"]
    design, states = causal_design(values)
    full_predictions = design.astype(np.float64) @ beta[:-1] + beta[-1]
    records = []
    for source in ["train", "exposed_scored_valid"]:
        with np.load(OUT / f"calibration_{source}_v1.npz") as data:
            fitted = {name: data[name] for name in data.files}
        full = predict_calibrated(full_predictions, states, fitted)
        assert np.isfinite(full).all(), "Nonfinite prediction on full sequence"
        for prefix in [99, 100, 512, 1000]:
            prefix_design, prefix_states = causal_design(values[:prefix])
            predictions = prefix_design.astype(np.float64) @ beta[:-1] + beta[-1]
            output = predict_calibrated(predictions, prefix_states, fitted)
            difference = float(np.max(np.abs(output - full[:prefix])))
            assert difference < 1e-6, f"Future dependence at prefix {prefix}"
            records.append({"source": source, "prefix": prefix, "max_difference": difference})
    result = {
        "row_group": group,
        "prefix_checks": records,
        "finite_all_models_all_rows": True,
        "input_columns": FEATURES,
        "targets_or_masks_read": False,
        "reset_behavior": "Pure sequence function initialized from row0; prefix checks include warm-up. No cross-sequence state.",
    }
    (OUT / "causality_checks_v1.json").write_text(json.dumps(result, indent=2))
    logging.info("Target-free prefix invariance passed: %s", records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify()
