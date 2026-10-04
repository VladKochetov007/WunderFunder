"""Locate the fixed q16x error gate failure without changing any gate or prediction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.gmetric import METRIC_CLIP


def main() -> None:
    base = Path("research/campaign_2026-10-04/checks")
    names = ("e412_package_q16x_v3.npz", "e412_package_q16x_v2.npz", "e412_package_f32_v3.npz")
    arrays = []
    hashes = {}
    for name in names:
        with (base / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
        with np.load(base / name, allow_pickle=False) as archive:
            arrays.append({key: archive[key] for key in archive.files})
    q16, v2, f32 = arrays
    assert np.array_equal(q16["pred"], v2["pred"])
    assert all(np.array_equal(q16[key], f32[key]) for key in ("y", "mask"))
    required = np.broadcast_to(np.arange(20000) >= 99, q16["mask"].shape)
    raw = np.abs(q16["pred"] - f32["pred"])
    clipped = np.abs(np.clip(q16["pred"], -METRIC_CLIP, METRIC_CLIP) - np.clip(f32["pred"], -METRIC_CLIP, METRIC_CLIP))
    subset = pq.ParquetFile("cache/valid_subset20.parquet")
    sequence_ids = [int(subset.read_row_group(group, columns=["seq_ix"]).column(0)[0].as_py()) for group in range(subset.num_row_groups)]
    assert len(sequence_ids) == len(raw) == 20
    locations = np.argwhere(required[..., None] & (raw > 3e-4))
    records = []
    for seq, step, target in locations:
        records.append({"subset_index": int(seq), "sequence_id": sequence_ids[seq], "step": int(step), "target_index": int(target),
                        "q16x": float(q16["pred"][seq, step, target]), "native_f32": float(f32["pred"][seq, step, target]),
                        "raw_abs_error": float(raw[seq, step, target]), "clipped_abs_error": float(clipped[seq, step, target]),
                        "scored_row": bool(q16["mask"][seq, step]), "target_value": float(q16["y"][seq, step, target])})
    records.sort(key=lambda row: row["raw_abs_error"], reverse=True)
    report = {"input_sha256": hashes, "fixed_gate_max_abs": 3e-4, "raw_gate_still_failed": bool(records), "metric_clip": METRIC_CLIP,
              "required_entries": int(required.sum() * 2), "raw_exceedance_count": len(records),
              "required_raw_quantiles_50_99_999_100": np.quantile(raw[required], [0.5, 0.99, 0.999, 1.0]).tolist(),
              "required_clipped_max_abs": float(clipped[required].max()),
              "scored_raw_max_abs": float(raw[q16["mask"]].max()),
              "scored_clipped_max_abs": float(clipped[q16["mask"]].max()),
              "all_raw_gate_exceedances": records,
              "scope": "Saved native arrays only; clipping is descriptive localization, not a relaxed gate. Independent Torch FP32 reference remains necessary."}
    with (base / "e412_q16x_localization.json").open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "all_raw_gate_exceedances"}, indent=2))
    print(json.dumps(records[:5], indent=2))


if __name__ == "__main__":
    main()
