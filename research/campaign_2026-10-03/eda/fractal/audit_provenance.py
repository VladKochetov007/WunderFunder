from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from continuation_probe import ROOT, moments, save_json, wp

OUT = Path(__file__).resolve().parent


def main() -> None:
    """Hash sampled raw arrays, screen exact boundary links and audit WP arithmetic."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    sample = json.loads((OUT / "sample_manifest.json").read_text())
    fingerprints = {}
    links = set()
    records = []
    for source in ("train", "valid"):
        path = ROOT / "datasets" / f"{source}.parquet"
        parquet = pq.ParquetFile(path)
        stat = path.stat()
        sample[f"{source}_file"]["mtime_ns"] = stat.st_mtime_ns
        features = (
            parquet.schema_arrow.names[3:115]
            if source == "train"
            else parquet.schema_arrow.names[4:116]
        )
        for record in sample["new_sample"][source]:
            columns = [*features, "t0", "t1", "need_prediction"]
            if source == "valid":
                columns.append("is_scored")
            table = parquet.read_row_group(record["row_group"], columns=columns, use_threads=False)
            values = np.column_stack([table[c].to_numpy() for c in features]).astype(np.float32)
            digest = hashlib.sha256()
            for column in columns:
                digest.update(column.encode())
                digest.update(np.ascontiguousarray(table[column].to_numpy()).tobytes())
            identity = f"{source}:{record['seq_ix']}"
            records.append(
                {
                    "source": source,
                    **record,
                    "columns": columns,
                    "sampled_array_sha256": digest.hexdigest(),
                }
            )
            # Boundary windows only: this screen cannot certify global chain separation.
            for start in [*range(241), *range(len(values) - 256, len(values) - 15)]:
                key = hashlib.sha256(values[start : start + 16].tobytes()).digest()
                if key in fingerprints and fingerprints[key] != identity:
                    links.add(tuple(sorted((fingerprints[key], identity))))
                else:
                    fingerprints[key] = identity
    save_json("sample_manifest.json", sample)
    rng = np.random.default_rng(7)
    target = rng.normal(size=1000) * 3
    prediction = rng.normal(size=1000) * 3 + target
    yy = np.clip(target, -2, 2)
    pp = np.clip(prediction, -2, 2)
    weight = np.abs(yy)
    yc = yy - np.average(yy, weights=weight)
    pc = pp - np.average(pp, weights=weight)
    direct = np.sum(weight * yc * pc) / np.sqrt(np.sum(weight * yc**2) * np.sum(weight * pc**2))
    split = moments(target[:317], prediction[:317]) + moments(target[317:], prediction[317:])
    error = abs(float(wp(split)) - direct)
    assert error < 1e-12
    save_json(
        "provenance_audit.json",
        {
            "sampled_arrays": records,
            "boundary_window_cross_sequence_links": sorted(links),
            "boundary_screen": "16 consecutive rows of all 112 features; every offset within first/last 256 rows; SHA256 exact bytes",
            "component_disjointness": "not certified; interior and unsampled campaign links not searched",
            "metric_sufficient_statistics_vs_direct_max_error": error,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
    )


if __name__ == "__main__":
    main()
