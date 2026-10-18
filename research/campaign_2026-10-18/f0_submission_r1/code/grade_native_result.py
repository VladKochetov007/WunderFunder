"""Independently regrade completed native predictions with the canonical metric."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.gmetric import GlobalAccumulator


def digest(path: Path) -> str:
    """Hash evidence with bounded file reads."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Cross-check original labels, mask, full replay hashes and pooled score."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    work = args.base / "submission-f0-r1"
    report = json.loads((work / "Q8_FULL_F0.json").read_text())
    if report["status"] != "passed_panel_check":
        raise ValueError("native replay did not complete")
    pred_path = work / "q8_full_f0.npy"
    if digest(pred_path) != report["predictions_sha256"]:
        raise ValueError("native prediction hash mismatch")
    fold_path = work / "evidence/fold0.json"
    if digest(fold_path) != report["fold_sha256"]:
        raise ValueError("fold hash mismatch")
    groups = json.loads(fold_path.read_text())["evaluation"]["valid"]
    if report["selected_source_row_groups"] != groups or report["rows"] != 9360000:
        raise ValueError("native replay omitted or reordered groups")
    prediction = np.load(pred_path, mmap_mode="r")
    if prediction.dtype != np.float32 or prediction.shape != (len(groups), 20000, 2):
        raise ValueError("native prediction shape/dtype mismatch")
    source_path = (args.base / "data/datasets/valid.parquet").resolve()
    source_stat = source_path.stat()
    if str(source_path) != report["panel"]:
        raise ValueError("parquet source path differs from replay")
    if source_stat.st_size != report["panel_bytes"]:
        raise ValueError("parquet source size differs from replay")
    if source_stat.st_mtime_ns != report["panel_mtime_ns_after"]:
        raise ValueError("parquet source modification time differs from replay")
    source_hash = digest(source_path)
    if source_hash != report["panel_sha256"]:
        raise ValueError("parquet source hash differs from replay")
    source = pq.ParquetFile(source_path)
    accumulated = GlobalAccumulator()
    selected: list[np.ndarray] = []
    for index, group in enumerate(groups):
        table = source.read_row_group(
            group, columns=["t0", "t1", "need_prediction", "is_scored"], use_threads=False
        )
        target = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(
            np.float32
        )
        mask = table["need_prediction"].to_numpy().astype(bool) & table[
            "is_scored"
        ].to_numpy().astype(bool)
        accumulated.add(target, prediction[index], mask)
        selected.append(np.asarray(prediction[index][mask]))
    score = accumulated.result()
    source_stat_after = source_path.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ino) != (
        source_stat_after.st_size,
        source_stat_after.st_mtime_ns,
        source_stat_after.st_ino,
    ):
        raise ValueError("parquet changed during independent grade")
    for key in ("t0", "t1", "wp"):
        if abs(score[key] - report["official_score"][key]) > 1e-12:
            raise ValueError(f"independent canonical metric differs for {key}")
    if score["selected_rows"] != 808863:
        raise ValueError("wrong official row count")
    compact_path = work / "q8_scored.npy"
    if compact_path.exists():
        raise FileExistsError(compact_path)
    np.save(compact_path, np.concatenate(selected))
    result = {
        "status": "passed",
        "score": score,
        "groups": len(groups),
        "zip_sha256": report["zip_sha256"],
        "parquet_sha256": source_hash,
        "prediction_sha256": report["predictions_sha256"],
        "native_replay_receipt_sha256": digest(work / "Q8_FULL_F0.json"),
        "scored_prediction_sha256": digest(compact_path),
        "grader_sha256": digest(Path(__file__)),
        "canonical_metric_sha256": digest(Path(__import__("wnn.gmetric", fromlist=["x"]).__file__)),
        "scope": "Complete actual native callback predictions on exposed F0; no public score forecast.",
    }
    with (work / "Q8_CANONICAL_GRADE.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
