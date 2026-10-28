"""Grade the fixed fresh TL3 endpoint on every official validation sequence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from wnn.gmetric import GlobalAccumulator
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.chronological_teacher import validate_chronological_teacher_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tl3_chronological_recipes import FINAL_WEIGHTS, PREFIX  # noqa: E402


def fixed_blend(predictions: list[np.ndarray]) -> np.ndarray:
    """Apply .34/.33/.33 in FP32 before the official scoring-time clipping."""
    if len(predictions) != 3 or any(p.shape != predictions[0].shape for p in predictions):
        raise ValueError("the fixed TL3 endpoint requires three aligned arrays")
    result = np.float32(0.34) * np.asarray(predictions[0], dtype=np.float32)
    result += np.float32(0.33) * np.asarray(predictions[1], dtype=np.float32)
    result += np.float32(0.33) * np.asarray(predictions[2], dtype=np.float32)
    return result


def load_prediction(
    root: Path, run: str, groups: list[int], split_sha256: str
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    """Bind a complete official-valid prediction bank to its row identities."""
    path = root / "cache" / f"pred_{run}_valid.npy"
    metadata = json.loads(path.with_suffix(".json").read_text())
    rows = metadata.get("row_groups")
    if (
        metadata.get("run") != run
        or metadata.get("split") != "valid"
        or metadata.get("split_manifest_sha256") != split_sha256
        or metadata.get("inference_precision") != "fp32"
        or not isinstance(rows, list)
        or len(rows) != len(set(rows))
        or set(rows) != set(groups)
        or metadata.get("prediction_sha256") != file_sha256(path)
    ):
        raise ValueError(f"prediction ancestry, precision or full-valid coverage differs: {run}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (len(groups), 20000, 2):
        raise ValueError(f"prediction array is not a full-sequence FP32 endpoint: {run}")
    return array, {group: i for i, group in enumerate(rows)}, metadata


def grade(data_root: Path, manifest_path: Path, output: Path) -> dict[str, object]:
    """Stream official targets once and score all fixed members and their blend."""
    split = load_campaign_split(manifest_path, data_root)
    groups = list(split.evaluation_groups)
    if len(groups) != 1873 or split.valid_groups or len(split.train_groups) != 10325:
        raise ValueError("TL3 restoration requires the admitted10325/1873 primary boundary")
    arrays, positions, member_records = [], [], []
    accumulators = {name: GlobalAccumulator() for name in [*FINAL_WEIGHTS, "fixed_tl3_blend"]}
    for name, weight in FINAL_WEIGHTS.items():
        run = PREFIX + name
        checkpoint = data_root / "artifacts" / "runs" / run / "model.pt"
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_hash = file_sha256(checkpoint)
        run_provenance = json.loads(checkpoint.with_name("provenance.json").read_text())
        validate_chronological_teacher_checkpoint(
            ck, run_provenance, split, "model.pt", checkpoint_hash, run
        )
        if ck["epoch"] != (1 if name == "w18final_distilled_s5" else 3):
            raise ValueError(f"fixed TL3 member endpoint changed: {run}")
        array, position, metadata = load_prediction(data_root, run, groups, split.sha256)
        if metadata.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"evaluation predictions came from another checkpoint: {run}")
        arrays.append(array)
        positions.append(position)
        member_records.append(
            {
                "run": run,
                "weight": weight,
                "checkpoint_sha256": checkpoint_hash,
                "prediction_sha256": metadata["prediction_sha256"],
                "epoch": ck["epoch"],
            }
        )
        del ck
    parquet = pq.ParquetFile(data_root / "datasets" / "valid.parquet")
    selected_rows = 0
    for group in groups:
        table = parquet.read_row_group(group, columns=["t0", "t1", "is_scored"], use_threads=False)
        if table.num_rows != 20000:
            raise ValueError("validation row group is not one whole20000-row sequence")
        targets = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(
            np.float32
        )
        mask = table["is_scored"].to_numpy().astype(bool, copy=True)
        mask[:99] = False
        selected_rows += int(mask.sum())
        predictions = [
            np.asarray(array[position[group]])
            for array, position in zip(arrays, positions, strict=True)
        ]
        if any(not np.isfinite(prediction[99:]).all() for prediction in predictions):
            raise ValueError("member predictions contain nonfinite post-warmup values")
        for name, prediction in zip(FINAL_WEIGHTS, predictions, strict=True):
            accumulators[name].add(targets, prediction, mask)
        accumulators["fixed_tl3_blend"].add(targets, fixed_blend(predictions), mask)
    if selected_rows != 3528273:
        raise ValueError("full official-valid scoring mask differs from the admitted receipt")
    scores = {name: accumulator.result() for name, accumulator in accumulators.items()}
    payload = {
        "schema": "wnn.tl3-chronological-score.v1",
        "split_manifest_sha256": split.sha256,
        "valid_sequences": len(groups),
        "selected_rows": selected_rows,
        "members": member_records,
        "scores": scores,
        "clean_gru_reference_wp": 0.6778561232160607,
        "blend_delta_to_clean_gru": scores["fixed_tl3_blend"]["wp"] - 0.6778561232160607,
        "exposure": "exposed official valid; fixed coefficients, no new public/private result",
        "baseline_uncertainty": "single-seed benchmark; no paired training-variance claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("preserve the prior immutable score; use a new output path")
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(partial, output)
    return payload


def main() -> None:
    """Read only already-fitted endpoints; never train or optimize blend weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(grade(args.data_root, args.split_manifest, args.output)))


if __name__ == "__main__":
    main()
