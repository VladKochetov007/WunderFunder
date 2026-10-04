"""Score fixed cached public predictors on the entire supplied validation file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import (  # noqa: E402
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    GlobalAccumulator,
    validate_sequence,
)
from wnn.splits import is_holdout  # noqa: E402

MODELS = {
    "combo_public_0.6655": (
        ("c20261004_combo_s3_final_r1_fp32", "c20261004_combo_s3_final_r1", 3),
        ("c20261004_combo_head_s3_final_fp32", "c20261004_combo_head_s3_final", 0),
    ),
    "D256_public_0.6641": (
        ("c20261001_final_d256_s3", "c20261001_final_d256_s3", 3),
        ("c20261001_final_d256_s3_gcorrh", "c20261001_final_d256_s3_gcorrh", 0),
    ),
}


def digest(path: Path) -> str:
    """Read source hashes with bounded memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def score(output: Path) -> None:
    """Pool official moments row-group by row-group without stacking validation."""
    if output.exists():
        raise FileExistsError(output)
    data = ROOT / "datasets/valid.parquet"
    parquet = pq.ParquetFile(data)
    count = parquet.num_row_groups
    expected_shape = (count, SEQUENCE_LENGTH, 2)
    predictions, provenance = {}, {}
    for name, members in MODELS.items():
        values, identities = [], []
        for export, run, epoch in members:
            path = ROOT / f"cache/pred_{export}_valid.npy"
            metadata = json.loads(path.with_suffix(".json").read_text())
            checkpoint = ROOT / f"artifacts/runs/{run}/model_ep{epoch}.pt"
            if metadata["run"] != run or metadata["epoch"] != epoch:
                raise ValueError(f"Wrong fixed predictor identity: {path}")
            if metadata["row_groups"] != list(range(count)):
                raise ValueError(f"Wrong validation row-group order: {path}")
            if digest(path) != metadata["prediction_sha256"]:
                raise ValueError(f"Prediction hash changed: {path}")
            if digest(checkpoint) != metadata["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash changed: {checkpoint}")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if value.shape != expected_shape:
                raise ValueError(f"Wrong validation export shape: {path}: {value.shape}")
            values.append(value)
            identities.append(
                {
                    "run": run,
                    "epoch": epoch,
                    "coefficient": 0.5,
                    "prediction_sha256": metadata["prediction_sha256"],
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "inference_precision": metadata.get("inference_precision", "not recorded"),
                }
            )
        predictions[name], provenance[name] = values, identities
    partitions = ("all_supplied_validation", "excluded_from_final_fit", "validation_in_final_fit")
    accumulators = {
        name: {partition: GlobalAccumulator() for partition in partitions} for name in MODELS
    }
    sizes = dict.fromkeys(partitions, 0)
    ids = []
    columns = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", *TARGET_COLUMNS]
    for group in range(count):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        sequence_id, need = validate_sequence(table)
        ids.append(sequence_id)
        target = np.column_stack([table[name].to_numpy() for name in TARGET_COLUMNS])
        mask = need & table["is_scored"].to_numpy().astype(bool)
        partition = (
            "excluded_from_final_fit" if is_holdout(sequence_id) else "validation_in_final_fit"
        )
        sizes["all_supplied_validation"] += 1
        sizes[partition] += 1
        for name, members in predictions.items():
            prediction = (np.asarray(members[0][group], np.float64) + members[1][group]) / 2
            if not np.isfinite(prediction[need]).all():
                raise ValueError(f"Nonfinite required-row prediction: {name}, group{group}")
            accumulators[name]["all_supplied_validation"].add(target, prediction, mask)
            accumulators[name][partition].add(target, prediction, mask)
    if len(set(ids)) != count:
        raise ValueError("Repeated sequence identifiers")
    result = {
        "metric_id": "global_wp_clip2_v1",
        "dataset": "datasets/valid.parquet",
        "rows": count * SEQUENCE_LENGTH,
        "partition_sequences": sizes,
        "mask": "need_prediction AND is_scored",
        "members": provenance,
        "scores": {
            name: {partition: accumulator.result() for partition, accumulator in parts.items()}
            for name, parts in accumulators.items()
        },
        "scope": (
            "Fixed cached predictors, equal parent/head mixture. Combo uses FP32 exports; "
            "legacy D256 export precision was not recorded. This is not the exact native-i16 "
            "callback replay. Most supplied validation sequences were fitted; the 368-sequence "
            "excluded partition is historically exposed, not an untouched test. No selection."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"partition_sequences": sizes, "scores": result["scores"]}))


def main() -> None:
    """Score the frozen models and save their supplied-validation scorecard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    score(arguments.output)


if __name__ == "__main__":
    main()
