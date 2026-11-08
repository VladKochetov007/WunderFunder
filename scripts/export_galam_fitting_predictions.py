"""Export one admitted GALAM epoch on fitting features for heterogeneous KD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.data import SequenceFile
from wnn.splits import CampaignSplit, file_sha256, load_campaign_split
from wnn.training.sequence_epoch_prediction import digest, epoch_admission
from wnn.training.sequence_model_helpers import detach_state
from wnn.training.sequence_precision import configure_precision_mode
from wnn.training.sequence_predict import load_endpoint

# Native checkpoint/index JSON has heterogeneous, producer-validated fields.
JsonObject = dict[str, Any]
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
CHECKPOINT_SHA256 = "c825ac5221490109844b090a0699b30225ac2a211287b7f8abf409b2a69f05d9"


def write_new(path: Path, value: object) -> None:
    """Publish a new bounded evidence file without replacing an earlier attempt."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def groups_hash(groups: list[int]) -> str:
    """Hash ordered physical row groups without inventing chronological identifiers."""
    return hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()


def validate_payload(payload: JsonObject, split: CampaignSplit) -> None:
    """Keep original normalization and require fitting-only native model ancestry."""
    fitting = [["train", group] for group in split.train_groups]
    evaluation = [["valid", group] for group in split.evaluation_groups]
    data, config = payload["data"], payload["config"]
    norm = data.get("normalization", {})
    if (
        data.get("fit_members") != fitting
        or data.get("evaluation_members") != evaluation
        or data.get("split_sha256") != split.sha256
        or data.get("teacher") is not None
        or data.get("hint_members") is not None
        or config.get("kd_weight") != 0
        or config.get("hint_weight") != 0
        or norm.get("scope") != "all_fit"
        or norm.get("sequence_count") != 10325
        or config["model"].get("architecture") != "galam"
        or config["model"].get("input_feature_mode", "raw112") != "raw112"
        or config.get("precision_mode", "fp32") != "fp32"
        or payload.get("epoch") != 2
    ):
        raise ValueError("GALAM epoch is not the declared clean, raw112, human-epoch3 teacher")


@torch.inference_mode()
def export(args: argparse.Namespace) -> None:
    """Read training features only, reset per sequence and retain original causal state."""
    started = time.monotonic()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or not torch.cuda.is_available():
        raise ValueError("select exactly one explicitly allocated CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise ValueError("this exporter has one bounded worker")
    configure_precision_mode("fp32")
    torch.set_num_threads(args.cpu_threads)
    split = load_campaign_split(args.split_manifest, args.data_root)
    if (
        split.sha256 != SPLIT_SHA256
        or len(split.train_groups) != 10325
        or split.valid_groups
        or len(split.evaluation_groups) != 1873
    ):
        raise ValueError("the admitted chronological split is required")
    if file_sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("the export must use the pinned g2-LSTM128 human-epoch3 checkpoint")
    admission = epoch_admission(args.checkpoint, args.epoch_index, args.epoch_index_hash)
    model, payload = load_endpoint(args.checkpoint, torch.device("cuda:0"), admission)
    validate_payload(payload, split)
    identity = admission["index"]["identity"]
    if digest(identity["data"]) != identity["data_sha256"]:
        raise ValueError("native data identity differs")
    if digest(identity["source_files"]) != identity["source_sha256"]:
        raise ValueError("native source identity differs")
    checkpoint_hash = file_sha256(args.checkpoint)
    index_hash = file_sha256(args.epoch_index)
    groups = list(split.train_groups)
    required = len(groups) * 20000 * 2 * 4 + (12 << 30)
    if shutil.disk_usage(args.output.parent).free < required:
        raise OSError("insufficient disk for the fitting bank and twelve-GiB headroom")
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(
        args.output / "request.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    path = args.output / "predictions.npy"
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(10325, 20000, 2))
    source = SequenceFile(args.data_root / "datasets/train.parquet")
    chunk = int(payload["config"]["chunk_rows"])
    batch_durations = []
    for start in range(0, len(groups), args.batch):
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("GALAM fitting export exceeded its bounded allocation")
        batch_started = time.monotonic()
        selected = groups[start : start + args.batch]
        rows = []
        for group in selected:
            features = source.read(group, targets=False, mask=False).features
            if features.shape != (20000, 112) or not np.isfinite(features).all():
                raise ValueError("invalid complete training feature sequence")
            rows.append(torch.from_numpy(np.array(features, dtype=np.float32, copy=True)))
        inputs = torch.stack(rows)
        del rows
        state = None
        for offset in range(0, 20000, chunk):
            prediction, _, state = model(inputs[:, offset : offset + chunk].to("cuda:0"), state)
            if not torch.isfinite(prediction).all():
                raise ValueError("nonfinite GALAM fitting prediction")
            end = offset + prediction.shape[1]
            values[start : start + len(selected), offset:end] = prediction.float().cpu().numpy()
            state = detach_state(state)
        values.flush()
        elapsed = time.monotonic() - batch_started
        batch_durations.append(elapsed)
        print(
            json.dumps(
                {
                    "completed_sequences": start + len(selected),
                    "total": len(groups),
                    "batch_seconds": elapsed,
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    values.flush()
    del values
    prediction_hash = file_sha256(path)
    if (
        file_sha256(args.checkpoint) != checkpoint_hash
        or file_sha256(args.epoch_index) != index_hash
    ):
        raise ValueError("original checkpoint or epoch index changed during export")
    side = {
        "schema": "wnn.galam-fitting-feature-export.v1",
        "producer": "galam_feature_export",
        "run": "g2-galam-lstm128",
        "epoch": 2,
        "split": "train",
        "feature_only": True,
        "inference_precision": "fp32",
        "split_manifest_sha256": split.sha256,
        "checkpoint_sha256": checkpoint_hash,
        "prediction_sha256": prediction_hash,
        "epoch_index_sha256": index_hash,
        "epoch_index_canonical_sha256": args.epoch_index_hash,
        "row_groups": groups,
        "row_groups_sha256": groups_hash(groups),
        "fit_groups": {"train": groups, "valid": []},
        "evaluation_valid_groups": list(split.evaluation_groups),
        "source_sha256": identity["source_sha256"],
        "normalization": payload["data"]["normalization"],
        "elapsed_seconds": time.monotonic() - started,
        "batch_seconds": batch_durations,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_uuid": visible,
    }
    write_new(path.with_suffix(".json"), side)
    qualification = {
        "schema": "wnn.teacher-source-qualification.v1",
        "run": side["run"],
        "producer": side["producer"],
        "epoch": 2,
        "checkpoint_sha256": checkpoint_hash,
        "prediction_sha256": prediction_hash,
        "sidecar_sha256": file_sha256(path.with_suffix(".json")),
        "split_manifest_sha256": split.sha256,
        "fit_groups": side["fit_groups"],
        "evaluation_valid_groups": side["evaluation_valid_groups"],
        "source_sha256": identity["source_sha256"],
        "normalization_fit_groups_sha256": groups_hash(groups),
    }
    write_new(args.output / "qualification.json", qualification)

    def pin(p: Path) -> JsonObject:
        """Bind each immutable artifact by its actual bytes and resolved path."""
        return {"path": str(p.resolve()), "sha256": file_sha256(p)}

    write_new(
        args.output / "source-record.json",
        {
            "run": side["run"],
            "family": "galam",
            "producer": side["producer"],
            "epoch": 2,
            "coefficient": 0.1,
            "prediction": pin(path),
            "sidecar": pin(path.with_suffix(".json")),
            "checkpoint": pin(args.checkpoint),
            "qualification": pin(args.output / "qualification.json"),
            "epoch_index": pin(args.epoch_index),
            "epoch_index_canonical_sha256": args.epoch_index_hash,
        },
    )


def main() -> None:
    """Parse one explicit feature-export allocation; no model fitting is performed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epoch-index", type=Path, required=True)
    parser.add_argument("--epoch-index-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.batch < 1 or args.cpu_threads < 1 or args.max_wall_seconds < 1:
        parser.error("resource bounds must be positive")
    export(args)


if __name__ == "__main__":
    main()
