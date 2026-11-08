from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wnn.gmetric import GlobalAccumulator
from wnn.splits import load_campaign_split
from wnn.training.sequence_distributed import initialize_distributed
from wnn.training.sequence_fit_utils import collate_sequences
from wnn.training.sequence_predict import load_endpoint, predict_sequence_batch
from wnn.training.sequence_provenance import file_sha256, source_identity
from wnn.training.sequence_records import SequenceRecords
from wnn.training.sequence_reference_cache import ReferenceCache

SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


def fixed_mix(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Compose raw FP32 predictions first; official clipping belongs to the metric."""
    if (
        reference.shape != candidate.shape
        or reference.dtype != np.float32
        or candidate.dtype != np.float32
    ):
        raise ValueError("fixed mixture requires matching raw FP32 prediction shapes")
    return np.float32(0.9) * reference + np.float32(0.1) * candidate


class ReadoutScores:
    """Accumulate unchanged official WP and monitor-compatible raw-prediction WMSE."""

    def __init__(self) -> None:
        self.pooled = {name: GlobalAccumulator() for name in ("candidate", "mixture", "reference")}
        self.squared = dict.fromkeys(self.pooled, 0.0)
        self.weight = 0.0

    def add(
        self, targets: np.ndarray, candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray
    ) -> None:
        """Score candidate and frozen mixture from the same single forward and same rows."""
        selected = mask[..., None]
        target = np.clip(np.where(selected, targets, 0.0), -2.0, 2.0).astype(np.float64)
        weight = np.abs(target)
        self.weight += float(weight.sum())
        for name, values in {
            "candidate": candidate,
            "mixture": fixed_mix(reference, candidate),
            "reference": reference,
        }.items():
            self.pooled[name].add(targets, values, mask)
            safe_values = np.where(selected, values, 0.0).astype(np.float64)
            self.squared[name] += float((weight * np.square(safe_values - target)).sum())

    def result(self) -> dict[str, object]:
        """Return explicit standalone, fixed-mixture and reused-reference metrics."""
        return {
            name: accumulator.result()
            | {"wmse": self.squared[name] / self.weight if self.weight else 0.0}
            for name, accumulator in self.pooled.items()
        }


def readout(args: argparse.Namespace) -> dict[str, object]:
    """Produce one fixed readout with compact scored-row predictions and exact ancestry."""
    started = time.monotonic()
    context = initialize_distributed()
    if context.world_size != 1:
        raise ValueError("fixed readouts use one independent GPU")
    split = load_campaign_split(args.split_manifest, args.data_root)
    if split.sha256 != SPLIT_SHA or split.valid_groups or len(split.train_groups) != 10325:
        raise ValueError("readout requires the exact official train-only primary split")
    members = [(split.evaluation_source, group) for group in split.evaluation_groups]
    if len(members) != 1873 or split.evaluation_mask != "is_scored":
        raise ValueError("readout requires all1873 official scored validation sequences")
    if file_sha256(args.reference_root / "manifest.json") != args.reference_manifest_sha256:
        raise ValueError("transferred clean reference manifest differs from ROOT's pin")
    reference = ReferenceCache(args.reference_root, split, members, membership="evaluation")
    model, payload = load_endpoint(args.checkpoint, context.device)
    expected_schema = "wnn.sequence-swa.v1" if args.readout == "swa3456" else "wnn.sequence-fit.v1"
    if payload.get("checkpoint_schema") != expected_schema:
        raise ValueError("declared fixed readout does not match the actual checkpoint schema")
    if (
        payload["config"]["epochs"] != 6
        or payload["data"]["fit_members"] != [["train", g] for g in split.train_groups]
        or payload["data"]["split_sha256"] != split.sha256
        or payload["data"]["teacher"] is not None
    ):
        raise ValueError("candidate readout ancestry differs from the fresh six-epoch recipe")
    dataset = SequenceRecords(
        args.data_root,
        members,
        0,
        training_mask="is_scored",
        input_feature_mode=payload["config"]["model"].get("input_feature_mode", "raw112"),
    )
    dataset.validate_source_schema()
    args.output.mkdir(parents=True, exist_ok=False)
    scores, entries = ReadoutScores(), []
    for start in range(0, len(dataset), args.batch_size):
        indices = list(range(start, min(start + args.batch_size, len(dataset))))
        batch = collate_sequences(dataset, indices)
        candidate = predict_sequence_batch(
            model, batch["x"], payload["config"]["chunk_rows"], context.device
        )
        refs = np.stack([reference.read(*members[index], 0)[0].numpy() for index in indices])
        scores.add(batch["y"].numpy(), candidate, refs, batch["mask"].numpy())
        for position, index in enumerate(indices):
            source, group = members[index]
            rows = np.flatnonzero(batch["mask"][position].numpy()).astype(np.int32)
            target = args.output / f"{source}-{group}.npz"
            with target.open("xb") as stream:
                np.savez(
                    stream, rows=rows, predictions=candidate[position, rows].astype(np.float32)
                )
            entries.append(
                {
                    "source": source,
                    "row_group": group,
                    "file": target.name,
                    "sha256": file_sha256(target),
                    "selected_rows": int(rows.size),
                }
            )
    result = scores.result()
    if any(score["selected_rows"] != 3528273 for score in result.values()):
        raise ValueError("fixed readout scored-row coverage differs from3528273")
    if abs(result["reference"]["wp"] - 0.6923774248169172) > 1e-10:
        raise ValueError("transferred clean TL3 does not reproduce its qualified full-valid score")
    reference.verify_unchanged()
    receipt = {
        "schema": "wnn.fixed-swa-e6-readout.v1",
        "complete": True,
        "readout": args.readout,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "split_sha256": split.sha256,
        "producer": payload["provenance"],
        "readout_implementation": source_identity(
            Path(__file__).parents[3], Path(__file__).parents[3] / "scripts/train_sequence.py"
        ),
        "config": payload["config"],
        "reference": reference.identity,
        "mixture_weights": {"reference": 0.9, "candidate": 0.1},
        "metrics": result,
        "sequence_count": len(entries),
        "selected_rows": 3528273,
        "elapsed_seconds": time.monotonic() - started,
        "prediction_schema": "raw FP32 scored rows only, int32 original row indices; all rows forwarded",
        "sequences": entries,
        "validation_exposure": "development-exposed; fixed readouts, no selection",
    }
    with (args.output / "readout.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main() -> None:
    """Run one predeclared post-fit readout; never update weights or optimize mixture weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readout", choices=("swa3456", "epoch6"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    result = readout(args)
    print(json.dumps({"readout": result["readout"], "metrics": result["metrics"]}))


if __name__ == "__main__":
    main()
