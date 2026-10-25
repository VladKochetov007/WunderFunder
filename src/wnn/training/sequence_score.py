from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wnn.data import SEQUENCE_LENGTH, WARMUP, SequenceFile
from wnn.evaluation import paired_component_moments
from wnn.gmetric import GlobalAccumulator, _merge_moments
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.sequence_cache import validate_export_schema, validate_producer
from wnn.training.sequence_epoch_prediction import EPOCH_KIND, validate_epoch_producer


def score_export(root: Path, split_manifest: Path, predictions: Path, output: Path) -> None:
    """Score every held-out row once with the canonical pooled official metric."""
    split = load_campaign_split(split_manifest, root)
    manifest_path = predictions / "manifest.json"
    if manifest_path.stat().st_size > 32_000_000:
        raise ValueError("prediction manifest exceeds bounded metadata size")
    meta = json.loads(manifest_path.read_text())
    if meta.get("schema") != "sequence-predictions-v1" or meta.get("complete") is not True:
        raise ValueError("score requires a completed held-out prediction export")
    if meta["split_sha256"] != split.sha256:
        raise ValueError("prediction split hash mismatch")
    validate_export_schema(meta, "evaluation")
    fitting = [("train", g) for g in split.train_groups] + [
        ("valid", g) for g in split.valid_groups
    ]
    epoch_evaluation = meta["producer"].get("kind") == EPOCH_KIND
    if epoch_evaluation:
        if json.loads(split_manifest.read_text()).get("schema") != "wnn.sequence-inner-split.v1":
            raise ValueError("retained epoch selection cannot consume outer evaluation")
        validate_epoch_producer(meta["producer"], split.sha256, fitting)
    else:
        validate_producer(meta["producer"], split.sha256, fitting)
    members = [(entry["split"], entry["row_group"]) for entry in meta["sequences"]]
    expected = [("valid", group) for group in split.evaluation_groups]
    if len(set(members)) != len(members) or set(members) != set(expected):
        raise ValueError("held-out prediction coverage is incomplete or duplicated")
    source = SequenceFile(root / "datasets/valid.parquet")
    accumulator = GlobalAccumulator()
    centered_by_component: dict[int, np.ndarray] = {}
    component_map = {(s, g): c for s, g, c in split.components}
    for entry in meta["sequences"]:
        path = predictions / entry["file"]
        if path.parent.resolve() != predictions.resolve() or path.is_symlink():
            raise ValueError("prediction file escapes export root")
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("prediction hash mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (SEQUENCE_LENGTH, 2 + meta["hint_dim"]):
            raise ValueError("prediction shape/dtype mismatch")
        if not np.isfinite(values).all():
            raise ValueError("nonfinite prediction export")
        seq = source.read(entry["row_group"])
        if seq.scored is None or seq.targets is None:
            raise ValueError("evaluation labels/mask unavailable")
        mask = seq.scored.copy()
        mask[:WARMUP] = False
        accumulator.add(
            seq.targets, values[:, :2], mask, group=component_map[("valid", entry["row_group"])]
        )
        if mask.any():
            moments, _ = paired_component_moments(
                seq.targets[None], values[None, :, :2], values[None, :, :2], mask[None]
            )
            component = component_map[("valid", entry["row_group"])]
            previous = centered_by_component.get(component, np.zeros((2, 6), dtype=np.float64))
            centered_by_component[component] = np.stack(
                [_merge_moments(previous[k], moments[0, k]) for k in range(2)]
            )
    receipt = {
        "score": accumulator.result(),
        "split_sha256": split.sha256,
        "predictions_manifest_sha256": file_sha256(manifest_path),
        "producer": meta["producer"],
        "sequences": len(members),
        "component_raw_sums": {str(k): v.tolist() for k, v in accumulator.groups.items()},
        "component_centered_moments": {
            str(k): v.tolist() for k, v in centered_by_component.items()
        },
        "interpretation": (
            "Inner development metric for epoch selection; not outer confirmation or public score."
            if epoch_evaluation
            else "Exposed fold; fixed endpoint, not checkpoint selection or public score."
        ),
    }
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Grade an immutable endpoint export without importing or executing a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score_export(args.data_root, args.split_manifest, args.predictions, args.output)
