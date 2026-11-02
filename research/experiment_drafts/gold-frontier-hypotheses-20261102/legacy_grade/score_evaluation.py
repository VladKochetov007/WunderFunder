"""Grade an exact source-bound legacy evaluation export without importing a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wnn.gmetric import METRIC_ID, WARMUP, GlobalAccumulator
from wnn.splits import file_sha256, load_campaign_split

ROWS = 20_000


def main() -> None:
    """Check prediction, checkpoint and fold identity before official streamed scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-cache-sha256", required=True)
    args = parser.parse_args()
    if args.run in ("", ".", "..") or Path(args.run).name != args.run or args.epoch < 0:
        raise ValueError("run must be a simple name and epoch must be nonnegative")
    root = args.data_root.resolve(strict=True)
    split = load_campaign_split(args.split_manifest, root)
    groups = list(split.evaluation_groups)
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("evaluation membership must be nonempty and unique")
    checkpoint = root / "artifacts" / "runs" / args.run / f"model_ep{args.epoch}.pt"
    prediction = root / "cache" / f"pred_{args.run}_valid.npy"
    sidecar_path = prediction.with_suffix(".json")
    target_path = root / "cache" / "valid_targets.npz"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    group_hash = hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
    pred_hash = file_sha256(prediction)
    checkpoint_hash = file_sha256(checkpoint)
    if (
        sidecar.get("run") != args.run
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != args.epoch
        or sidecar.get("campaign_export_scope") != "evaluation"
        or sidecar.get("feature_only") is not True
        or sidecar.get("split_manifest_sha256") != split.sha256
        or sidecar.get("campaign_split_manifest_sha256") != split.sha256
        or sidecar.get("row_groups") != groups
        or sidecar.get("evaluation_membership_sha256") != group_hash
        or sidecar.get("checkpoint_sha256") != checkpoint_hash
        or sidecar.get("prediction_sha256") != pred_hash
        or sidecar.get("inference_precision") != "fp32"
    ):
        raise ValueError(
            "prediction sidecar differs from the requested checkpoint or evaluation split"
        )
    predicted = np.load(prediction, mmap_mode="r", allow_pickle=False)
    if predicted.shape != (len(groups), ROWS, 2) or predicted.dtype != np.float32:
        raise ValueError("prediction shape or dtype differs from full float32 sequences")
    target_hash = file_sha256(target_path)
    if target_hash != args.target_cache_sha256.lower():
        raise ValueError("target cache differs from the independently pinned SHA256")
    with np.load(target_path, allow_pickle=False) as data:
        target_groups = np.asarray(data["row_group"])
        targets = np.asarray(data["y"])
        masks = np.asarray(data["mask"])
    if (
        target_groups.ndim != 1
        or target_groups.dtype.kind not in "iu"
        or len(set(target_groups.tolist())) != len(target_groups)
        or targets.shape != (len(target_groups), ROWS, 2)
        or masks.shape != (len(target_groups), ROWS)
        or targets.dtype.kind != "f"
        or masks.dtype.kind not in "biu"
        or not np.isin(masks, (0, 1)).all()
    ):
        raise ValueError("target cache row groups, shape, dtype or mask are invalid")
    positions = {int(group): index for index, group in enumerate(target_groups)}
    if any(group not in positions for group in groups):
        raise ValueError("evaluation group missing from target cache")
    acc = GlobalAccumulator()
    for index, group in enumerate(groups):
        target = targets[positions[group]]
        pred = predicted[index]
        mask = masks[positions[group]].astype(bool, copy=True)
        mask[:WARMUP] = False
        if not np.isfinite(pred[mask]).all() or not np.isfinite(target[mask]).all():
            raise ValueError(f"nonfinite prediction or scored target in group {group}")
        acc.add(target, pred, mask)
    record = {
        "schema": "wnn.legacy-evaluation-grade.v1",
        "run": args.run,
        "epoch": args.epoch,
        "metric_id": METRIC_ID,
        "result": acc.result(),
        "groups": len(groups),
        "evaluation_membership_sha256": group_hash,
        "split_manifest_sha256": split.sha256,
        "checkpoint_sha256": checkpoint_hash,
        "prediction_sha256": pred_hash,
        "prediction_sidecar_sha256": file_sha256(sidecar_path),
        "target_cache_sha256": target_hash,
        "exposure": "development; not an independent holdout claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
