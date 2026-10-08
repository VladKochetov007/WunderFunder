"""Grade one fold-0 validation prediction with the official metric on the frozen 468 groups.

The metric is ``global_wp_clip2_v1`` from ``wnn.gmetric``: pooled over every selected row of the
frozen evaluation groups, with the fixed scored mask and the warm-up exclusion. The prediction
sidecar must bind the same run, epoch, split manifest and row order as the target cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SEQUENCE_ROWS = 20_000
EXPECTED_GROUPS = 468


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="repository root with cache/ and src/")
    parser.add_argument("--run", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from wnn.gmetric import METRIC_ID, WARMUP, GlobalAccumulator  # noqa: E402

    split_path = root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = json.loads(split_path.read_text())
    eval_groups = [int(g) for g in split["evaluation"]["valid"]]
    if len(eval_groups) != EXPECTED_GROUPS or len(set(eval_groups)) != EXPECTED_GROUPS:
        raise ValueError("frozen fold-0 evaluation panel is not 468 unique groups")
    split_sha = _file_sha256(split_path)

    pred_path = root / "cache" / f"pred_{args.run}_valid.npy"
    sidecar = json.loads(pred_path.with_suffix(".json").read_text())
    if (
        sidecar.get("run") != args.run
        or sidecar.get("split") != "valid"
        or int(sidecar.get("epoch")) != args.epoch
        or sidecar.get("split_manifest_sha256") != split_sha
    ):
        raise ValueError("prediction sidecar disagrees with the requested run, epoch or split")
    prediction_sha = _file_sha256(pred_path)
    if sidecar.get("prediction_sha256") != prediction_sha:
        raise ValueError("prediction bytes differ from the sidecar hash")

    with np.load(root / "cache" / "valid_targets.npz", allow_pickle=False) as cache:
        row_groups = [int(g) for g in cache["row_group"]]
        if sidecar.get("row_groups") != row_groups:
            raise ValueError("prediction row order differs from valid_targets")
        position = {group: index for index, group in enumerate(row_groups)}
        if any(group not in position for group in eval_groups):
            raise ValueError("evaluation group missing from valid_targets")
        indices = np.asarray([position[g] for g in eval_groups], dtype=np.int64)
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        is_scored = np.asarray(cache["mask"][indices], dtype=bool)

    predictions = np.asarray(
        np.load(pred_path, mmap_mode="r", allow_pickle=False)[indices], dtype=np.float32
    )
    if predictions.shape != (EXPECTED_GROUPS, SEQUENCE_ROWS, 2):
        raise ValueError(f"unexpected prediction shape {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite predictions on the evaluation groups")
    if targets.shape != predictions.shape or is_scored.shape != targets.shape[:2]:
        raise ValueError("target/mask shapes do not match predictions")

    mask = is_scored & (np.arange(SEQUENCE_ROWS) >= WARMUP)[None, :]
    accumulator = GlobalAccumulator()
    accumulator.add(targets, predictions, mask)
    result = dict(accumulator.result())
    if result.get("metric_id", METRIC_ID) != METRIC_ID:
        raise ValueError("metric id changed unexpectedly")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / f"eval468_pred_{args.run}_ep{args.epoch}.npy", predictions)
    record = {
        "run": args.run,
        "epoch": args.epoch,
        "metric_id": METRIC_ID,
        "result": result,
        "selected_rows": int(mask.sum()),
        "evaluation_groups": len(eval_groups),
        "prediction_sha256": prediction_sha,
        "split_manifest_sha256": split_sha,
    }
    (args.out_dir / f"{args.run}.grade.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"run": args.run, "result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
