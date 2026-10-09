"""Score one exported run on its evaluation fold and write a ~1 KB ``score.json`` (hyper-parameter search ingest).

Reads ``cache/pred_<run>_valid.npy`` (+ sidecar) produced by ``predict_run.py``, aligns it to the fold's fixed evaluation
groups and writes the official ``global_wp_clip2_v1`` plus per-market-class weighted Pearson, the fraction of the
prediction mass that is non-finite, and cheap training diagnostics from ``history.json``.  With ``--cleanup`` it deletes intermediate checkpoints and
the training state after scoring; predictions are kept (every good trial is a blend candidate) unless ``--drop-predictions``.  Nothing here selects or modifies the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grade_prediction_bank import SCORED_FROM_ROW, Lazy, aligned, score, sha256  # noqa: E402

from wnn.data import market_class  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--cleanup", action="store_true", help="delete intermediate checkpoints and training state")
    parser.add_argument("--drop-predictions", action="store_true", help="also delete the prediction array (default: keep it as a blend candidate)")
    args = parser.parse_args()
    run_dir = ROOT / "artifacts/runs" / args.run
    split = json.loads(args.split.read_text())
    groups = [int(g) for g in split["evaluation"]["valid"]]
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as cache:
        position = {int(g): i for i, g in enumerate(cache["row_group"])}
        index = [position[g] for g in groups]
        y = np.asarray(cache["y"][index], dtype=np.float32)
        mask = np.asarray(cache["mask"][index], dtype=bool)
    mask &= (np.arange(y.shape[1]) >= SCORED_FROM_ROW)[None, :]
    prediction = ROOT / "cache" / f"pred_{args.run}_valid.npy"
    result = aligned({"prediction": str(prediction.relative_to(ROOT))}, groups, sha256(args.split))
    if isinstance(result, str):
        raise SystemExit(f"{args.run}: {result}")
    array, order = result
    lazy = Lazy(array, order)
    overall = score(y, mask, lazy)
    classes = np.array([market_class(y[i][mask[i]]) for i in range(len(y))])
    by_class = {}
    full = lazy.load() if hasattr(lazy, "load") else np.asarray(array)[order]
    for c in sorted(set(classes.tolist())):
        idx = np.where(classes == c)[0]
        by_class[int(c)] = score(y[idx], mask[idx], full[idx])["wp"]
    finite = float(np.isfinite(full[mask]).mean()) if mask.any() else 0.0
    history = []
    if (run_dir / "history.json").exists():
        history = json.loads((run_dir / "history.json").read_text())
    skipped = 0
    attempts = 0
    if history:
        cumulative = history[-1].get("update_diagnostics", {}).get("cumulative", {})
        skipped, attempts = cumulative.get("amp_skipped_updates", 0), cumulative.get("optimizer_attempts", 0)
    payload = {"schema": "wnn.hpo-score.v1", "run": args.run, "wp": overall["wp"], "t0": overall["t0"], "t1": overall["t1"],
               "class_wp": by_class, "finite_fraction": finite, "amp_skipped": skipped, "optimizer_attempts": attempts,
               "train_seconds": round(sum(row.get("train_seconds", 0) for row in history), 1), "epoch": args.epoch}
    (run_dir / "score.json").write_text(json.dumps(payload))
    print(json.dumps(payload), flush=True)
    if args.drop_predictions:
        prediction.unlink(missing_ok=True)
    if args.cleanup:
        for leftover in run_dir.glob("model_ep*.pt"):
            if leftover.name != f"model_ep{args.epoch}.pt":
                leftover.unlink(missing_ok=True)
        (run_dir / "training_state.pt").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
