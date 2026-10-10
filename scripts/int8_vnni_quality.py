"""Official-metric quality of the dynamic int8 (q8vnni) engine path against float32 and i16.

Runs held-out fold-1 sequences through `wnn_native.Predictor` (the production streaming path:
sanitize, causal features, per-sequence reset) once per precision and scores each prediction set
with `wnn.gmetric.global_wp` (clipped, pooled, two-target average) on the scored rows.

    PYTHONPATH=src .venv/bin/python -I scripts/int8_vnni_quality.py --lib DIR \
        --weights bd26_h230_f1_s3_ep3.npz --sequences 60 --out result.json

`--lib` is a directory holding the wnn_native.abi3.so built from this tree (the shared .venv build
is not touched). `--weights` is an npz of the checkpoint (see `scripts/export_native_weights.py`
for the same tensors). Evaluation groups are the first N of the fold-1 evaluation list in the
frozen split manifest, so fitting never saw them (the run was trained with fold 1 held out).
Reports WP per precision, WP difference from float32, per-target correlations, the largest
absolute output difference and a paired sequence-bootstrap interval for the WP difference.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402
from wnn.gmetric import global_wp  # noqa: E402

SPLIT = MAIN / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
VALID = MAIN / "datasets/valid.parquet"


def sequences(count: int):
    groups = json.loads(SPLIT.read_text())["evaluation"]["valid"][:count]
    pf = pq.ParquetFile(VALID)
    for g in groups:
        t = pf.read_row_group(g, use_threads=False)
        feats = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        y = np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        need = t["need_prediction"].to_numpy().astype(bool)
        mask = t["is_scored"].to_numpy().astype(bool) & need
        yield g, feats, y, mask


def predict(model, feats: np.ndarray) -> np.ndarray:
    model.reset()
    out = np.empty((len(feats), 2), dtype=np.float32)
    for i, row in enumerate(feats):
        out[i] = model.step(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--sequences", type=int, default=60)
    ap.add_argument("--precisions", default="f32,i16,q8x,q8vnni")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.lib))
    native = importlib.import_module("wnn_native")
    assert Path(native.__file__).resolve().parent == args.lib.resolve(), native.__file__
    with np.load(args.weights) as z:
        weights = {k: np.ascontiguousarray(z[k], dtype=np.float32) for k in z.files}
    names = args.precisions.split(",")
    models = {n: native.Predictor(weights, precision=n) for n in names}
    for n, m in models.items():
        print(n, "->", m.precision, flush=True)

    store = {n: [] for n in names}
    targets, masks, ids = [], [], []
    for g, feats, y, mask in sequences(args.sequences):
        ids.append(g)
        targets.append(y)
        masks.append(mask)
        for n, m in models.items():
            store[n].append(predict(m, feats))
        print("sequence", g, "done", flush=True)

    def score(preds):
        return global_wp(np.concatenate(targets), np.concatenate(preds), np.concatenate(masks))

    result = {"sequences": ids, "weights": str(args.weights), "models": {}}
    base = score(store["f32"])
    rng = np.random.default_rng(0)
    worst = {}
    for n in names:
        res = score(store[n])
        diff = np.concatenate([store[n][i][masks[i]] - store["f32"][i][masks[i]] for i in range(len(ids))])
        entry = {
            "wp": res["wp"],
            "t0": res["t0"],
            "t1": res["t1"],
            "wp_minus_f32": res["wp"] - base["wp"],
            "max_abs_output_diff_vs_f32": float(np.abs(diff).max()) if n != "f32" else 0.0,
            "rms_output_diff_vs_f32": float(np.sqrt((diff**2).mean())),
            "selected_rows": res["selected_rows"],
        }
        # Paired bootstrap over sequences for the WP difference from float32.
        if n != "f32":
            deltas = []
            for _ in range(200):
                pick = rng.integers(0, len(ids), len(ids))
                s = [(np.concatenate([targets[i] for i in pick]), np.concatenate([store[x][i] for i in pick]), np.concatenate([masks[i] for i in pick])) for x in (n, "f32")]
                deltas.append(global_wp(*s[0])["wp"] - global_wp(*s[1])["wp"])
            entry["bootstrap_wp_delta_95"] = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
        worst[n] = entry
        print(n, json.dumps(entry), flush=True)
    result["models"] = worst
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
