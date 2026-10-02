"""In-sample vs out-of-sample fit of distillation teachers (H-061 diagnostic).

Global WP of a teacher's cached train predictions against train targets (rows the teacher was fitted on) next to
its WP on valid evaluation rows it never saw. A large in-sample excess means the distillation target on train rows
carries memorised label noise. Subsamples train sequences to stay small; memory-maps everything.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def wp(y, p, m):
    out = []
    for t in range(2):
        yy = np.clip(y[..., t][m], -2, 2).astype(np.float64)
        pp = np.clip(p[..., t][m], -2, 2).astype(np.float64)
        w = np.abs(yy)
        sw = w.sum()
        my, mp = (w * yy).sum() / sw, (w * pp).sum() / sw
        cov = (w * (yy - my) * (pp - mp)).sum()
        out.append(cov / np.sqrt((w * (yy - my) ** 2).sum() * (w * (pp - mp) ** 2).sum()))
    return float(np.mean(out)), [float(v) for v in out]


def main():
    teacher, n_train = sys.argv[1], int(sys.argv[2])
    tt = np.load(ROOT / "cache" / "train_targets.npz", mmap_mode="r")
    print({k: tt[k].shape for k in tt.files}, flush=True)
    pred = np.load(ROOT / "cache" / f"pred_{teacher}_train.npy", mmap_mode="r")
    side = json.loads((ROOT / "cache" / f"pred_{teacher}_train.json").read_text())
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(pred.shape[0], size=min(n_train, pred.shape[0]), replace=False))
    y = np.asarray(tt["y"][idx]) if "y" in tt.files else None
    m = np.zeros((len(idx), 20000), bool); m[:, 99:] = True  # train cache has no scored mask: rows >= 99 on both sides
    p = np.asarray(pred[idx])
    res = {"teacher": teacher, "train_seqs": int(len(idx)), "train_wp": wp(y, p, m)}
    vt = np.load(ROOT / "cache" / "valid_targets.npz")
    vp = np.load(ROOT / "cache" / f"pred_{teacher}_valid.npy", mmap_mode="r")
    vm = np.zeros(vt["mask"].shape, bool); vm[:, 99:] = True
    if side.get("split_manifest_sha256"):
        from wnn.splits import load_campaign_split
        fold = teacher.rsplit("_f", 1)[1]
        split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json", ROOT)
        ev = np.isin(vt["row_group"], np.array(sorted(split.evaluation_groups)))
        vm &= ev[:, None]
    else:
        from wnn.splits import is_holdout
        ev = np.array([is_holdout(int(s)) for s in vt["seq_ix"]])
        vm &= ev[:, None]
    res["valid_eval_seqs"] = int(ev.sum())
    res["valid_wp"] = wp(vt["y"], np.asarray(vp), vm)
    res["insample_excess"] = res["train_wp"][0] - res["valid_wp"][0]
    print(json.dumps(res))


main()
