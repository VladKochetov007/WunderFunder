"""Assemble scored-row flat arrays for a fold: y, members' predictions, heuristic causal features, EWMA stats.
Usage: build_flat.py FOLD OUTDIR  -> OUTDIR/flat_f{FOLD}.npz (+ member preds .npy)"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from crossfold_blend import load_fold

BANKS = {0: "BANK_FOLD0_FULL.jsonl", 1: "BANK_FOLD1.jsonl"}

def main():
    fold = int(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    split = ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
    bank = ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank" / BANKS[fold]
    y, mask, comp, models = load_fold(bank, fold, split, ROOT / "cache/valid_targets.npz")
    groups = [int(g) for g in json.loads(split.read_text())["evaluation"]["valid"]]
    with np.load(ROOT / "cache/valid_targets.npz") as t:
        pos = {int(g): i for i, g in enumerate(t["row_group"])}
        cls_all = t["market_class"]
    gidx = np.array([pos[g] for g in groups])
    h = np.load(ROOT / "cache/heur_valid.npz")
    hseq, hstep = h["seq"], h["step"]
    names = [str(n) for n in h["names"]]
    keep_feats = ["since_book", "since_trade", "log_step", "rate_book_hl64", "rate_trade_hl64", "absd_p0_hl64", "depth_imb", "tradevol_hl16"]
    fi = [names.index(n) for n in keep_feats]
    Xh = h["X"]
    order = np.argsort(hseq, kind="stable"); hs = hseq[order]
    lo = np.searchsorted(hs, np.arange(1873)); hi = np.searchsorted(hs, np.arange(1873), side="right")
    seq_i, step_i, feats = [], [], []
    for i, g in enumerate(gidx):
        rows = order[lo[g]:hi[g]]
        steps = hstep[rows]
        want = np.nonzero(mask[i])[0]
        sel = {int(s): r for s, r in zip(steps, rows)}
        miss = [s for s in want if int(s) not in sel]
        if miss:
            raise SystemExit(f"heur missing {len(miss)} scored rows seq {i}")
        feats.append(Xh[[sel[int(s)] for s in want]][:, fi]); seq_i.append(np.full(len(want), i, np.int32)); step_i.append(want.astype(np.int32))
    seq_i = np.concatenate(seq_i); step_i = np.concatenate(step_i); feats = np.concatenate(feats).astype(np.float32)
    ys = np.clip(y[mask], -2, 2).astype(np.float32)
    np.savez(out / f"flat_f{fold}.npz", y=ys, seq=seq_i, step=step_i, feats=feats, feat_names=np.array(keep_feats),
             comp=comp, market_class=cls_all[gidx], groups=np.array(groups))
    members = {}
    for stem, (name, arr, o) in models.items():
        if "lead" in name: continue
        try:
            p = np.asarray(arr[o], dtype=np.float32)
            members[stem] = name
            np.save(out / f"p_f{fold}_{stem}.npy", np.clip(p[mask], -2, 2).astype(np.float32))
        except Exception as e:
            print("skip", name, e)
    (out / f"members_f{fold}.json").write_text(json.dumps(members, indent=1))
    print(fold, len(members), ys.shape)

main()
