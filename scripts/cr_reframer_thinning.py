"""R11 probe: how does the incumbent react to label-invariant row thinning (event-clock invariance)?

Targets change only on i0 price-code events and depend only on future quote events, so dropping rows whose i0
price codes (columns 0:COLS, default 22; 44 = whole i0 book) equal the previous row's leaves every kept row's
target exactly unchanged.
For sampled fold-0 evaluation sequences the fold-0 OOS members (c20260929_ft_since_s{3,4}_f0) are run on CPU on
(A) the original rows and (B) a copy with a random share of droppable rows (step >= 99) removed. Reported on the
kept scored rows: corr(pA, pB), pooled Global WP of A and B by class. Large WP loss in B means row density carries
signal (thinning augmentation would destroy it); equal WP with corr well below 1 means density-sensitivity without
signal (augmentation could regularise); corr ~1 means the model is already invariant.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_reframer_thinning.py OUT.json [PER_CLASS] [DROP] [COLS]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import train_rnn  # noqa: E402
from train_rnn import build_inputs, forward_chunk, make_model  # noqa: E402

from wnn.data import SequenceFile  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402

spec = importlib.util.spec_from_file_location("campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

COLS = int(sys.argv[4]) if len(sys.argv) > 4 else 22  # 22: no i0 price-code change; 44: i0 book unchanged
MEMBERS = ["c20260929_ft_since_s3_f0", "c20260929_ft_since_s4_f0"]


def load(name):
    ck = torch.load(ROOT / "artifacts" / "runs" / name / "model.pt", map_location="cpu", weights_only=False)
    for flag in ("raw_only", "online_norm", "heuristics", "prop_input", "arch"):
        if ck["args"].get(flag) and flag != "arch":
            raise ValueError(f"{name}: unsupported flag {flag}")
    model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0])
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def run(model, x):
    inputs = build_inputs(torch.from_numpy(x)[None])
    state, outs = None, []
    for s in range(0, inputs.shape[1], 4000):
        out, state = forward_chunk(model, inputs, slice(s, s + 4000), state)
        outs.append(out[..., :2].float())
    return torch.cat(outs, 1)[0].numpy().astype(np.float64)


def main():
    out = Path(sys.argv[1])
    per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    drop = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    torch.set_num_threads(3)
    train_rnn.RAW_ONLY = train_rnn.ONLINE_NORM = train_rnn.HEURISTICS = False
    models = [load(n) for n in MEMBERS]
    targets = cs.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
    position = {int(g): i for i, g in enumerate(targets["row_group"])}
    split = load_campaign_split(ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json", ROOT)
    groups = np.array(split.evaluation_groups)
    cls = np.array([int(targets["market_class"][position[int(g)]]) for g in groups])
    rng = np.random.default_rng(11)
    pick = np.concatenate([rng.choice(groups[cls == c], min(per_class, (cls == c).sum()), replace=False) for c in range(4)])
    cached = [np.load(ROOT / "cache" / f"pred_{n}_valid.npy", mmap_mode="r") for n in MEMBERS]
    sf = SequenceFile(ROOT / "datasets" / "valid.parquet")
    mom = {"A": [], "B": [], "AB": []}
    corr, parity, seq_cls, drop_share = [], [], [], []
    for g in pick:
        i = position[int(g)]
        s = sf.read(int(g))
        x = s.features
        y = s.targets.astype(np.float64)
        m = s.scored.copy()
        m[:99] = False
        same = np.r_[False, np.all(x[1:, :COLS] == x[:-1, :COLS], 1)]
        droppable = same & (np.arange(len(x)) >= 99)
        kill = droppable & (rng.random(len(x)) < drop)
        keep = ~kill
        pA = sum(0.5 * run(mod, x) for mod in models)
        pB_kept = sum(0.5 * run(mod, np.ascontiguousarray(x[keep])) for mod in models)
        pc = sum(0.5 * np.asarray(c[i], np.float64) for c in cached)
        parity.append(float(np.corrcoef(pA[m, 0], pc[m, 0])[0, 1]))
        sel = m[keep]
        yk, pAk = y[keep], pA[keep]
        ones = sel.astype(np.float64)
        mom["A"].append(np.stack([cs.moments(yk[:, k], pAk[:, k], ones) for k in range(2)]))
        mom["B"].append(np.stack([cs.moments(yk[:, k], pB_kept[:, k], ones) for k in range(2)]))
        pAB = 0.5 * (pAk + pB_kept)  # two-view average: metric value of the view diversity
        mom["AB"].append(np.stack([cs.moments(yk[:, k], pAB[:, k], ones) for k in range(2)]))
        corr.append(float(np.mean([np.corrcoef(pAk[sel, k], pB_kept[sel, k])[0, 1] for k in range(2)])))
        seq_cls.append(int(targets["market_class"][i]))
        drop_share.append(float(kill.mean()))
        print(g, seq_cls[-1], f"drop {kill.mean():.3f} corrAB {corr[-1]:.4f} parity {parity[-1]:.5f}", flush=True)
    seq_cls = np.array(seq_cls)
    A, B, AB = np.stack(mom["A"]), np.stack(mom["B"]), np.stack(mom["AB"])
    report = {"members": MEMBERS, "drop_probability": drop, "n_sequences": len(pick),
              "mean_drop_share": float(np.mean(drop_share)), "parity_vs_cached_min": float(np.min(parity)),
              "wp_A": cs.pooled(A)["wp"], "wp_B": cs.pooled(B)["wp"], "wp_AB": cs.pooled(AB)["wp"],
              "by_class": {int(c): {"wp_A": cs.pooled(A[seq_cls == c])["wp"], "wp_B": cs.pooled(B[seq_cls == c])["wp"],
                                    "wp_AB": cs.pooled(AB[seq_cls == c])["wp"],
                                    "corr_AB_median": float(np.median(np.array(corr)[seq_cls == c]))} for c in range(4)}}
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
