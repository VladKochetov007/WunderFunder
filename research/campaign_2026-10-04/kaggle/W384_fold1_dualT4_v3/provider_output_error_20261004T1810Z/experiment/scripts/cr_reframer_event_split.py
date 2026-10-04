"""R04 probe: how much of D is resolved at the first target-change event, per class, and where is the model's covariance?

Identity per scored row r: D_r = J1 + R1 with J1 = D_r - D(e1) (resolved at the first future target-change row e1)
and R1 = D(e1) (the next episode's D). Reports per class the |y|-weighted variance shares of J1 and R1, their
covariance, and the OOS mix's weighted correlation with D, J1 and R1. If R1 carries a large share that the model
already tracks (corr with R1 well above 0), the label factorisation adds supervision only for the J1 part, which
aux-next already covers.

    WNN_CPU_PCT=6 WNN_MEM_MAX=6G scripts/limited .venv/bin/python scripts/cr_reframer_event_split.py OUT.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.splits import load_campaign_split  # noqa: E402

spec = importlib.util.spec_from_file_location("campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def wstats(w, cols):
    """Weighted covariance matrix of the given columns."""
    W = w.sum()
    X = np.stack(cols, 1)
    mu = (w[:, None] * X).sum(0) / W
    Xc = X - mu
    return (Xc * w[:, None]).T @ Xc / W


def main() -> None:
    out = Path(sys.argv[1])
    targets = cs.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
    position = {int(g): i for i, g in enumerate(targets["row_group"])}
    steps = np.arange(20_000)
    acc = {c: [] for c in range(4)}
    for fold in range(4):
        split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json", ROOT)
        preds = [np.load(ROOT / "cache" / f"pred_c20260929_ft_since_s{s}_f{fold}_valid.npy", mmap_mode="r") for s in (3, 4)]
        for g in split.evaluation_groups:
            i = position[g]
            y = np.clip(np.asarray(targets["y"][i], np.float64), -2, 2)
            m = np.asarray(targets["mask"][i], bool) & (steps >= 99)
            p = sum(0.5 * np.asarray(pr[i], np.float64) for pr in preds)
            D = (y[:, 0] - y[:, 1]) / 2
            ch = np.flatnonzero(np.any(y[1:] != y[:-1], 1)) + 1
            k = np.searchsorted(ch, steps, side="right")
            ok = m & (k < len(ch))
            e1 = ch[np.minimum(k, len(ch) - 1)]
            J1 = D - D[e1]
            R1 = D[e1]
            w = np.abs(y).mean(1)
            Dg = (p[:, 0] - p[:, 1]) / 2
            acc[int(targets["market_class"][i])].append(np.c_[w, D, J1, R1, Dg][ok])
        print("fold", fold, flush=True)
    report = {}
    for c in range(4):
        a = np.concatenate(acc[c])
        w, D, J1, R1, Dg = a.T
        C = wstats(w, [D, J1, R1, Dg])
        sd = np.sqrt(np.diag(C))
        report[str(c)] = {
            "rows": int(len(a)),
            "var_D": float(C[0, 0]),
            "share_var_J1": float(C[1, 1] / C[0, 0]),
            "share_var_R1": float(C[2, 2] / C[0, 0]),
            "share_2cov_J1R1": float(2 * C[1, 2] / C[0, 0]),
            "corr_Dg_D": float(C[3, 0] / (sd[3] * sd[0])),
            "corr_Dg_J1": float(C[3, 1] / (sd[3] * sd[1])),
            "corr_Dg_R1": float(C[3, 2] / (sd[3] * sd[2])),
            "share_cov_Dg_D_from_J1": float(C[3, 1] / C[3, 0]),
        }
        print(c, json.dumps(report[str(c)]), flush=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
