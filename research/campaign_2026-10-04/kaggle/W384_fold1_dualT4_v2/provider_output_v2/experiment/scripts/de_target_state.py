"""Current-state (spread) versus future-path share of the metric, and the structural-output bound.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/de_target_state.py

Uses the TRAIN-fitted current-code spread table of scripts/de_target_law.py and TRAIN fit-component
code-conditional target means (|y|-weighted). Graded on fold-0 evaluation sequences of the clean manifest,
rows step >= 99 AND is_scored, with the reference member cache/pred_c20260929_ft_since_s3_f0_valid.npy.
Writes research/campaign_2026-10-01/data_eval/target_state.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from de_public_proxy import sequence_table  # noqa: E402
from de_target_law import PRICE, QUANTA, code_hash, read  # noqa: E402

from wnn.gmetric import wp_from_sums  # noqa: E402

OUT = ROOT / "research" / "campaign_2026-10-01" / "data_eval"
MODEL = ROOT / "cache" / "pred_c20260929_ft_since_s3_f0_valid.npy"


def sums(y, p, w):
    wy, wp_ = w * y, w * p
    return np.array([w.sum(), wy.sum(), wp_.sum(), (wy * y).sum(), (wp_ * p).sum(), (wy * p).sum()])


class Pool:
    """Accumulate six-sum Global WP statistics for named two-target predictions."""

    def __init__(self):
        self.s = defaultdict(lambda: np.zeros((2, 6)))

    def add(self, name, y, p):
        for k in range(2):
            yc, pc = np.clip(y[:, k], -2, 2), np.clip(p[:, k], -2, 2)
            self.s[name][k] += sums(yc, pc, np.abs(yc))

    def wp(self):
        return {n: float(wp_from_sums(v).mean()) for n, v in self.s.items()}


def main() -> None:
    law = json.loads((OUT / "target_law_train.json").read_text())
    tables = np.load(OUT / "target_law_tables.npz")
    spread = dict(zip(tables["spread_code"].tolist(), tables["spread_ticks"].tolist(), strict=True))
    # code-conditional |y|-weighted target means on TRAIN fit components (real units, clipped)
    source = pq.ParquetFile(ROOT / "datasets" / "train.parquet")
    acc = defaultdict(lambda: np.zeros(6))  # w0, w0*t0, w1, w1*t1, wS, wS*S
    for _k, groups in law["selection"]["fit"].items():
        for g in groups:
            code, _, yi, cls = read(source, g)
            y = np.clip(yi[99:] * QUANTA[cls], -2, 2)
            c = code[99:]
            S = y[:, 0] + y[:, 1]
            wS = np.abs(y).sum(1)
            uniq, inv = np.unique(c, return_inverse=True)
            vals = np.stack([np.bincount(inv, v, len(uniq)) for v in (
                np.abs(y[:, 0]), np.abs(y[:, 0]) * y[:, 0], np.abs(y[:, 1]), np.abs(y[:, 1]) * y[:, 1], wS, wS * S)], 1)
            for u, v in zip(uniq.tolist(), vals, strict=True):
                acc[u] += v
    mean0 = {u: v[1] / v[0] for u, v in acc.items() if v[0] > 0}
    mean1 = {u: v[3] / v[2] for u, v in acc.items() if v[2] > 0}
    meanS = {u: v[5] / v[4] for u, v in acc.items() if v[4] > 0}
    print("code means", len(mean0), flush=True)

    t = sequence_table()
    rows = np.flatnonzero(t["fold"] == 0)
    with np.load(ROOT / "cache" / "valid_targets.npz") as z:
        Y, M, RG = z["y"], z["mask"], z["row_group"].astype(int)
    P = np.load(MODEL, mmap_mode="r")
    vsrc = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    pool = Pool()
    dec = defaultdict(float)
    by_cls = defaultdict(Pool)
    err = defaultdict(lambda: np.zeros(4))
    for i in rows:
        tab = vsrc.read_row_group(int(RG[i]), columns=PRICE, use_threads=False)
        x = np.column_stack([tab.column(k).to_numpy() for k in PRICE]).astype(np.float32)
        code = code_hash(x)
        sel = M[i] & (np.arange(len(x)) >= 99)
        if not sel.any():
            continue
        cls = int(t["market_class"][i])
        q = QUANTA[cls]
        y = np.asarray(Y[i][sel], np.float64)
        p = np.asarray(P[i][sel], np.float64)
        c = code[sel].tolist()
        s = np.array([spread.get(u, -1) for u in c], np.float64)
        known = s >= 1
        sq = np.where(known, s * q, 0.0)  # spread in target units
        pS, pD = p[:, 0] + p[:, 1], (p[:, 0] - p[:, 1]) / 2
        S, D = y[:, 0] + y[:, 1], (y[:, 0] - y[:, 1]) / 2
        interior = (y[:, 0] <= 0) & (y[:, 1] <= 0)
        eS = np.array([meanS.get(u, np.nan) for u in c])
        kS = np.isfinite(eS)
        e0 = np.array([mean0.get(u, 0.0) for u in c])
        e1 = np.array([mean1.get(u, 0.0) for u in c])

        def from_sd(s_, d_):
            return np.stack([s_ / 2 + d_, s_ / 2 - d_], 1)

        variants = {
            "model": p,
            "pS_to_minus_spread": from_sd(np.where(known, -sq, pS), pD),
            "pS_to_code_mean_S": from_sd(np.where(kS, eS, pS), pD),
            "pS_oracle_interior_only": from_sd(np.where(interior, S, pS), pD),
            "pS_oracle_all": from_sd(S, pD),
            "pD_oracle_all": from_sd(pS, D),
            "code_mean_only": np.stack([e0, e1], 1),
            "minus_spread_only": from_sd(np.where(known, -sq, 0.0), np.zeros(len(y))),
        }
        for beta in (0.25, 0.5, 0.75):
            variants[f"pS_blend_code_mean_S_{beta:g}"] = from_sd(np.where(kS, (1 - beta) * pS + beta * eS, pS), pD)
        for name, v in variants.items():
            pool.add(name, y, v)
            by_cls[cls].add(name, y, v)
        # metric-weighted variance decomposition per target (weights |y_k| clipped)
        yc = np.clip(y, -2, 2)
        S, D = yc[:, 0] + yc[:, 1], (yc[:, 0] - yc[:, 1]) / 2  # clipped components: yc0 = S/2 + D, yc1 = S/2 - D
        wS_ = np.abs(yc).sum(1)
        dec["wS_all"] += wS_.sum()
        dec["wS_S"] += (wS_ * S).sum()
        dec["wS_SS"] += (wS_ * S**2).sum()
        dec["wS_res_code"] += (wS_ * kS * (S - np.nan_to_num(eS)) ** 2 + wS_ * ~kS * (S - S.mean()) ** 2).sum()
        dec["wS_res_spread"] += (wS_ * known * (S + sq) ** 2 + wS_ * ~known * (S - S.mean()) ** 2).sum()
        for k in range(2):
            w = np.abs(yc[:, k])
            dec[f"w{k}"] += w.sum()
            dec[f"wy{k}"] += (w * yc[:, k]).sum()
            dec[f"wyy{k}"] += (w * yc[:, k] ** 2).sum()
            ek = e0 if k == 0 else e1
            dec[f"res_code{k}"] += (w * (yc[:, k] - ek) ** 2).sum()
            half = np.where(known, -sq / 2, 0.0)
            dec[f"half_s{k}"] += (w * half).sum()
            dec[f"half_ss{k}"] += (w * half**2).sum()
            dec[f"half_sy{k}"] += (w * half * yc[:, k]).sum()
            dec[f"wS2_{k}"] += (w * (S / 2) ** 2).sum()
            dec[f"wD2_{k}"] += (w * D**2).sum()
            dec[f"wSD_{k}"] += (w * (S / 2) * (D if k == 0 else -D)).sum()
            dec[f"wS_{k}"] += (w * S / 2).sum()
            dec[f"wD_{k}"] += (w * (D if k == 0 else -D)).sum()
            dec[f"w_interior{k}"] += (w * interior).sum()
            dec[f"w_known{k}"] += (w * known).sum()
        wi = np.abs(yc).sum(1)
        e = pS + sq
        err["interior"] += [(wi * interior * known).sum(), (wi * interior * known * e**2).sum(),
                            (wi * interior * known * (pS - S) ** 2).sum(), 0]
        err["all"] += [(wi * known).sum(), (wi * known * (pS - S) ** 2).sum(),
                       (wi * known * (-sq - S) ** 2).sum(), (wi * kS * (np.nan_to_num(eS) - S) ** 2).sum()]
    out = {"n_sequences": int(len(rows)), "wp": pool.wp(), "wp_by_class": {c: v.wp() for c, v in sorted(by_cls.items())}}
    decomposition = {}
    for k in range(2):
        W = dec[f"w{k}"]
        my = dec[f"wy{k}"] / W
        var = dec[f"wyy{k}"] / W - my**2
        mh, vh = dec[f"half_s{k}"] / W, dec[f"half_ss{k}"] / W - (dec[f"half_s{k}"] / W) ** 2
        cov = dec[f"half_sy{k}"] / W - mh * my
        mS, mD = dec[f"wS_{k}"] / W, dec[f"wD_{k}"] / W
        vS, vD = dec[f"wS2_{k}"] / W - mS**2, dec[f"wD2_{k}"] / W - mD**2
        cSD = dec[f"wSD_{k}"] / W - mS * mD
        decomposition[f"t{k}"] = {
            "weighted_variance": var,
            "r2_minus_half_spread_linear": cov**2 / (vh * var),
            "r2_code_conditional_mean_out_of_sample": 1 - dec[f"res_code{k}"] / W / var,
            "share_var_S_half": vS / var, "share_var_D": vD / var, "share_2cov_SD": 2 * cSD / var,
            "interior_weight_share": dec[f"w_interior{k}"] / W, "known_spread_weight_share": dec[f"w_known{k}"] / W,
        }
    mS_ = dec["wS_S"] / dec["wS_all"]
    vS_ = dec["wS_SS"] / dec["wS_all"] - mS_**2
    decomposition["S_sum"] = {
        "note": "S = clip(t0)+clip(t1) weighted by |clip t0|+|clip t1|; R2 of current-state predictors out of sample",
        "r2_minus_spread_exact": 1 - dec["wS_res_spread"] / dec["wS_all"] / vS_,
        "r2_code_conditional_mean": 1 - dec["wS_res_code"] / dec["wS_all"] / vS_,
    }
    out["variance_decomposition"] = decomposition
    out["model_s_error"] = {
        "interior_rows_weighted_rmse_pS_vs_minus_spread": float(np.sqrt(err["interior"][1] / err["interior"][0])),
        "all_known_rows_weighted_rmse_pS_vs_S": float(np.sqrt(err["all"][1] / err["all"][0])),
        "all_known_rows_weighted_rmse_minus_spread_vs_S": float(np.sqrt(err["all"][2] / err["all"][0])),
        "all_known_rows_weighted_rmse_code_mean_S_vs_S": float(np.sqrt(err["all"][3] / err["all"][0])),
    }
    (OUT / "target_state.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
