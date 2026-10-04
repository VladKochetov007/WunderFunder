"""cr_mature step 2/3: matured-label residual feedback on class-3 VALID sequences of the 4 clean folds.

Reference p = mean of cache/pred_c20260929_ft_since_s{3,4}_f{k}_valid.npy on fold-k evaluation rows. For a run k
(rows E_k .. E_{k+1}-1, one constant label) the residual is r_k = clip(Y_k) - beta * mean_run(clip(p)), skipping
Y_k = 0 (zero metric weight). At event s the feature is an EMA (rate alpha, alpha = 1: newest only) over runs
k <= s - L in run order; it applies to rows E_s .. E_{s+1}-1, so no row uses events after itself. p' = p + kappa *
feature on class-3 rows only. Label sources Y_k:
  true   true label at maturity L (upper bound under the timing rule; leaks where the label is not determined)
  rec    TRAIN mode-table label from the decoded path up to E_{k+L} (deployable): exact path key, then shorter
         path keys, then the summary key (scripts/cr_mature_lag.py tables)
  true_R / shuf_R  restricted to runs whose exact path key at L has a partner run in another fold-k sequence;
         shuf_R replaces the label by that partner's true label (same observed path up to E_{k+L}, different
         future). shuf_R ~ true_R means the gain is carried by the observed path; a drop means future information.
beta and (L, alpha, kappa) are cross-fitted: chosen on the other three folds, scored on the held fold.

    WNN_CPU_PCT=25 WNN_MEM_MAX=12G scripts/limited .venv/bin/python scripts/cr_mature_feedback.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numba as nb
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cr_mature_common import QUANTA, ROOT, Decoder, read_group  # noqa: E402
from cr_mature_lag import JS, event_keys, summary_keys  # noqa: E402
from wnn.evaluation import paired_component_bootstrap  # noqa: E402
from wnn.gmetric import wp_from_sums, wp_sums  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-01/creative/cr_mature"
MEMBERS = ("c20260929_ft_since_s3_f{k}", "c20260929_ft_since_s4_f{k}")
LS = (2, 4, 8, 12, 16, 24, 32, 48)
ALPHAS = (0.05, 0.2, 1.0)
KAPPAS = np.array([-0.25, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0])
VARIANTS = ("true", "rec", "true_R", "shuf_R")
QUICK = int(os.environ.get("CR_QUICK", "0"))  # > 0: only this many evaluation sequences per fold (smoke test)
_DEC = None


def process(args):
    group, y, p, m = args
    global _DEC
    if _DEC is None:
        _DEC = Decoder()
    x, a6, _ = read_group(pq.ParquetFile(ROOT / "datasets/valid.parquet"), group, labels=False)
    d = _DEC.decode(x, a6)
    rows = np.flatnonzero(d["E"])
    A, B, S = d["ask"][rows], d["bid"][rows], d["seg"][rows]
    kp = event_keys(A, B, rows, S, len(a6), False)
    ks = summary_keys(A, B, S)
    bounds = np.r_[rows, len(a6)]
    pc = np.clip(p, -2, 2)
    cs = np.vstack([np.zeros((1, 2)), np.cumsum(pc, 0)])
    pmean = (cs[bounds[1:]] - cs[bounds[:-1]]) / np.diff(bounds)[:, None]
    ytrue = y[rows].astype(np.float64)
    run_of_row = np.searchsorted(rows, np.arange(len(a6)), side="right") - 1  # -1 before the first event
    sc = np.flatnonzero(m)
    return {"group": group, "kp": kp, "ks": ks, "ytrue": ytrue, "pmean": pmean, "start": rows,
            "run_sc": run_of_row[sc], "y_sc": y[sc].astype(np.float64), "p_sc": p[sc].astype(np.float64)}


def load_tables():
    out = {}
    for var in ("kp", "ks"):
        z = np.load(OUT / f"lag_tables_{var}.npz")
        out[var] = {j: (z[f"k{j}"], z[f"l{j}"]) for j in JS}
    return out


def lookup(table, keys):
    tk, tl = table
    if len(tk) == 0:
        return np.full(len(keys), -1, np.int64)
    pos = np.minimum(np.searchsorted(tk, keys), len(tk) - 1)
    hit = (keys >= 0) & (tk[pos] == keys)
    return np.where(hit, tl[pos], -1)


def decode_label(code):
    return np.stack([code // 101 - 50, code % 101 - 50], 1).astype(np.float64) * QUANTA[3]


def rec_labels(seq, L, tables):
    """Reconstructed label (value units) per run from the path up to E_{k+L}; NaN where no table hits."""
    n = len(seq["start"])
    code = np.full(n, -1, np.int64)
    for j in [j for j in JS if j <= L][::-1]:
        if j < 2:
            break
        miss = code < 0
        code[miss] = lookup(tables["kp"][j], seq["kp"][JS.index(j)][miss])
    for j in [j for j in JS if j <= min(L, 10)][::-1]:
        miss = code < 0
        code[miss] = lookup(tables["ks"][j], seq["ks"][JS.index(j)][miss])
    lab = decode_label(np.maximum(code, 0))
    lab[code < 0] = np.nan
    return lab


@nb.njit(cache=True)
def feature(lab, pmean, beta, L, alpha, start_ok):
    """Per event s: EMA over runs k <= s - L of r_k = clip(Y_k) - beta * pmean_k (Y_k != 0, finite)."""
    n = lab.shape[0]
    out = np.zeros((n, 2))
    for t in range(2):
        e = 0.0
        seen = False
        for s in range(n):
            k = s - L
            if k >= 0 and start_ok[k]:
                v = lab[k, t]
                if np.isfinite(v) and v != 0.0:
                    r = min(max(v, -2.0), 2.0) - beta[t] * pmean[k, t]
                    e = r if not seen else (1 - alpha) * e + alpha * r
                    seen = True
            out[s, t] = e
    return out


@nb.njit(cache=True)
def partner_index(keys, sid):
    """Within each block of equal (shuffled) keys, the next run cyclically that belongs to another sequence."""
    n = len(keys)
    out = np.full(n, -1, np.int64)
    a = 0
    while a < n:
        b = a
        while b < n and keys[b] == keys[a]:
            b += 1
        if keys[a] >= 0 and b - a >= 2:
            for t in range(a, b):
                for u in range(1, b - a):
                    v = a + (t - a + u) % (b - a)
                    if sid[v] != sid[t]:
                        out[t] = v
                        break
        a = b
    return out


def seq_sums(y, q):
    return wp_sums(y.T, q.T)  # (2, 6)


def main() -> None:
    z = np.load(ROOT / "cache/valid_targets.npz")
    rg_all, cls_all = z["row_group"], z["market_class"]
    y_all, m_all = z["y"], z["mask"]  # load once: NpzFile re-reads the array on every key access
    pos = {int(g): i for i, g in enumerate(rg_all)}
    tables = load_tables()
    folds = {}
    for k in range(4):
        man = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json").read_text())
        comp = {int(s["row_group"]): int(s["component"]) for s in man["sequences"]["valid"]}
        groups = sorted(int(g) for g in man["evaluation"]["valid"])
        if QUICK:
            groups = groups[:QUICK]
        preds = [np.load(ROOT / f"cache/pred_{n.format(k=k)}_valid.npy", mmap_mode="r") for n in MEMBERS]
        base, c3 = {}, []
        for g in groups:
            i = pos[g]
            p = sum(np.asarray(pr[i], np.float64) for pr in preds) / len(preds)
            y, m = y_all[i].astype(np.float64), m_all[i]
            base[g] = wp_sums(y.T, p.T, m[None].repeat(2, 0))
            if cls_all[i] == 3:
                c3.append((g, y, p, m))
        with ProcessPoolExecutor(4) as ex:
            seqs = list(ex.map(process, c3, chunksize=4))
        folds[k] = {"groups": groups, "comp": comp, "base": base, "seqs": seqs}
        print("fold", k, "sequences", len(groups), "class3", len(seqs), flush=True)
    # beta per target from class-3 scored rows (|y| weights, clipped), cross-fitted
    bsum = {k: np.zeros((2, 2)) for k in folds}
    for k, f in folds.items():
        for s in f["seqs"]:
            yc, pc = np.clip(s["y_sc"], -2, 2), np.clip(s["p_sc"], -2, 2)
            w = np.abs(yc)
            bsum[k] += np.stack([(w * yc * pc).sum(0), (w * pc * pc).sum(0)])
    beta = {k: sum(bsum[j][0] for j in folds if j != k) / sum(bsum[j][1] for j in folds if j != k) for k in folds}
    print("beta", {k: v.round(3).tolist() for k, v in beta.items()}, flush=True)
    # partner (shuffled-future) labels: exact path key at L shared with a run of another fold-k sequence
    rng = np.random.default_rng(20261001)
    partner, cover = {}, {}
    for k, f in folds.items():
        for L in LS:
            if L not in JS:
                continue
            ji = JS.index(L)
            keys = np.concatenate([s["kp"][ji] for s in f["seqs"]])
            sid = np.concatenate([np.full(len(s["start"]), i) for i, s in enumerate(f["seqs"])])
            labs = np.concatenate([s["ytrue"] for s in f["seqs"]])
            order = np.lexsort((rng.random(len(keys)), keys))
            mate = partner_index(keys[order], sid[order])
            out = np.full((len(keys), 2), np.nan)
            has = mate >= 0
            out[order[has]] = labs[order[mate[has]]]
            wr = np.minimum(np.abs(labs), 2.0).sum(1)
            cover[k, L] = float(wr[has[np.argsort(order)]].sum() / wr.sum())
            partner[k, L] = np.split(out, np.cumsum([len(s["start"]) for s in f["seqs"]])[:-1])
    # sums: S[k][(variant, L, alpha)] -> (n_kappa, 2, 6) for class-3 rows of fold k; per-seq kept for the bootstrap
    res = {}
    for k, f in folds.items():
        base_c3 = sum(f["base"][s["group"]] for s in f["seqs"])
        other = sum(f["base"].values()) - base_c3
        res[k] = {"other": other, "base_c3": base_c3, "S": {}}
        for vi, var in enumerate(VARIANTS):
            for L in LS:
                if var in ("true_R", "shuf_R") and L not in JS:
                    continue
                for alpha in ALPHAS:
                    acc = np.zeros((len(KAPPAS), 2, 6))
                    for si, s in enumerate(f["seqs"]):
                        if var == "true":
                            lab = s["ytrue"]
                        elif var == "rec":
                            lab = rec_labels(s, L, tables)
                        else:
                            mate = partner[k, L][si]
                            lab = np.where(np.isfinite(mate), s["ytrue"] if var == "true_R" else mate, np.nan)
                        ok = s["start"] >= 99
                        feat = feature(lab, s["pmean"], beta[k], L, alpha, ok)
                        rr = s["run_sc"]
                        fr = np.where(rr[:, None] >= 0, feat[np.maximum(rr, 0)], 0.0)
                        for a, kap in enumerate(KAPPAS):
                            acc[a] += seq_sums(s["y_sc"], s["p_sc"] + kap * fr)
                    res[k]["S"][var, L, alpha] = acc
            print("fold", k, "variant", var, "done", flush=True)
    k0 = int(np.flatnonzero(KAPPAS == 0.0)[0])

    def pooled(k, cfg, a):
        return float(wp_from_sums(res[k]["other"] + res[k]["S"][cfg][a]).mean())

    report = {"LS": LS, "alphas": ALPHAS, "kappas": KAPPAS.tolist(), "beta": {k: v.tolist() for k, v in beta.items()},
              "base_pooled": {k: pooled(k, ("true", LS[0], ALPHAS[0]), k0) for k in folds}, "variants": {},
              "partner_coverage_label_w": {f"{k}_{L}": v for (k, L), v in cover.items()}}
    for var in VARIANTS:
        rep = {"by_L": {}}
        for L in LS:
            if (var, L, ALPHAS[0]) not in res[0]["S"]:
                continue
            # cross-fit (alpha, kappa) at fixed L
            held = []
            for k in folds:
                score = {(al, a): np.mean([pooled(j, (var, L, al), a) - pooled(j, (var, L, al), k0) for j in folds if j != k])
                         for al in ALPHAS for a in range(len(KAPPAS))}
                al, a = max(score, key=score.get)
                held.append({"fold": k, "alpha": al, "kappa": float(KAPPAS[a]),
                             "delta": pooled(k, (var, L, al), a) - pooled(k, (var, L, al), k0)})
            # in-sample best per fold (for reference)
            oracle = [max(pooled(k, (var, L, al), a) - pooled(k, (var, L, al), k0) for al in ALPHAS for a in range(len(KAPPAS)))
                      for k in folds]
            rep["by_L"][str(L)] = {"crossfit": held, "crossfit_mean": float(np.mean([h["delta"] for h in held])),
                                   "insample_best_mean": float(np.mean(oracle))}
            print(var, L, round(rep["by_L"][str(L)]["crossfit_mean"], 6), [round(h["delta"], 5) for h in held],
                  "insample", round(float(np.mean(oracle)), 6), flush=True)
        report["variants"][var] = rep
    # bootstrap for the deployable variant at its cross-fitted L (L also chosen on other folds) and true at L=32
    boot = {}
    for var, fixed_L in (("rec", None), ("true", 32), ("true", 8)):
        cand_all, ref_all, comp_all, is_c3 = [], [], [], []
        picks = []
        for k, f in folds.items():
            grid = LS if fixed_L is None else (fixed_L,)
            score = {(L, al, a): np.mean([pooled(j, (var, L, al), a) - pooled(j, (var, L, al), k0) for j in folds if j != k])
                     for L in grid for al in ALPHAS for a in range(len(KAPPAS))}
            L, al, a = max(score, key=score.get)
            picks.append({"fold": k, "L": L, "alpha": al, "kappa": float(KAPPAS[a])})
            cand = dict(f["base"])
            for s in f["seqs"]:
                lab = s["ytrue"] if var == "true" else rec_labels(s, L, tables)
                feat = feature(lab, s["pmean"], beta[k], L, al, s["start"] >= 99)
                rr = s["run_sc"]
                fr = np.where(rr[:, None] >= 0, feat[np.maximum(rr, 0)], 0.0)
                cand[s["group"]] = seq_sums(s["y_sc"], s["p_sc"] + KAPPAS[a] * fr)
            for g in f["groups"]:
                cand_all.append(cand[g])
                ref_all.append(f["base"][g])
                comp_all.append(f["comp"][g])
                is_c3.append(cls_all[pos[g]] == 3)
        cand_all, ref_all, is_c3 = np.array(cand_all), np.array(ref_all), np.array(is_c3)
        bs = paired_component_bootstrap(cand_all, ref_all, np.array(comp_all), n_boot=2000)
        d3 = float(wp_from_sums(cand_all[is_c3].sum(0)).mean() - wp_from_sums(ref_all[is_c3].sum(0)).mean())
        bs3 = paired_component_bootstrap(cand_all[is_c3], ref_all[is_c3], np.array(comp_all)[is_c3], n_boot=2000)
        boot[f"{var}_L{fixed_L or 'xfit'}"] = {"picks": picks, "pooled_all_folds": bs, "class3_only": bs3,
                                               "class3_delta": d3}
        print(var, fixed_L, picks, round(bs["mean_diff"], 6), bs["ci95"], "class3", round(d3, 5), bs3["ci95"], flush=True)
    report["bootstrap"] = boot
    (OUT / "feedback_report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print("wrote", OUT / "feedback_report.json")


if __name__ == "__main__":
    main()
