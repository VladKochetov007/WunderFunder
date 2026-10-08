#!/usr/bin/env python3
"""H-284: target-law identity audit (t0+t1 vs feature spreads) and projection oracle, fold0 468 panel.

Identity search uses feature columns and targets at the same row t (diagnostic).
Projection oracle uses features and base predictions only, plus the per-row spread
from the identified feature pair; market class is taken from the label cache, so the
oracle is not servable as written.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CAMP = ROOT / "research/campaign_2026-10-06/bandit_continuation"
E529 = CAMP / "continuous_execution_E529"
OUT = CAMP / "delayed_label_H281"
CLIP = 2.0
WARMUP = 99
STRIDE = 10
t0c = time.time()
spec = importlib.util.spec_from_file_location("campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
sys.path.insert(0, str(ROOT / "src"))
from wnn.evaluation import paired_component_bootstrap  # noqa: E402

split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
EVAL = [int(g) for g in split["evaluation"]["valid"]]
IDX = np.asarray(EVAL)
COMP_OF = {int(i["row_group"]): int(i["component"]) for i in split["sequences"]["valid"]}
COMPS = np.asarray([COMP_OF[g] for g in EVAL])
targets = scorer.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
MC = np.asarray(targets["market_class"][IDX])
MASK = np.asarray(targets["mask"][IDX], dtype=bool) & (np.arange(20000) >= WARMUP)[None, :]
Y = np.asarray(targets["y"][IDX], dtype=np.float64)

pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
names = pf.schema_arrow.names
FEAT = [n for n in names if n.startswith("i0_") or n.startswith("i1_") or (n.startswith("a") and n[1:].isdigit())]
assert len(FEAT) == 112

# Stride-subsampled rows per class for identity search (exact check on all rows later).
rows = {c: [] for c in range(4)}
for i, g in enumerate(EVAL):
    tbl = pf.read_row_group(g, columns=FEAT + ["t0", "t1"])
    F = np.column_stack([tbl.column(c).to_numpy().astype(np.float64) for c in FEAT])
    t = np.column_stack([tbl.column("t0").to_numpy(), tbl.column("t1").to_numpy()]).astype(np.float64)
    sel = np.arange(WARMUP, 20000, STRIDE)
    rows[int(MC[i])].append((F[sel].astype(np.float32), t[sel].astype(np.float32)))
print("loaded", f"t={time.time()-t0c:.0f}s", flush=True)

report = {"features": FEAT, "per_class": {}}
for c in range(4):
    Fc = np.concatenate([r[0] for r in rows[c]]).astype(np.float64)
    Tc = np.concatenate([r[1] for r in rows[c]]).astype(np.float64)
    out = {"n_rows": int(len(Fc))}
    Fm = Fc - Fc.mean(0)
    C = Fm.T @ Fm / len(Fm)
    sd = np.sqrt(np.diag(C)) + 1e-12
    for tname, tv in [("t0", Tc[:, 0]), ("t1", Tc[:, 1]), ("sum", Tc[:, 0] + Tc[:, 1]), ("diff", Tc[:, 0] - Tc[:, 1])]:
        tm = tv - tv.mean()
        sdt = tm.std() + 1e-12
        cvec = Fm.T @ tm / len(Fm)  # cov(F, target)
        r_single = cvec / (sd * sdt)
        j = int(np.argmax(np.abs(r_single)))
        # pair differences F_i - F_j: corr = (c_i - c_j) / sqrt(var_ij) / sdt
        var = np.diag(C)[:, None] + np.diag(C)[None, :] - 2 * C
        num = cvec[:, None] - cvec[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            rp = num / (np.sqrt(np.maximum(var, 1e-30)) * sdt)
        np.fill_diagonal(rp, 0.0)
        ii, jj = np.unravel_index(np.argmax(np.abs(rp)), rp.shape)
        out[tname] = {"best_single": FEAT[j], "best_single_corr": float(r_single[j]),
                      "best_diff": [FEAT[ii], FEAT[jj]], "best_diff_corr": float(rp[ii, jj])}
    # exact check for the best sum-diff candidate: residual of sum vs -(F_i - F_j)
    best = out["sum"]["best_diff"]
    a, b = FEAT.index(best[0]), FEAT.index(best[1])
    d = Fc[:, a] - Fc[:, b]
    resid = Tc[:, 0] + Tc[:, 1] + d
    out["sum_plus_diff_residual"] = {"pair": best, "max_abs": float(np.max(np.abs(resid))),
                                     "median_abs": float(np.median(np.abs(resid))),
                                     "sum_std": float(np.std(Tc[:, 0] + Tc[:, 1]))}
    report["per_class"][str(c)] = out
    print("class", c, json.dumps({k: out[k] for k in ("sum", "sum_plus_diff_residual")}), flush=True)

# Projection oracle on class 3 using the class-3 identity pair (diagnostic only).
pair3 = report["per_class"]["3"]["sum"]["best_diff"]
a3, b3 = FEAT.index(pair3[0]), FEAT.index(pair3[1])
S4 = np.asarray(np.load(E529 / "exports/H272_PARENT_V2_CLEANFOLD_VALID/hidden_only.npy", mmap_mode="r")[IDX], dtype=np.float64)
proj = S4.copy()
base_S = np.zeros((468, 2, 6)); new_S = np.zeros((468, 2, 6))
for i, g in enumerate(EVAL):
    if int(MC[i]) != 3:
        for k in range(2):
            base_S[i, k] = scorer.moments(Y[i, :, k], S4[i, :, k], MASK[i])
            new_S[i, k] = base_S[i, k]
        continue
    tbl = pf.read_row_group(g, columns=[FEAT[a3], FEAT[b3]])
    s_hat = -(tbl.column(0).to_numpy().astype(np.float64) - tbl.column(1).to_numpy().astype(np.float64))
    p = S4[i]
    resid = (p[:, 0] + p[:, 1]) - s_hat
    p2 = p.copy()
    p2[:, 0] -= resid / 2
    p2[:, 1] -= resid / 2
    proj[i] = p2
    for k in range(2):
        base_S[i, k] = scorer.moments(Y[i, :, k], S4[i, :, k], MASK[i])
        new_S[i, k] = scorer.moments(Y[i, :, k], p2[:, k], MASK[i])
base_wp = scorer.pooled(base_S)["wp"]
new_wp = scorer.pooled(new_S)["wp"]
bs = paired_component_bootstrap(new_S, base_S, COMPS, n_boot=2000, seed=271828, centered=True)
report["projection_oracle_class3"] = {"pair": pair3, "base_wp": base_wp, "projected_wp": new_wp,
                                      "delta": new_wp - base_wp, "bootstrap": bs}
print("oracle delta", new_wp - base_wp, bs["ci95"], flush=True)
(OUT / "H284_identity_audit_20261008.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
print("wrote", f"t={time.time()-t0c:.0f}s", flush=True)
