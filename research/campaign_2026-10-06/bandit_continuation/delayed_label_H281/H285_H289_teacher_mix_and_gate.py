#!/usr/bin/env python3
"""H-285 teacher-mix proposals and H-289 threshold shrink gate, fold0 468 panel.

H-285: 8 fold0-clean teacher members (families_valid.npy, 8 members in source order) on the
468 eval groups. Current mix = source coefficients (sum 0.8125). Caruana greedy ensemble
selection with replacement, objective = official WP of the mix; proposals use full-panel
greedy (P1) and greedy with a decorrelation penalty (P2). Component-CV: greedy on training
folds, scored on held-out folds. Mixes are scaled to sum 0.8125 before scoring.
H-289: threshold shrink on H272S4 hidden-only: |pred| < T -> alpha*pred, T = q-quantile of
|pred| on training folds; (q, alpha) chosen on training folds, scored on held-out folds.
Labels enter only at fit/selection time. Shrink and mix weights are fixed before scoring.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CAMP = ROOT / "research/campaign_2026-10-06/bandit_continuation"
E529 = CAMP / "continuous_execution_E529"
TEA = ROOT / "cache/bd26_h230_fold0_teacher_family_all_rows_v1"
OUT = CAMP / "delayed_label_H281"
SEED = 271828
N_FOLDS = 5
CLIP = 2.0
WARMUP = 99
COEF_SUM = 0.8125
t0c = time.time()
spec = importlib.util.spec_from_file_location("campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
sys.path.insert(0, str(ROOT / "src"))
from wnn.evaluation import paired_component_bootstrap  # noqa: E402

meta = json.loads((TEA / "metadata.json").read_text())
NAMES = ["strong", "full_finetuned", "raw_wmse", "samefold_baseline"]
FAMS = NAMES
split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
EVAL = [int(g) for g in split["evaluation"]["valid"]]
IDX = np.asarray(EVAL)
COMP_OF = {int(i["row_group"]): int(i["component"]) for i in split["sequences"]["valid"]}
COMPS = np.asarray([COMP_OF[g] for g in EVAL])
UNIQ = np.unique(COMPS)
perm = np.random.default_rng(SEED).permutation(len(UNIQ))
FOLD = np.asarray([{int(UNIQ[p]): i % N_FOLDS for i, p in enumerate(perm)}[int(c)] for c in COMPS])

targets = scorer.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
Y = np.asarray(targets["y"][IDX], dtype=np.float64)
MASK = np.asarray(targets["mask"][IDX], dtype=bool) & (np.arange(20000) >= WARMUP)[None, :]
FAMRAW = np.load(TEA / "families_valid.npy", mmap_mode="r")
# Channel layout (verified): channel = family_index * 2 + target; 4 families, 2 targets.
FAMS = ["strong", "full_finetuned", "raw_wmse", "samefold_baseline"]
FAMP = np.stack([np.asarray(FAMRAW[IDX][..., 2 * f:2 * f + 2], dtype=np.float32) for f in range(4)], axis=-1)  # (468,20000,2,4)
del FAMRAW
MSRC = np.asarray(np.load(TEA / "main_source_valid.npy", mmap_mode="r")[IDX], dtype=np.float32)
S4 = np.asarray(np.load(E529 / "exports/H272_PARENT_V2_CLEANFOLD_VALID/hidden_only.npy", mmap_mode="r")[IDX], dtype=np.float32)
SEL_I, SEL_T = np.nonzero(MASK)
Ysel = np.clip(Y[SEL_I, SEL_T, :], -CLIP, CLIP)
Wsel = np.abs(Ysel)

coef_list = meta["source_coefficients"]
coef = np.zeros(4)
for c in coef_list:
    coef[FAMS.index(c["family"])] += c["coefficient"]
print("family coefficients", dict(zip(FAMS, coef.round(4).tolist())), "sum", coef.sum(), flush=True)
# Check: sum_f coef_f * family_f must reproduce the source mix (first 5 seqs).
recon = np.tensordot(FAMP[:5].astype(np.float64), coef, axes=([3], [0]))
gap = float(np.max(np.abs(recon - MSRC[:5].astype(np.float64))))
print("family reconstruction max abs gap (first 5 seqs):", gap, flush=True)


def moments_seqs(pred, seqs):
    """pred is aligned to seqs: pred[j] belongs to sequence seqs[j]."""
    S = np.zeros((len(seqs), 2, 6))
    for j, i in enumerate(seqs):
        for k in range(2):
            S[j, k] = scorer.moments(Y[i, :, k], pred[j, :, k], MASK[i])
    return S


def wp(S):
    return scorer.pooled(S)["wp"]


def mix_pred(w, seqs):
    """w: (8,) weights, already summing to COEF_SUM; returns (len(seqs),20000,2)."""
    return np.tensordot(FAMP[seqs].astype(np.float64), w, axes=([3], [0])).astype(np.float32)


def greedy(seqs, steps=10, penalty=0.0, cur_sel=None):
    """Caruana greedy with replacement on training sequences; returns integer counts."""
    counts = np.zeros(4, dtype=int)
    sub = FAMP[seqs]
    running = np.zeros((len(seqs), 20000, 2), dtype=np.float64)
    for step in range(steps):
        best, best_val = None, -np.inf
        for j in range(4):
            cand = (running + sub[..., j]) / (step + 1)
            val = wp(moments_seqs(cand.astype(np.float32), seqs))
            if penalty:
                val -= penalty * cand_corr_to_mix(cand, running, seqs)
            if val > best_val:
                best, best_val = j, val
        counts[best] += 1
        running = running + sub[..., best]
    return counts


def cand_corr_to_mix(cand, running, seqs):
    """Mean residual-free prediction correlation between candidate and current mix (metric-weighted)."""
    if running.sum() == 0:
        return 0.0
    m = MASK[seqs]
    ii, tt = np.nonzero(m)
    a = cand[ii, tt, :].astype(np.float64)
    b = running[ii, tt, :].astype(np.float64)
    w = np.abs(np.clip(Y[np.asarray(seqs)[ii], tt, :], -CLIP, CLIP))
    cs = []
    for k in range(2):
        wa = (w[:, k] * a[:, k]).sum() / w[:, k].sum()
        wb = (w[:, k] * b[:, k]).sum() / w[:, k].sum()
        cov = (w[:, k] * (a[:, k] - wa) * (b[:, k] - wb)).sum()
        cs.append(cov / np.sqrt((w[:, k] * (a[:, k] - wa) ** 2).sum() * (w[:, k] * (b[:, k] - wb) ** 2).sum()))
    return float(np.mean(cs))


def weights_from_counts(counts):
    return COEF_SUM * counts / counts.sum()


all_seqs = np.arange(468)
res = {"members": NAMES, "coef_current": coef.tolist(), "coef_sum": float(coef.sum()),
       "family_reconstruction_gap_first5": gap}
# Per-family teacher WP and residual correlation (metric-weighted, selected rows).
member_S = {j: moments_seqs(FAMP[..., j], all_seqs) for j in range(4)}
res["member_wp"] = {NAMES[j]: scorer.pooled(member_S[j]) for j in range(4)}
FSEL = FAMP[SEL_I, SEL_T, :, :]  # (N,2,4)
corr = np.zeros((2, 4, 4))
for k in range(2):
    w = Wsel[:, k]
    sw = w.sum()
    x = np.clip(FSEL[:, k, :], -CLIP, CLIP).astype(np.float64)
    mu = (w[:, None] * x).sum(0) / sw
    xc = x - mu
    C = (w[:, None] * xc).T @ xc / sw
    d = np.sqrt(np.diag(C))
    corr[k] = C / np.outer(d, d)
res["pred_corr_t0"] = corr[0].tolist()
res["pred_corr_t1"] = corr[1].tolist()
print("member WP", {NAMES[j]: round(res["member_wp"][NAMES[j]]["wp"], 5) for j in range(4)}, flush=True)

# Current mix (scaled to COEF_SUM) on full panel and component-CV (no selection).
cur_pred = mix_pred(coef, all_seqs)
cur_S = moments_seqs(cur_pred, all_seqs)
cur_wp = scorer.pooled(cur_S)
res["current_mix_wp"] = cur_wp
print("current mix WP", cur_wp["wp"], flush=True)

# P1: full-panel greedy; P2: greedy with decorrelation penalty (exposed, descriptive).
c1 = greedy(all_seqs, steps=10)
c2 = greedy(all_seqs, steps=10, penalty=0.002)
w1, w2 = weights_from_counts(c1), weights_from_counts(c2)
for label, w in (("P1_greedy", w1), ("P2_greedy_decorr", w2)):
    S = moments_seqs(mix_pred(w, all_seqs), all_seqs)
    bs = paired_component_bootstrap(S, cur_S, COMPS, n_boot=2000, seed=SEED, centered=True)
    res[label] = {"coef": dict(zip(NAMES, w.round(5).tolist())), "wp": scorer.pooled(S), "bootstrap_vs_current": bs}
    print(label, round(scorer.pooled(S)["wp"], 6), bs["ci95"], flush=True)

# Component-CV of greedy selection (selection on training folds only).
cv_S = np.zeros((468, 2, 6))
cv_cur_S = np.zeros((468, 2, 6))
cv_counts = []
for f in range(N_FOLDS):
    tr = all_seqs[FOLD != f]
    te = all_seqs[FOLD == f]
    cnt = greedy(tr, steps=10)
    cv_counts.append(cnt.tolist())
    wf = weights_from_counts(cnt)
    cv_S[te] = moments_seqs(mix_pred(wf, te), te)
    cv_cur_S[te] = cur_S[te]
cv_wp = scorer.pooled(cv_S)
cv_bs = paired_component_bootstrap(cv_S, cur_S, COMPS, n_boot=2000, seed=SEED, centered=True)
res["cv_greedy"] = {"wp": cv_wp, "bootstrap_vs_current": cv_bs, "fold_counts": cv_counts}
print("CV greedy", round(cv_wp["wp"], 6), cv_bs["ci95"], flush=True)

# H-289: threshold shrink gate on H272S4 hidden-only, selection on training folds.
base_S = moments_seqs(S4, all_seqs)
base_wp = scorer.pooled(base_S)
res["h289_base_wp"] = base_wp
QS = [0.1, 0.25, 0.5]
ALPHAS = [0.0, 0.5, 0.8]
gate_S = np.zeros((468, 2, 6))
picks = []
for f in range(N_FOLDS):
    tr = all_seqs[FOLD != f]
    te = all_seqs[FOLD == f]
    best = None
    for q in QS:
        for a in ALPHAS:
            pred_tr = S4[tr].copy()
            for k in range(2):
                T = np.quantile(np.abs(S4[tr][:, :, k][MASK[tr]]), q)
                m = np.abs(pred_tr[:, :, k]) < T
                pred_tr[:, :, k] = np.where(m, a * pred_tr[:, :, k], pred_tr[:, :, k])
            score = wp(moments_seqs(pred_tr, tr))
            if best is None or score > best[0]:
                best = (score, q, a)
    _, q, a = best
    picks.append({"fold": f, "q": q, "alpha": a, "train_wp": best[0]})
    pred_te = S4[te].copy()
    for k in range(2):
        T = np.quantile(np.abs(S4[tr][:, :, k][MASK[tr]]), q)
        m = np.abs(pred_te[:, :, k]) < T
        pred_te[:, :, k] = np.where(m, a * pred_te[:, :, k], pred_te[:, :, k])
    gate_S[te] = moments_seqs(pred_te, te)
gate_wp = scorer.pooled(gate_S)
gate_bs = paired_component_bootstrap(gate_S, base_S, COMPS, n_boot=2000, seed=SEED, centered=True)
res["h289_gate"] = {"wp": gate_wp, "delta": gate_wp["wp"] - base_wp["wp"], "bootstrap_vs_base": gate_bs, "picks": picks}
print("H289 gate", round(gate_wp["wp"], 6), "delta", round(gate_wp["wp"] - base_wp["wp"], 6), gate_bs["ci95"], flush=True)

(OUT / "H285_H289_results_20261008.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
print("wrote", f"t={time.time()-t0c:.0f}s", flush=True)
