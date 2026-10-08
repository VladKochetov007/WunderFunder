#!/usr/bin/env python3
"""H-280 per-architecture breakdown on the fold0 excluded-468 panel.

Reuses the frozen campaign_score moments/pooled (campaign_score.py:85,119) and
wnn.evaluation.paired_component_bootstrap (src/wnn/evaluation.py:89). Same 5-fold
component split as blend_lab_analysis.py (seed 271828). Only the 468 evaluation
rows are loaded from each array.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CAMP = ROOT / "research/campaign_2026-10-06/bandit_continuation"
E529 = CAMP / "continuous_execution_E529"
OUT = CAMP / "blend_lab_H274"
RESULT_JSON = OUT / "blend_lab_architecture_20261008.json"
SEED = 271828
N_BOOT = 2000
N_FOLDS = 5
WARMUP = 99
CLIP = 2.0
WEIGHTS = (0.0, 0.1, 0.25, 0.5)

sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
from wnn.evaluation import paired_component_bootstrap  # noqa: E402

t0 = time.time()
split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
EVAL = [int(g) for g in split["evaluation"]["valid"]]
COMP_OF = {int(i["row_group"]): int(i["component"]) for i in split["sequences"]["valid"]}
COMPS = np.asarray([COMP_OF[g] for g in EVAL], dtype=np.int64)
UNIQ = np.unique(COMPS)
EVAL_IDX = np.asarray(EVAL, dtype=np.int64)
targets = scorer.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
Y = np.asarray(targets["y"][EVAL_IDX], dtype=np.float32)
MASK = np.asarray(targets["mask"][EVAL_IDX], dtype=bool) & (np.arange(20000) >= WARMUP)[None, :]

# Same fold assignment as blend_lab_analysis.py.
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(UNIQ))
fold_of_comp = {int(UNIQ[p]): i % N_FOLDS for i, p in enumerate(perm)}
FOLD = np.asarray([fold_of_comp[int(c)] for c in COMPS])

H233 = E529 / "exports/H233_C1S3_E1_VALID_V2"
H272D = E529 / "exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1"
H272S4 = E529 / "exports/H272_PARENT_V2_CLEANFOLD_VALID"
E540 = CAMP / "numerical_embedding_E540/kaggle_v12_root_approved_20261008_r1/downloads/provider_output/results/e540-v12-fixed-checkpoint-export-20261008/artifacts/cache"
# name -> (path, kind, family, trunk, cost class)
SPEC = {
    "R0_c20261003_GRU256x2": (ROOT / "cache/pred_c20261003_combo_s3_f0_valid.npy", "full", "GRU256", "T_R0", "full extra GRU256x2 trunk"),
    "H272S4_parent": (H272S4 / "parent.npy", "full", "GRU256", "T_S4", "full extra GRU256x2 trunk"),
    "H272S4_hidden_only_readout": (H272S4 / "hidden_only.npy", "full", "GRU256", "T_S4", "full extra GRU256x2 trunk"),
    "H272S4_multiplicative_readout": (H272S4 / "multiplicative.npy", "full", "GRU256", "T_S4", "full extra GRU256x2 trunk"),
    "H233_hidden_only": (H233 / "pred_hidden_only.npy", "full", "GRU256", "T_C1", "trunk T_C1 (shared)"),
    "H233_additive": (H233 / "pred_additive.npy", "full", "GRU256", "T_C1", "trunk T_C1 (shared)"),
    "H233_conditional_residual": (H233 / "pred_conditional_residual.npy", "full", "GRU256", "T_C1", "trunk T_C1 (shared)"),
    "H233_multiplicative": (H233 / "pred_multiplicative.npy", "full", "GRU256", "T_C1", "trunk T_C1 (shared)"),
    "H272d_hidden_only": (H272D / "pred_hidden_only.npy", "full", "GRU256", "T_C1", "trunk T_C1 (shared)"),
    "H272d_additive": (H272D / "pred_additive.npy", "full", "GRU256", "T_C1", "trunk T_C1 (shared)"),
    "H272d_conditional_residual": (H272D / "pred_conditional_residual.npy", "full", "GRU256", "T_C1", "trunk T_C1 (shared)"),
    "H272d_multiplicative": (H272D / "pred_multiplicative.npy", "full", "GRU256", "T_C1", "trunk T_C1 (shared)"),
    "H232_GRU384_v7b": (CAMP / "continuous_execution_E529/downloads/h232_v7b_ep3_export_r8/h232_v7b_ep3_export_source_r2/cache/pred_h232_v7b_ep3_export_only_r2_20261008_valid.npy", "full", "GRU384", "T_H232", "full extra GRU384 trunk"),
    "H218_block_memory_s7": (ROOT / "cache/pred_h218_block_memory_cleanfold0_seed7_ep3_fp32_eval468_v1_valid.npy", "subset", "block-memory attention", "T_H218", "full extra attention trunk"),
    "E540_v8_linear_GRU128": (E540 / "pred_e540v8-linear-s540-fp32-eval_valid.npy", "subset", "E540 GRU128 linear", "T_E540v8", "small extra GRU128 trunk"),
    "E540_v9_PLE_GRU128": (E540 / "pred_e540v9-ple-s540-fp32-eval_valid.npy", "subset", "E540 GRU128 PLE", "T_E540v9", "small extra GRU128 trunk"),
}
FAMILY_OF = {n: v[2] for n, v in SPEC.items()}
G256 = [n for n, v in SPEC.items() if v[2] == "GRU256"]
C1_HEADS = [n for n in G256 if n.startswith("H233_") or n.startswith("H272d_")]
S4_MEMBERS = [n for n in G256 if n.startswith("H272S4_") or n.startswith("R0_")]

P, S_of = {}, {}
for name, (path, kind, *_r) in SPEC.items():
    arr = np.load(path, mmap_mode="r")
    if kind == "full":
        assert arr.shape == (1873, 20000, 2)
        P[name] = np.asarray(arr[EVAL_IDX], dtype=np.float32)
    else:
        assert arr.shape == (468, 20000, 2)
        P[name] = np.asarray(arr, dtype=np.float32)
    S = np.zeros((468, 2, 6))
    for i in range(468):
        for k in range(2):
            S[i, k] = scorer.moments(Y[i, :, k], P[name][i, :, k], MASK[i])
    S_of[name] = S
print("loaded", len(P), f"t={time.time()-t0:.0f}s", flush=True)


def wp(S):
    return scorer.pooled(S)


def blend_S(pa, pb, w, seqs):
    out = np.zeros((len(seqs), 2, 6))
    for j, i in enumerate(seqs):
        for k in range(2):
            out[j, k] = scorer.moments(Y[i, :, k], (1 - w) * pa[i, :, k] + w * pb[i, :, k], MASK[i])
    return out


# Full-panel descriptive: standalone WP and correlation with the best GRU256 by full-panel WP.
standalone = {n: wp(S_of[n])["wp"] for n in SPEC}
best256_full = max(G256, key=lambda n: standalone[n])

# Metric-weighted correlation over selected rows, vs best GRU256 (full-panel descriptive).
sel_i, sel_t = np.nonzero(MASK)
Yc = np.clip(Y[sel_i, sel_t, :], -CLIP, CLIP).astype(np.float64)
Wt = np.abs(Yc)
corr_vs_best = {}
for name in SPEC:
    corr_vs_best[name] = {}
    for k in range(2):
        a = np.clip(P[name][sel_i, sel_t, k], -CLIP, CLIP).astype(np.float64)
        b = np.clip(P[best256_full][sel_i, sel_t, k], -CLIP, CLIP).astype(np.float64)
        w = Wt[:, k]
        ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
        cov = (w * (a - ma) * (b - mb)).sum() / w.sum()
        va = (w * (a - ma) ** 2).sum() / w.sum()
        vb = (w * (b - mb) ** 2).sum() / w.sum()
        corr_vs_best[name][f"t{k}"] = float(cov / np.sqrt(va * vb))

# Fold-wise conditional gain. Best GRU256 chosen per fold on training folds among all
# GRU256-family members; the family representative is chosen per fold among that family's
# members on training folds; blend weight chosen per fold on training folds.
fam_results = {}
families = {
    "GRU256 second trunk (S4 parent/readouts + R0)": S4_MEMBERS,
    "readout heads on C1 trunk (non-best)": C1_HEADS,
    "GRU384 (H232)": ["H232_GRU384_v7b"],
    "block-memory attention (H218)": ["H218_block_memory_s7"],
    "E540 GRU128 linear": ["E540_v8_linear_GRU128"],
    "E540 GRU128 PLE": ["E540_v9_PLE_GRU128"],
}
TCN_NOTE = "TCN (H264): member arrays not on disk; not included"
fold_out = {fam: [] for fam in families}
pooled_base = {fam: np.zeros((468, 2, 6)) for fam in families}
pooled_blend = {fam: np.zeros((468, 2, 6)) for fam in families}
picks_base = []
for f in range(N_FOLDS):
    tr_seq = np.where(FOLD != f)[0]
    te_seq = np.where(FOLD == f)[0]
    best = max(G256, key=lambda n: wp(S_of[n][tr_seq])["wp"])
    picks_base.append(best)
    for fam, members in families.items():
        cand = members
        if fam.startswith("readout heads"):
            cand = [m for m in C1_HEADS if m != best]
        rep = max(cand, key=lambda n: wp(S_of[n][tr_seq])["wp"])
        # blend weight chosen on training folds (rep weight w on top of best GRU256)
        train_scores = {w: wp(blend_S(P[best], P[rep], w, tr_seq))["wp"] for w in WEIGHTS}
        w_star = max(train_scores, key=train_scores.get)
        pooled_base[fam][te_seq] = S_of[best][te_seq]
        pooled_blend[fam][te_seq] = blend_S(P[best], P[rep], w_star, te_seq)
        fold_out[fam].append({"fold": f, "rep": rep, "rep_train_wp": wp(S_of[rep][tr_seq])["wp"],
                              "w_cv": w_star, "w_train_wp": train_scores})

results = {}
for fam in families:
    base = wp(pooled_base[fam])
    blend = wp(pooled_blend[fam])
    bs = paired_component_bootstrap(pooled_blend[fam], pooled_base[fam], COMPS,
                                    n_boot=N_BOOT, seed=SEED, centered=True)
    results[fam] = {
        "pooled_best_gru256_heldout_wp": base["wp"],
        "pooled_blend_heldout_wp": blend["wp"],
        "conditional_gain_vs_best_gru256": blend["wp"] - base["wp"],
        "bootstrap": {k: v for k, v in bs.items()},
        "folds": fold_out[fam],
    }
    print(fam, round(blend["wp"] - base["wp"], 6), bs.get("ci95"), flush=True)

out = {
    "best_gru256_per_fold": picks_base,
    "best_gru256_full_panel_descriptive": best256_full,
    "tcn_note": TCN_NOTE,
    "standalone_full_panel_wp": standalone,
    "family_of": FAMILY_OF,
    "cost_class": {n: v[4] for n, v in SPEC.items()},
    "trunk": {n: v[3] for n, v in SPEC.items()},
    "corr_vs_best_gru256_full_panel_descriptive": corr_vs_best,
    "families": results,
    "runtime_seconds": time.time() - t0,
}
RESULT_JSON.write_text(json.dumps(out, indent=2, default=float) + "\n")
print("wrote", RESULT_JSON, f"t={time.time()-t0:.0f}s", flush=True)
