#!/usr/bin/env python3
"""H-274/H-275/H-276 blend lab on the fold0 excluded-468 development panel.

Loads only the 468 evaluation sequences from each prediction array and only the
matching target/mask rows. Metric and bootstrap code is reused from the frozen
graders: campaign_score.moments/pooled (campaign_score.py:85,119) and
wnn.evaluation.paired_component_bootstrap (src/wnn/evaluation.py:89).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CAMP = ROOT / "research/campaign_2026-10-06/bandit_continuation"
E529 = CAMP / "continuous_execution_E529"
OUT = CAMP / "blend_lab_H274"
PLOTS = OUT / "plots"
RESULT_JSON = OUT / "blend_lab_results_20261008.json"
SEED = 271828
N_BOOT = 2000
N_FOLDS = 5
WARMUP = 99
CLIP = 2.0

sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
from wnn.evaluation import paired_component_bootstrap  # noqa: E402
from wnn.gmetric import global_wp  # noqa: E402

t_start = time.time()
split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
EVAL = [int(g) for g in split["evaluation"]["valid"]]
assert len(EVAL) == 468
COMP_OF = {int(item["row_group"]): int(item["component"]) for item in split["sequences"]["valid"]}
COMPS = np.asarray([COMP_OF[g] for g in EVAL], dtype=np.int64)
UNIQ = np.unique(COMPS)
assert len(UNIQ) == 141, len(UNIQ)
EVAL_IDX = np.asarray(EVAL, dtype=np.int64)

targets = scorer.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
Y = np.asarray(targets["y"][EVAL_IDX], dtype=np.float32)  # (468, 20000, 2)
MASK = np.asarray(targets["mask"][EVAL_IDX], dtype=bool)
MASK &= (np.arange(Y.shape[1]) >= WARMUP)[None, :]
assert Y.shape == (468, 20000, 2)


def report_value(path: Path, getter):
    return getter(json.loads(path.read_text()))


def _primary_reported(path: Path, mode: str, w: float):
    """Weight 0 is the shared hidden-only control; other weights live under the mode key."""
    d = json.loads(path.read_text())["primary_vs_trained_hidden_only"]
    rows = d["additive"] if mode == "hidden_only" else d[mode]
    for row in rows:
        if abs(row["weight_on_readout"] - w) < 1e-12:
            return row["score"]["wp"]
    return None


def h233_reported(mode: str, w: float):
    return _primary_reported(E529 / "H233_FIXED_EP1_RESULT_20261008_R1.json", mode, w)


def h272d_reported(mode: str, w: float):
    return _primary_reported(E529 / "H272_FIXED_EP1_COMPARISON_RESULT_20261008_R2.json", mode, w)


def h272s4_reported(mode: str, w: float):
    d = json.loads((E529 / "H272_S4_CLEANFOLD_V2_GRADE_RESULT_20261008_R1.json").read_text())
    for row in d["fixed_comparisons_all_reported"][mode]:
        if abs(row["alpha_on_readout"] - w) < 1e-12:
            return row["score"]["wp"]
    return None


def blend_reported(path: Path, key: str, alpha: float):
    d = json.loads(path.read_text())
    for row in d["predeclared_blends_all_reported"]:
        if abs(row[key] - alpha) < 1e-12:
            return row["score"]["wp"]
    return None


H233_DIR = E529 / "exports/H233_C1S3_E1_VALID_V2"
H272D_DIR = E529 / "exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1"
H272S4_DIR = E529 / "exports/H272_PARENT_V2_CLEANFOLD_VALID"
H218_JSON = CAMP / "attention_training_H218/H218_FOLD0_REFERENCE_BLEND_20261008_R1.json"
H232_JSON = CAMP / "wide_baseline_heterogeneous_H232/H232_FIXED_REFERENCE_BLEND_RESULT_20261008_R2.json"
H232_NPY = E529 / "downloads/h232_v7b_ep3_export_r8/h232_v7b_ep3_export_source_r2/cache/pred_h232_v7b_ep3_export_only_r2_20261008_valid.npy"
E540_DIR = CAMP / "numerical_embedding_E540/kaggle_v12_root_approved_20261008_r1/downloads/provider_output/results/e540-v12-fixed-checkpoint-export-20261008/artifacts/cache"

# name, path, trunk, full (1873 rows) or subset (468 rows), reported WP, membership/source note
MODELS = [
    ("R0_c20261003_GRU256x2", ROOT / "cache/pred_c20261003_combo_s3_f0_valid.npy", "trunk_R0", "full",
     blend_reported(H218_JSON, "alpha_on_h218", 0.0), "H218 JSON reference lineage: fit train 10535 + valid 1405, overlap 0"),
    ("H272S4_parent", H272S4_DIR / "parent.npy", "trunk_H272S4", "full",
     0.6810293773471175, "H272 S4 cleanfold v2 grade membership: train 10535 + valid 1405, overlap 0"),
    ("H272S4_hidden_only_readout", H272S4_DIR / "hidden_only.npy", "trunk_H272S4", "full",
     h272s4_reported("hidden_only", 1.0), "same S4 trunk as parent (head)"),
    ("H272S4_multiplicative_readout", H272S4_DIR / "multiplicative.npy", "trunk_H272S4", "full",
     h272s4_reported("multiplicative", 1.0), "same S4 trunk as parent (head)"),
    ("H233_hidden_only", H233_DIR / "pred_hidden_only.npy", "trunk_H233", "full",
     h233_reported("hidden_only", 0.0), "H233 fixed-ep1 membership: train 10535 + valid 1405, overlap 0"),
    ("H233_additive", H233_DIR / "pred_additive.npy", "trunk_H233", "full",
     h233_reported("additive", 1.0), "H233 head"),
    ("H233_conditional_residual", H233_DIR / "pred_conditional_residual.npy", "trunk_H233", "full",
     h233_reported("conditional_residual", 1.0), "H233 head"),
    ("H233_multiplicative", H233_DIR / "pred_multiplicative.npy", "trunk_H233", "full",
     h233_reported("multiplicative", 1.0), "H233 head"),
    ("H272d_hidden_only", H272D_DIR / "pred_hidden_only.npy", "trunk_H272d", "full",
     h272d_reported("hidden_only", 0.0), "H272 direct-label fixed-ep1 membership: train 10535 + valid 1405, overlap 0"),
    ("H272d_additive", H272D_DIR / "pred_additive.npy", "trunk_H272d", "full",
     h272d_reported("additive", 1.0), "H272 direct-label head"),
    ("H272d_conditional_residual", H272D_DIR / "pred_conditional_residual.npy", "trunk_H272d", "full",
     h272d_reported("conditional_residual", 1.0), "H272 direct-label head"),
    ("H272d_multiplicative", H272D_DIR / "pred_multiplicative.npy", "trunk_H272d", "full",
     h272d_reported("multiplicative", 1.0), "H272 direct-label head"),
    ("H232_GRU384_v7b", H232_NPY, "trunk_H232", "full",
     blend_reported(H232_JSON, "alpha_on_h232", 1.0), "H232 reference-blend membership: train 10535 + valid 1405, overlap 0"),
    ("H218_block_memory_s7", ROOT / "cache/pred_h218_block_memory_cleanfold0_seed7_ep3_fp32_eval468_v1_valid.npy", "trunk_H218", "subset468",
     0.6624036459470397, "H218 candidate sidecar: fold0 clean export, row order = evaluation list"),
    ("E540_v8_linear_GRU128", E540_DIR / "pred_e540v8-linear-s540-fp32-eval_valid.npy", "trunk_E540v8", "subset468",
     0.6361859352, "E540 fixed-pair md table (ep0 subset, excluded 468)"),
    ("E540_v9_PLE_GRU128", E540_DIR / "pred_e540v9-ple-s540-fp32-eval_valid.npy", "trunk_E540v9", "subset468",
     0.6254095463, "E540 fixed-pair md table (ep0 subset, excluded 468)"),
]
for name, path, *_ in MODELS:
    assert path.exists(), path


def load_eval_rows(path: Path, kind: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if kind == "full":
        assert arr.shape == (1873, 20000, 2), (path, arr.shape)
        out = np.asarray(arr[EVAL_IDX], dtype=np.float32)
    else:
        assert arr.shape == (468, 20000, 2), (path, arr.shape)
        out = np.asarray(arr, dtype=np.float32)
    assert np.isfinite(out).all()
    return out


def sequence_moments(pred: np.ndarray) -> np.ndarray:
    S = np.zeros((pred.shape[0], 2, 6), dtype=np.float64)
    for i in range(pred.shape[0]):
        for k in range(2):
            S[i, k] = scorer.moments(Y[i, :, k], pred[i, :, k], MASK[i])
    return S


def wp_of(S: np.ndarray) -> dict:
    return scorer.pooled(S)


P: dict[str, np.ndarray] = {}
S_of: dict[str, np.ndarray] = {}
inventory = []
for name, path, trunk, kind, reported, note in MODELS:
    pred = load_eval_rows(path, kind)
    P[name] = pred
    S = sequence_moments(pred)
    S_of[name] = S
    pooled = wp_of(S)
    rec = {
        "name": name, "trunk": trunk, "path": str(path.relative_to(ROOT)), "rows": kind,
        "membership_note": note, "wp_reported": reported,
        "wp_recomputed_scorer_pooled": pooled["wp"], "t0": pooled["t0"], "t1": pooled["t1"],
    }
    if kind == "full":
        full = global_wp(Y, pred, MASK)
        rec["wp_recomputed_gmetric_global_wp"] = float(full["wp"]) if isinstance(full, dict) else float(full)
    if reported is not None:
        rec["abs_diff_vs_reported"] = abs(pooled["wp"] - reported)
        rec["match_1e8"] = bool(abs(pooled["wp"] - reported) < 1e-8)
    inventory.append(rec)
    print(f"{name}: wp={pooled['wp']:.10f} reported={reported} t={time.time()-t_start:.0f}s", flush=True)

# Drop any model that fails the 1e-8 reproduction (none expected; rule kept explicit).
kept = [r["name"] for r in inventory if r.get("match_1e8", True)]
dropped = [r["name"] for r in inventory if not r.get("match_1e8", True)]
names = kept
print("kept", len(names), "dropped", dropped, flush=True)

# Component fold assignment (5 folds by dependence component, seeded).
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(UNIQ))
fold_of_comp = {int(UNIQ[p]): i % N_FOLDS for i, p in enumerate(perm)}
FOLD_OF_SEQ = np.asarray([fold_of_comp[int(c)] for c in COMPS])

# Selected rows (scored, step >= 99) flattened for stacking.
sel_i, sel_t = np.nonzero(MASK)
Yc = np.clip(Y[sel_i, sel_t, :], -CLIP, CLIP).astype(np.float64)  # (N, 2)
W = np.abs(Yc)  # metric weights per target
N = len(sel_i)
step_feat = sel_t.astype(np.float64) / 20000.0
print("selected rows", N, flush=True)

FEAT = {k: np.stack([np.clip(P[n][sel_i, sel_t, k], -CLIP, CLIP) for n in names], axis=1).astype(np.float64) for k in range(2)}
train_row_fold = FOLD_OF_SEQ[sel_i]

# Pair equal blends: raw mix then official clipping; precompute per-sequence moments.
pair_names = []
S_pair = {}
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        key = f"{names[a]}+{names[b]}"
        blend = 0.5 * P[names[a]] + 0.5 * P[names[b]]
        S_pair[key] = sequence_moments(blend)
        pair_names.append(key)
        del blend
print("pairs", len(pair_names), f"t={time.time()-t_start:.0f}s", flush=True)

# ---------- Stackers ----------
from scipy.optimize import nnls  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402


def fit_nnls(X, y, w):
    G = (X * w[:, None]).T @ X
    h = (X * w[:, None]).T @ y
    G += 1e-9 * np.trace(G) / X.shape[1] * np.eye(X.shape[1])
    L = np.linalg.cholesky(G)
    beta, _ = nnls(L.T, np.linalg.solve(L, h))
    return beta


def fit_ridge(X, y, w, lam=1e-3):
    G = (X * w[:, None]).T @ X
    h = (X * w[:, None]).T @ y
    return np.linalg.solve(G + lam * np.trace(G) / X.shape[1] * np.eye(X.shape[1]), h)


def sample_by_weight(rows, w, n, rs):
    p = w[rows] / w[rows].sum()
    return rows[rs.choice(len(rows), size=min(n, len(rows)), replace=True, p=p)]


def stacked_seq_pred(seqs, fn):
    """Predict all 20000 steps of the given sequences; fn(k, feature_block) -> values."""
    out = np.zeros((len(seqs), 20000, 2), dtype=np.float32)
    for k in range(2):
        for j0 in range(0, len(seqs), 16):
            chunk = seqs[j0:j0 + 16]
            block = np.stack([np.clip(P[n][chunk, :, k], -CLIP, CLIP) for n in names], axis=-1)
            out[j0:j0 + 16, :, k] = fn(k, block.reshape(-1, len(names)), chunk).reshape(len(chunk), 20000)
    return out


METHODS = ["best_single_train_selected", "best_equal_pair_train_selected", "nnls_stack",
           "ridge_stack", "mlp_stack", "residual_booster_on_best_single",
           "diagnostic_scaled_best_single_no_booster"]
S_method = {m: np.zeros((468, 2, 6)) for m in METHODS}
picks = []
rs = np.random.default_rng(SEED)
for f in range(N_FOLDS):
    test_seq = np.where(FOLD_OF_SEQ == f)[0]
    train_seq = np.where(FOLD_OF_SEQ != f)[0]
    tr = train_row_fold != f
    # best single and best equal pair by training-fold official WP
    single_wp = {n: wp_of(S_of[n][train_seq])["wp"] for n in names}
    best_single = max(single_wp, key=single_wp.get)
    pair_wp = {k: wp_of(S_pair[k][train_seq])["wp"] for k in pair_names}
    best_pair = max(pair_wp, key=pair_wp.get)
    S_method["best_single_train_selected"][test_seq] = S_of[best_single][test_seq]
    S_method["best_equal_pair_train_selected"][test_seq] = S_pair[best_pair][test_seq]
    picks.append({"fold": f, "best_single": best_single, "best_single_train_wp": single_wp[best_single],
                  "best_equal_pair": best_pair, "best_pair_train_wp": pair_wp[best_pair],
                  "test_sequences": int(len(test_seq))})
    print("fold", f, picks[-1], flush=True)

    # per-target learned stackers
    nnls_betas, ridge_betas, mlp_models, hgb_models, a_coef = {}, {}, {}, {}, {}
    best_idx = names.index(best_single)
    for k in range(2):
        X, y, w = FEAT[k][tr], Yc[tr, k], W[tr, k]
        nnls_betas[k] = fit_nnls(X, y, w)
        ridge_betas[k] = fit_ridge(X, y, w)
        Xs = np.column_stack([X, step_feat[tr]])
        rows = np.arange(len(y))
        pick = sample_by_weight(rows, w, 200_000, rs)
        mlp = MLPRegressor(hidden_layer_sizes=(16,), activation="tanh", alpha=1e-3,
                           learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                           random_state=SEED)
        mlp.fit(Xs[pick], y[pick])
        mlp_models[k] = mlp
        b = X[:, best_idx]
        a_coef[k] = float(np.sum(w * y * b) / np.sum(w * b * b))
        resid = y - a_coef[k] * b
        hgb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=200, random_state=SEED)
        hgb.fit(Xs, resid, sample_weight=w)
        hgb_models[k] = hgb

    picks[-1]["a_coef_per_target"] = [a_coef[0], a_coef[1]]

    def nnls_fn(k, block, chunk):
        return block @ nnls_betas[k]

    def ridge_fn(k, block, chunk):
        return block @ ridge_betas[k]

    def mlp_fn(k, block, chunk):
        steps = np.tile(np.arange(20000) / 20000.0, len(chunk))
        return mlp_models[k].predict(np.column_stack([block, steps]))

    def booster_fn(k, block, chunk):
        steps = np.tile(np.arange(20000) / 20000.0, len(chunk))
        base = block[:, best_idx]
        return a_coef[k] * base + hgb_models[k].predict(np.column_stack([block, steps]))

    def scaled_base_fn(k, block, chunk):
        return a_coef[k] * block[:, best_idx]

    for method, fn in [("nnls_stack", nnls_fn), ("ridge_stack", ridge_fn),
                       ("mlp_stack", mlp_fn), ("residual_booster_on_best_single", booster_fn),
                       ("diagnostic_scaled_best_single_no_booster", scaled_base_fn)]:
        pred_test = stacked_seq_pred(test_seq, fn)
        for local, i in enumerate(test_seq):
            for k in range(2):
                S_method[method][i, k] = scorer.moments(Y[i, :, k], pred_test[local, :, k], MASK[i])
        del pred_test
    print(f"fold {f} done t={time.time()-t_start:.0f}s", flush=True)

# ---------- Pooled held-out scores and paired component bootstrap ----------
ref = S_method["best_single_train_selected"]
results = {}
for m in METHODS:
    pooled = wp_of(S_method[m])
    bs = paired_component_bootstrap(S_method[m], ref, COMPS, n_boot=N_BOOT, seed=SEED, centered=True)
    results[m] = {"pooled": pooled, "vs_best_single_bootstrap": {k: v for k, v in bs.items()}}
    print(m, round(pooled["wp"], 10), "delta", round(pooled["wp"] - wp_of(ref)["wp"], 8), flush=True)

# Full-panel descriptive (exposed; not used for selection).
full_single = {n: wp_of(S_of[n])["wp"] for n in names}

# ---------- Correlation matrices (metric-weighted over selected rows) ----------
corr = {}
for k in range(2):
    w = W[:, k]
    wsum = w.sum()
    X = FEAT[k]
    R = Yc[:, k:k + 1] - X
    for label, A in (("prediction", X), ("residual", R)):
        mu = (w[:, None] * A).sum(0) / wsum
        D = A - mu
        C = (w[:, None] * D).T @ D / wsum
        sd = np.sqrt(np.diag(C))
        corr[f"{label}_t{k}"] = (C / np.outer(sd, sd)).tolist()
corr_names = names

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOTS.mkdir(parents=True, exist_ok=True)
for label in ("prediction", "residual"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    for k, ax in enumerate(axes):
        mat = np.asarray(corr[f"{label}_t{k}"])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=(-1 if label == "residual" else 0.9), vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5)
        ax.set_title(f"{label} correlation, target t{k}, metric-weighted, 468 groups")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{label}_correlation_t0_t1.png", dpi=130)
    plt.close(fig)

out = {
    "inventory": inventory, "kept": kept, "dropped": dropped,
    "fold_picks": picks, "methods": results,
    "full_panel_single_wp_exposed_descriptive": full_single,
    "correlation_names": corr_names, "correlation": corr,
    "selected_rows": int(N), "components": int(len(UNIQ)),
    "max_rss_note": "campaign_score-based moments; float32 predictions for 468 groups only",
    "runtime_seconds": time.time() - t_start,
}
RESULT_JSON.write_text(json.dumps(out, indent=2, default=float) + "\n")
print("wrote", RESULT_JSON, f"t={time.time()-t_start:.0f}s", flush=True)
