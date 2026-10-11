#!/usr/bin/env python3
"""Bounded F1 fixed-prediction selector; all fitted blend weights are component-split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


MODELS = [
    "vt_f1_dsq_dm_t3_a9_e6",
    "vt_f1_h391_dsq_h384_t3_a9_e6_s3_r2",
    "vt_f1_muonall_h384_e3",
    "vt_f1_muonall_h384_e3_s4",
    "vt_f1_muonall_h384_e3_s5",
]
CHUNK = 4
ROW_START = 99
SEED = 271828


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = float(weights.sum())
    if total <= 0:
        return 0.0, 0.0
    mean = float(np.sum(weights * values, dtype=np.float64) / total)
    var = float(np.sum(weights * (values - mean) ** 2, dtype=np.float64) / total)
    return mean, max(var, 0.0) ** 0.5


def score(y: np.ndarray, p: np.ndarray, mask: np.ndarray, GlobalAccumulator) -> dict:
    acc = GlobalAccumulator()
    for i in range(0, len(y), CHUNK):
        sl = slice(i, i + CHUNK)
        acc.add(y[sl], p[sl], mask[sl])
    return {k: float(v) for k, v in acc.result().items() if k in ("wp", "t0", "t1")}


def weighted_corr_pack(
    y: np.ndarray,
    mask: np.ndarray,
    pred_chunks,
    model_count: int,
) -> dict:
    sums = {kind: [[np.zeros(model_count), np.zeros((model_count, model_count)), 0.0]
                   for _ in range(2)] for kind in ("prediction", "residual")}
    output_scales = np.zeros((2, model_count, 2), dtype=np.float64)
    scale_sums = np.zeros((2, model_count, 3), dtype=np.float64)
    for start in range(0, len(y), CHUNK):
        sl = slice(start, start + CHUNK)
        yc = np.clip(y[sl], -2.0, 2.0).astype(np.float64)
        sel_mask = mask[sl]
        for target in range(2):
            valid = sel_mask & (np.arange(y.shape[1])[None, :] >= ROW_START)
            w = np.abs(yc[..., target])[valid]
            if not w.size:
                continue
            cols = []
            raw = []
            for loader in pred_chunks:
                zraw = np.asarray(loader(sl)[..., target], dtype=np.float64)[valid]
                z = np.clip(zraw, -2.0, 2.0)
                raw.append(zraw)
                cols.append(z)
            for i, values in enumerate(raw):
                sw = float(w.sum())
                scale_sums[target, i, 0] += sw
                scale_sums[target, i, 1] += float(np.dot(w, values))
                scale_sums[target, i, 2] += float(np.dot(w, values * values))
            for kind in ("prediction", "residual"):
                x = np.stack(cols, axis=1)
                if kind == "residual":
                    x = x - yc[..., target][valid, None]
                sw = float(w.sum())
                s, ss, wsum = sums[kind][target]
                s += (x * w[:, None]).sum(axis=0)
                ss += x.T @ (x * w[:, None])
                sums[kind][target] = [s, ss, wsum + sw]

    result = {"weighted_prediction_corr_by_target": [], "weighted_residual_corr_by_target": [],
              "weighted_output_mean_sd_by_target": []}
    for target in range(2):
        scale_rows = []
        for i, name in enumerate(MODELS):
            sw, sy, sy2 = scale_sums[target, i]
            mean = sy / sw if sw else 0.0
            sd = max(sy2 / sw - mean * mean, 0.0) ** 0.5 if sw else 0.0
            scale_rows.append({"model": name, "weighted_mean_raw_output": mean,
                               "weighted_sd_raw_output": sd})
        result["weighted_output_mean_sd_by_target"].append(scale_rows)
    for kind, dest in (("prediction", "weighted_prediction_corr_by_target"),
                       ("residual", "weighted_residual_corr_by_target")):
        rows = []
        for target in range(2):
            s, ss, sw = sums[kind][target]
            cov = ss / sw - np.outer(s / sw, s / sw)
            sd = np.sqrt(np.maximum(np.diag(cov), 1e-30))
            corr = cov / np.outer(sd, sd)
            rows.append({"target": target, "models": MODELS,
                         "matrix": corr.tolist(), "covariance": cov.tolist()})
        result[dest] = rows
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    root = args.main_root.resolve()
    sys.path.insert(0, str(root / "src"))
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments
    from wnn.gmetric import GlobalAccumulator

    split_path = root / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
    target_path = root / "cache/valid_targets.npz"
    split = json.loads(split_path.read_text())
    groups = [int(g) for g in split["evaluation"]["valid"]]
    row_component = {int(r["row_group"]): int(r["component"]) for r in split["sequences"]["valid"]}
    components = np.asarray([row_component[g] for g in groups], dtype=np.int64)
    if len(groups) != 457 or len(components) != len(groups):
        raise SystemExit(f"Unexpected fold-1 membership: {len(groups)} groups")

    with np.load(target_path, allow_pickle=False) as cache:
        target_pos = {int(g): i for i, g in enumerate(cache["row_group"])}
        target_order = np.asarray([target_pos[g] for g in groups], dtype=np.int64)
        y = np.asarray(cache["y"][target_order], dtype=np.float32)
        mask = np.asarray(cache["mask"][target_order], dtype=bool)
    mask &= np.arange(y.shape[1])[None, :] >= ROW_START

    loaded: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    identities = {}
    for name in MODELS:
        path = root / f"cache/pred_{name}_valid.npy"
        sidepath = path.with_suffix(".json")
        if not path.is_file() or not sidepath.is_file():
            raise SystemExit(f"missing scored local source artifact for {name}: {path}")
        side = json.loads(sidepath.read_text())
        if side["split_manifest_sha256"] != sha256(split_path):
            raise SystemExit(f"split pin mismatch for {name}")
        pos = {int(g): i for i, g in enumerate(side["row_groups"])}
        order = np.asarray([pos[g] for g in groups], dtype=np.int64)
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.shape != (1873, 20000, 2):
            raise SystemExit(f"unexpected shape for {name}: {arr.shape}")
        loaded[name] = (arr, order, side)
        identities[name] = {
            "prediction_path": str(path), "prediction_sha256": sha256(path),
            "sidecar_sha256": sha256(sidepath), "prediction_bytes": path.stat().st_size,
            "checkpoint_sha256": side.get("checkpoint_sha256"), "epoch": side.get("epoch"),
            "split_manifest_sha256": side.get("split_manifest_sha256"),
        }

    def loader(name: str):
        arr, order, _ = loaded[name]
        def get(sl: slice):
            return np.asarray(arr[order[sl]], dtype=np.float32)
        return get

    loaders = [loader(n) for n in MODELS]
    corr = weighted_corr_pack(y, mask, loaders, len(MODELS))

    index = {name: i for i, name in enumerate(MODELS)}
    # The first nine are predeclared, fixed equal-weight mixtures.
    fixed_specs = [
        ("h384_s3_s4_equal", ["vt_f1_muonall_h384_e3", "vt_f1_muonall_h384_e3_s4"], [0.5, 0.5]),
        ("h384_s3_s5_equal", ["vt_f1_muonall_h384_e3", "vt_f1_muonall_h384_e3_s5"], [0.5, 0.5]),
        ("h384_s4_s5_equal", ["vt_f1_muonall_h384_e3_s4", "vt_f1_muonall_h384_e3_s5"], [0.5, 0.5]),
        ("h384_s3_s4_s5_equal", ["vt_f1_muonall_h384_e3", "vt_f1_muonall_h384_e3_s4", "vt_f1_muonall_h384_e3_s5"], [1/3, 1/3, 1/3]),
        ("h256_h391_equal", ["vt_f1_dsq_dm_t3_a9_e6", "vt_f1_h391_dsq_h384_t3_a9_e6_s3_r2"], [0.5, 0.5]),
        ("h256_h384_s3_equal", ["vt_f1_dsq_dm_t3_a9_e6", "vt_f1_muonall_h384_e3"], [0.5, 0.5]),
        ("h256_h384_s4_equal", ["vt_f1_dsq_dm_t3_a9_e6", "vt_f1_muonall_h384_e3_s4"], [0.5, 0.5]),
        ("h256_h384_s5_equal", ["vt_f1_dsq_dm_t3_a9_e6", "vt_f1_muonall_h384_e3_s5"], [0.5, 0.5]),
        ("h256_half_h384_trio_half", ["vt_f1_dsq_dm_t3_a9_e6", "vt_f1_muonall_h384_e3", "vt_f1_muonall_h384_e3_s4", "vt_f1_muonall_h384_e3_s5"], [0.5, 1/6, 1/6, 1/6]),
    ]

    def pred_for(names: list[str], weights: list[float], sl: slice) -> np.ndarray:
        total = np.zeros_like(loaders[0](sl), dtype=np.float32)
        for n, w in zip(names, weights, strict=True):
            total += np.float32(w) * loader(n)(sl)
        return total

    fixed_results = []
    for name, members, weights in fixed_specs:
        p = np.concatenate(
            [pred_for(members, weights, slice(i, min(i + CHUNK, len(groups))))
             for i in range(0, len(groups), CHUNK)], axis=0
        )
        fixed_results.append({"candidate": name, "members": members, "weights": weights,
                              "full_fold": score(y, p, mask, GlobalAccumulator),
                              "scored_on_all_457_exposed_fold1_groups": True})

    # Deterministic component holdout for the two data-calibrated recipes.
    unique_components = sorted(set(map(int, components)))
    ordered_components = sorted(unique_components,
        key=lambda c: hashlib.sha256(f"selector-r1|{SEED}|{c}".encode()).hexdigest())
    n_cal = int(round(0.65 * len(ordered_components)))
    cal_ids = set(ordered_components[:n_cal])
    cal = np.asarray([int(c) in cal_ids for c in components], dtype=bool)
    ev = ~cal
    if cal.sum() < 100 or ev.sum() < 50:
        raise SystemExit("component split too small for calibrated blend diagnostic")
    eval_components = components[ev]
    cal_idx = np.flatnonzero(cal)
    ev_idx = np.flatnonzero(ev)
    cal_y = y[cal_idx]
    cal_mask = mask[cal_idx]
    eval_y = y[ev_idx]
    eval_mask = mask[ev_idx]
    student = "vt_f1_dsq_dm_t3_a9_e6"
    teachers = ["vt_f1_muonall_h384_e3", "vt_f1_muonall_h384_e3_s4",
                "vt_f1_muonall_h384_e3_s5"]
    blend_groups = [student, "equal_mean(H384_s3_s4_s5)"]
    invsd_weights = np.zeros((2, 2), dtype=np.float64)
    cov_weights = np.zeros_like(invsd_weights)
    cal_scales = []
    scale_sums = np.zeros((2, 2, 3), dtype=np.float64)
    resid_sum = np.zeros((2, 2), dtype=np.float64)
    resid_cross = np.zeros((2, 2, 2), dtype=np.float64)
    weight_sum = np.zeros(2, dtype=np.float64)
    for start in range(0, len(cal_idx), CHUNK):
        ids = cal_idx[start:start + CHUNK]
        yc = np.clip(cal_y[start:start + CHUNK], -2.0, 2.0).astype(np.float64)
        wm = np.abs(yc)
        valid = cal_mask[start:start + CHUNK] & (np.arange(y.shape[1])[None, :] >= ROW_START)
        raw_student = np.asarray(loaded[student][0][loaded[student][1][ids]], dtype=np.float32)
        raw_teacher = np.zeros_like(raw_student)
        for n in teachers:
            raw_teacher += np.asarray(loaded[n][0][loaded[n][1][ids]], dtype=np.float32) / 3.0
        raw = [raw_student, raw_teacher]
        for t in range(2):
            w = wm[..., t][valid]
            if not w.size:
                continue
            weight_sum[t] += w.sum()
            xcols = []
            rcols = []
            yv = yc[..., t][valid]
            for i, arr in enumerate(raw):
                vals = np.asarray(arr[..., t][valid], dtype=np.float64)
                clipped = np.clip(vals, -2.0, 2.0)
                scale_sums[t, i, 0] += np.sum(w)
                scale_sums[t, i, 1] += np.dot(w, clipped)
                scale_sums[t, i, 2] += np.dot(w, clipped * clipped)
                xcols.append(clipped)
                rcols.append(clipped - yv)
            resid = np.stack(rcols, axis=1)
            resid_sum[t] += (resid * w[:, None]).sum(axis=0)
            resid_cross[t] += resid.T @ (resid * w[:, None])

    for t in range(2):
        scales = []
        sw = weight_sum[t]
        for i in range(2):
            mean = scale_sums[t, i, 1] / sw
            var = max(scale_sums[t, i, 2] / sw - mean * mean, 0.0)
            scales.append(max(var ** 0.5, 1e-8))
        inv = 1.0 / np.asarray(scales)
        invsd_weights[t] = inv / inv.sum()
        # Weighted residual covariance on calibration components only.
        mean = resid_sum[t] / sw
        cov = resid_cross[t] / sw - np.outer(mean, mean)
        ridge = max(float(np.trace(cov)) / 2 * 0.10, 1e-10)
        raw = np.linalg.solve(cov + ridge * np.eye(2), np.ones(2))
        raw = np.maximum(raw, 0.0)
        raw = raw / raw.sum() if raw.sum() else np.full(2, 0.5)
        cov_weights[t] = 0.8 * 0.5 + 0.2 * raw
        cal_scales.append({"target": t, "weighted_sd_by_blend_group": scales,
                           "residual_covariance": cov.tolist(), "ridge": ridge,
                           "unshrunk_nonnegative_weights": raw.tolist()})

    # Fit one scalar per model/target on calibration components only. Measure
    # scale-aligned residual correlations on the disjoint component complement.
    ls_num = np.zeros((2, len(MODELS)), dtype=np.float64)
    ls_den = np.zeros_like(ls_num)
    for start in range(0, len(cal_idx), CHUNK):
        ids = cal_idx[start:start + CHUNK]
        ypart = np.clip(cal_y[start:start + CHUNK], -2.0, 2.0).astype(np.float64)
        mpart = cal_mask[start:start + CHUNK]
        pred_part = [np.asarray(loaded[n][0][loaded[n][1][ids]], dtype=np.float64)
                     for n in MODELS]
        for t in range(2):
            valid = mpart & (np.arange(y.shape[1])[None, :] >= ROW_START)
            w = np.abs(ypart[..., t])[valid]
            if not w.size:
                continue
            yv = ypart[..., t][valid]
            for i in range(len(MODELS)):
                xv = pred_part[i][..., t][valid]
                ls_num[t, i] += np.dot(w, yv * xv)
                ls_den[t, i] += np.dot(w, xv * xv)
    ls_scales = np.divide(ls_num, ls_den, out=np.zeros_like(ls_num), where=ls_den > 0)
    ls_scales = np.maximum(ls_scales, 0.0)
    eval_sum = np.zeros((2, len(MODELS)), dtype=np.float64)
    eval_cross = np.zeros((2, len(MODELS), len(MODELS)), dtype=np.float64)
    eval_weight = np.zeros(2, dtype=np.float64)
    for start in range(0, len(ev_idx), CHUNK):
        ids = ev_idx[start:start + CHUNK]
        ypart = np.clip(eval_y[start:start + CHUNK], -2.0, 2.0).astype(np.float64)
        mpart = eval_mask[start:start + CHUNK]
        pred_part = [np.asarray(loaded[n][0][loaded[n][1][ids]], dtype=np.float64)
                     for n in MODELS]
        for t in range(2):
            valid = mpart & (np.arange(y.shape[1])[None, :] >= ROW_START)
            w = np.abs(ypart[..., t])[valid]
            if not w.size:
                continue
            yv = ypart[..., t][valid]
            errors = np.stack([
                yv - np.clip(ls_scales[t, i] * pred_part[i][..., t][valid], -2.0, 2.0)
                for i in range(len(MODELS))
            ], axis=1)
            eval_weight[t] += w.sum()
            eval_sum[t] += (errors * w[:, None]).sum(axis=0)
            eval_cross[t] += errors.T @ (errors * w[:, None])
    aligned_corr = []
    for t in range(2):
        mu = eval_sum[t] / eval_weight[t]
        cov = eval_cross[t] / eval_weight[t] - np.outer(mu, mu)
        sd = np.sqrt(np.maximum(np.diag(cov), 1e-30))
        aligned_corr.append({"target": t, "models": MODELS,
                             "matrix": (cov / np.outer(sd, sd)).tolist(),
                             "covariance": cov.tolist()})
    corr["calibration_only_zero_intercept_scale_alignment"] = {
        "fit_components": len(cal_ids), "fit_groups": int(cal.sum()),
        "evaluation_components": len(set(map(int, eval_components))),
        "evaluation_groups": int(ev.sum()),
        "per_target_scales": ls_scales.tolist(),
        "weighted_residual_corr_by_target": aligned_corr,
        "method": "weighted least-squares through-origin slopes fit on calibration components; errors measured only on complement",
    }

    fitted_results = []
    for name, coeffs in (("inverse_output_sd_calibration_split", invsd_weights),
                         ("residual_covariance_nonnegative_80pct_equal_shrink", cov_weights)):
        acc = GlobalAccumulator()
        for start in range(0, len(ev_idx), CHUNK):
            ids = ev_idx[start:start + CHUNK]
            p = np.zeros_like(eval_y[start:start + CHUNK], dtype=np.float32)
            arr, order, _ = loaded[student]
            x_student = np.asarray(arr[order[ids]], dtype=np.float32)
            x_teacher = np.zeros_like(x_student)
            for model in teachers:
                arr, order, _ = loaded[model]
                x_teacher += np.asarray(arr[order[ids]], dtype=np.float32) / 3.0
            for t in range(2):
                p[..., t] = (np.float32(coeffs[t, 0]) * x_student[..., t]
                             + np.float32(coeffs[t, 1]) * x_teacher[..., t])
            acc.add(eval_y[start:start + CHUNK], p, eval_mask[start:start + CHUNK])
        fitted_results.append({"candidate": name, "members": [student, *teachers],
                               "grouping": blend_groups,
                               "targetwise_group_weights": coeffs.tolist(),
                               "evaluated_on_disjoint_component_complement": True,
                               "evaluation_groups": int(ev.sum()),
                               "score": {k: float(v) for k, v in acc.result().items()
                                         if k in ("wp", "t0", "t1")}})

    # One additional scale-only arm: fit a zero-intercept, targetwise WLS
    # multiplier for the complete fixed H256/H384-trio blend on calibration
    # components. Apply it to the raw un-clipped blend; GlobalAccumulator then
    # applies the official prediction clip before scoring the complement.
    blend_scale_num = np.zeros(2, dtype=np.float64)
    blend_scale_den = np.zeros(2, dtype=np.float64)
    for start in range(0, len(cal_idx), CHUNK):
        ids = cal_idx[start:start + CHUNK]
        ypart = np.clip(cal_y[start:start + CHUNK], -2.0, 2.0).astype(np.float64)
        mpart = cal_mask[start:start + CHUNK]
        raw = np.zeros_like(cal_y[start:start + CHUNK], dtype=np.float32)
        arr, order, _ = loaded[student]
        raw += np.float32(0.5) * np.asarray(arr[order[ids]], dtype=np.float32)
        for model in teachers:
            arr, order, _ = loaded[model]
            raw += np.float32(1.0 / 6.0) * np.asarray(arr[order[ids]], dtype=np.float32)
        for t in range(2):
            valid = mpart & (np.arange(y.shape[1])[None, :] >= ROW_START)
            w = np.abs(ypart[..., t])[valid]
            if not w.size:
                continue
            xv = np.asarray(raw[..., t][valid], dtype=np.float64)
            yv = ypart[..., t][valid]
            blend_scale_num[t] += np.dot(w, xv * yv)
            blend_scale_den[t] += np.dot(w, xv * xv)
    blend_scales = np.divide(blend_scale_num, blend_scale_den,
                             out=np.zeros_like(blend_scale_num),
                             where=blend_scale_den > 0)
    blend_scale_acc = GlobalAccumulator()
    scale_candidate = np.zeros_like(eval_y, dtype=np.float32)
    equal_candidate = np.zeros_like(eval_y, dtype=np.float32)
    for start in range(0, len(ev_idx), CHUNK):
        ids = ev_idx[start:start + CHUNK]
        raw = np.zeros_like(eval_y[start:start + CHUNK], dtype=np.float32)
        arr, order, _ = loaded[student]
        raw += np.float32(0.5) * np.asarray(arr[order[ids]], dtype=np.float32)
        for model in teachers:
            arr, order, _ = loaded[model]
            raw += np.float32(1.0 / 6.0) * np.asarray(arr[order[ids]], dtype=np.float32)
        scaled = raw.copy()
        for t in range(2):
            scaled[..., t] *= np.float32(blend_scales[t])
        scale_candidate[start:start + CHUNK] = scaled
        equal_candidate[start:start + CHUNK] = raw
        blend_scale_acc.add(eval_y[start:start + CHUNK], scaled,
                            eval_mask[start:start + CHUNK])
    fitted_results.append({
        "candidate": "h256_half_h384_trio_half_targetwise_calibrated_global_scale",
        "members": [student, *teachers],
        "fixed_base_candidate": "h256_half_h384_trio_half",
        "targetwise_zero_intercept_wls_scales_fit_on_calibration_components": blend_scales.tolist(),
        "scale_fit": "weighted least-squares through-origin scalar for the complete raw blend, per target; fit on clipped targets with official absolute-clipped-target weights",
        "scale_application": "multiply raw complete blend before official prediction clipping at +/-2",
        "evaluated_on_disjoint_component_complement": True,
        "evaluation_groups": int(ev.sum()),
        "score": {k: float(v) for k, v in blend_scale_acc.result().items()
                  if k in ("wp", "t0", "t1")},
    })
    scaled_mom, equal_mom = paired_component_moments(
        eval_y, scale_candidate, equal_candidate, eval_mask
    )
    scale_bootstrap = paired_component_bootstrap(
        scaled_mom, equal_mom, eval_components, n_boot=args.bootstrap,
        seed=SEED, centered=True
    )

    # Match all simple fixed recipes and singles on the same complement used by fitted recipes.
    single_complement = {}
    full_selected = {}
    for n in MODELS:
        arr, order, _ = loaded[n]
        full_selected[n] = np.asarray(arr[order[ev_idx]], dtype=np.float32)
        single_complement[n] = score(eval_y, full_selected[n], eval_mask, GlobalAccumulator)
    fixed_complement = []
    for cname, members, coeff in fixed_specs:
        acc = GlobalAccumulator()
        for start in range(0, len(ev_idx), CHUNK):
            ids = ev_idx[start:start + CHUNK]
            p = np.zeros_like(eval_y[start:start + CHUNK], dtype=np.float32)
            for model, weight in zip(members, coeff, strict=True):
                arr, order, _ = loaded[model]
                p += np.float32(weight) * np.asarray(arr[order[ids]], dtype=np.float32)
            acc.add(eval_y[start:start + CHUNK], p, eval_mask[start:start + CHUNK])
        fixed_complement.append({"candidate": cname, "members": members, "weights": coeff,
                                 "score": {k: float(v) for k, v in acc.result().items()
                                           if k in ("wp", "t0", "t1")}})

    # Component-bootstrap paired gains for the most decision-relevant fixed ensembles.
    component_map = eval_components
    bootstrap_results = []
    boot_specs = [
        ("h384_s3_s4_s5_equal", [MODELS[2], MODELS[3], MODELS[4]], [1/3]*3, MODELS[4]),
        ("h256_half_h384_trio_half", [MODELS[0], MODELS[2], MODELS[3], MODELS[4]], [0.5, 1/6, 1/6, 1/6], MODELS[0]),
        ("h256_half_h384_trio_half_vs_seed5", [MODELS[0], MODELS[2], MODELS[3], MODELS[4]], [0.5, 1/6, 1/6, 1/6], MODELS[4]),
        ("h256_h391_equal", [MODELS[0], MODELS[1]], [0.5, 0.5], MODELS[0]),
        ("h256_h384_s5_equal", [MODELS[0], MODELS[4]], [0.5, 0.5], MODELS[0]),
        ("h384_s4_s5_equal", [MODELS[3], MODELS[4]], [0.5, 0.5], MODELS[4]),
    ]
    ev_indexes = np.flatnonzero(ev)
    for cname, members, coeff, reference in boot_specs:
        cand = np.zeros_like(eval_y, dtype=np.float32)
        for m, w in zip(members, coeff, strict=True):
            cand += np.float32(w) * full_selected[m]
        ref = full_selected[reference]
        cmask = eval_mask.copy()
        cmom, rmom = paired_component_moments(eval_y, cand, ref, cmask)
        result = paired_component_bootstrap(cmom, rmom, component_map,
                                            n_boot=args.bootstrap, seed=SEED, centered=True)
        bootstrap_results.append({"candidate": cname, "reference": reference,
                                  "evaluation_component_count": len(set(map(int, component_map))),
                                  "bootstrap": result})

    result = {
        "schema": "wnn.submission-selector-f1.v1",
        "official_metric": "global_wp_clip2_v1",
        "fold": 1,
        "exposure": "historically exposed; all overall-F1 comparisons exploratory, component complement is only a fixed-prediction combiner diagnostic",
        "evaluation_groups": len(groups),
        "group_membership_sha256": sha256(split_path),
        "component_split": {"method": "sha256 sorted component IDs", "seed": SEED,
                             "calibration_fraction": 0.65, "calibration_groups": int(cal.sum()),
                             "complement_groups": int(ev.sum()),
                             "calibration_components": len(cal_ids),
                             "complement_components": len(set(map(int, eval_components)))},
        "models": identities,
        "singles_full_fold": {},
        "mixture_pool_size": len(fixed_specs) + len(fitted_results),
        "predeclared_fixed_mixture_count": len(fixed_specs),
        "calibration_fitted_mixture_count": len(fitted_results),
        "fixed_mixtures_full_fold": fixed_results,
        "fitted_mixtures_component_complement": fitted_results,
        "fixed_mixtures_component_complement": fixed_complement,
        "fixed_single_scores_component_complement": single_complement,
        "calibration_diagnostics": cal_scales,
        "weighted_correlations_and_output_scales": corr,
        "component_bootstrap_complement": bootstrap_results,
        "calibrated_overall_scale_vs_fixed_base_component_bootstrap": {
            "candidate": "h256_half_h384_trio_half_targetwise_calibrated_global_scale",
            "reference": "h256_half_h384_trio_half",
            "bootstrap": scale_bootstrap,
        },
        "scope_limitations": [
            "No full-data/in-sample prediction was scored.",
            "Prop0 predictions named by an earlier remote headroom receipt are unavailable locally, so their scores are recorded separately and no pairwise correlation is claimed.",
            "The component complement is not a pristine holdout: fold 1 has prior model/blend selection exposure.",
            "Weights are diagnostic fixed-prediction blend parameters, not refit cross-validation or public-score estimates.",
        ],
    }
    for i, name in enumerate(MODELS):
        pp = np.concatenate([loader(name)(slice(j, min(j + CHUNK, len(groups))))
                             for j in range(0, len(groups), CHUNK)], axis=0)
        result["singles_full_fold"][name] = score(y, pp, mask, GlobalAccumulator)
    args.output.mkdir(parents=True, exist_ok=True)
    outfile = args.output / "F1_SELECTOR_METRICS_R1.json"
    outfile.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(outfile), "sha256": sha256(outfile),
                      "mixture_pool_size": result["mixture_pool_size"],
                      "calibration_groups": int(cal.sum()), "complement_groups": int(ev.sum()),
                      "singles": result["singles_full_fold"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
