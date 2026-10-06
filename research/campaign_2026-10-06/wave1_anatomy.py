"""Where the seed-4 champion's score lives, and whether a new causal state adds any.

Descriptive pass uses the frozen equal parent/head blend on every supplied
validation sequence. Fits never see holdout sequences (``wnn.splits.is_holdout``).
The primary residual model is predeclared: alpha 1, one Hawkes decay taken from
the train inter-event gap, no coefficient search.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numba as nb
import numpy as np
import pyarrow.parquet as pq

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, TARGET_QUANTA
from wnn.gmetric import GlobalAccumulator
from wnn.splits import is_holdout

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLIP = 2.0
AGE_EDGES = (0, 1, 8, 32, 128)  # bins: 0, 1-7, 8-31, 32-127, 128+
N_AGE = 5
N_CLOCK = 4
PRIMARY_ALPHA = 1.0
TRAIN_BETA_GROUPS = 40
TRAIN_RIDGE_GROUPS = 400
FEATURE_NAMES = (
    "log_age_or",
    "log_age_and",
    "ev_and",
    "ev_or",
    "lam_p0",
    "lam_p11",
    "lam_both",
    "lam_trade",
    "lam_vol",
    "dep_bid",
    "dep_ask",
    "i1_disp0",
    "i1_disp11",
    "vol_imb",
    "a6",
    "da6",
)
CONTROL_NAMES = ("pred0", "pred1")
COLS = (
    "i0_p0",
    "i0_p11",
    "i0_v0",
    "i0_v11",
    "i0_dp0",
    "i0_dv0",
    "i1_p0",
    "i1_p11",
    "a6",
)


def _split_bit(seq_ix: int, salt: str) -> int:
    digest = hashlib.sha256(f"{int(seq_ix)}:{salt}".encode()).hexdigest()
    return int(digest, 16) % 2


def _age_bin(age: np.ndarray) -> np.ndarray:
    """Map nonnegative ages to five buckets."""
    bins = np.zeros(age.shape, dtype=np.int8)
    bins[age >= 1] = 1
    bins[age >= 8] = 2
    bins[age >= 32] = 3
    bins[age >= 128] = 4
    return bins


def _changed(column: np.ndarray) -> np.ndarray:
    out = np.zeros(column.shape[0], dtype=bool)
    out[1:] = column[1:] != column[:-1]
    return out


def _since_last(event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Age and index of the last True, with row 0 as the start if no event yet."""
    idx = np.arange(event.shape[0])
    last = np.maximum.accumulate(np.where(event, idx, 0))
    return idx - last, last


@nb.njit(cache=True)
def _compensators(marks: np.ndarray, decay: float) -> np.ndarray:
    """Causal Hawkes sums, one column per mark. marks is (T, K) float64."""
    t_len, n_marks = marks.shape
    out = np.empty((t_len, n_marks), dtype=np.float64)
    acc = np.zeros(n_marks, dtype=np.float64)
    for t in range(t_len):
        for k in range(n_marks):
            acc[k] = decay * acc[k] + marks[t, k]
        out[t] = acc
    return out


def _read_columns(pf: pq.ParquetFile, index: int) -> dict[str, np.ndarray]:
    table = pf.read_row_group(index, columns=list(COLS), use_threads=False)
    return {name: table.column(name).to_numpy() for name in COLS}


def _clock_and_features(
    cols: dict[str, np.ndarray], decay: float
) -> dict[str, np.ndarray]:
    """Causal clocks, ages, Hawkes compensators and depletion from one sequence."""
    dp0 = _changed(cols["i0_p0"])
    dp11 = _changed(cols["i0_p11"])
    dtrade = _changed(cols["i0_dp0"]) | _changed(cols["i0_dv0"])
    dvol = _changed(cols["i0_v0"]) | _changed(cols["i0_v11"])
    da6 = _changed(cols["a6"])
    clock = (dp0.astype(np.int8) + 2 * dp11.astype(np.int8)).astype(np.int8)
    or_ev = dp0 | dp11
    and_ev = dp0 & dp11
    age_or, last_or = _since_last(or_ev)
    age_and, _last_and = _since_last(and_ev)
    marks = np.column_stack(
        [
            (dp0 & ~dp11).astype(np.float64),
            (dp11 & ~dp0).astype(np.float64),
            (dp0 & dp11).astype(np.float64),
            dtrade.astype(np.float64),
            dvol.astype(np.float64),
        ]
    )
    lams = _compensators(marks, decay)
    v0 = cols["i0_v0"].astype(np.float64)
    v11 = cols["i0_v11"].astype(np.float64)
    drop0 = np.zeros(v0.shape[0], dtype=np.float64)
    drop11 = np.zeros(v11.shape[0], dtype=np.float64)
    drop0[1:] = np.maximum(0.0, -(v0[1:] - v0[:-1]))
    drop11[1:] = np.maximum(0.0, -(v11[1:] - v11[:-1]))
    c0 = np.cumsum(drop0)
    c11 = np.cumsum(drop11)
    dep_bid = c0 - c0[last_or]
    dep_ask = c11 - c11[last_or]
    i1_0 = cols["i1_p0"].astype(np.float64)
    i1_11 = cols["i1_p11"].astype(np.float64)
    disp0 = i1_0 - i1_0[last_or]
    disp11 = i1_11 - i1_11[last_or]
    features = np.column_stack(
        [
            np.log1p(age_or.astype(np.float64)),
            np.log1p(age_and.astype(np.float64)),
            and_ev.astype(np.float64),
            or_ev.astype(np.float64),
            lams,
            dep_bid,
            dep_ask,
            disp0,
            disp11,
            v0 - v11,
            cols["a6"].astype(np.float64),
            da6.astype(np.float64),
        ]
    )
    return {
        "clock": clock,
        "or_ev": or_ev,
        "and_ev": and_ev,
        "age_or": age_or,
        "age_and": age_and,
        "da6": da6,
        "features": features,
    }


def _score(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    acc = GlobalAccumulator()
    acc.add(y, p, mask)
    return float(acc.result()["wp"])


def _chunk_score(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    acc = GlobalAccumulator()
    step = 200
    for start in range(0, y.shape[0], step):
        acc.add(y[start : start + step], p[start : start + step], mask[start : start + step])
    result = acc.result()
    return {"t0": float(result["t0"]), "t1": float(result["t1"]), "wp": float(result["wp"])}


def _mean_or_gap(train: pq.ParquetFile) -> float:
    gaps: list[float] = []
    seen = 0
    for index in range(train.num_row_groups):
        if index in CORRUPT_TRAIN_ROW_GROUPS:
            continue
        table = train.read_row_group(index, columns=["i0_p0", "i0_p11"], use_threads=False)
        p0 = table.column("i0_p0").to_numpy()
        p11 = table.column("i0_p11").to_numpy()
        event = np.zeros(p0.shape[0], dtype=bool)
        event[1:] = (p0[1:] != p0[:-1]) | (p11[1:] != p11[:-1])
        at = np.flatnonzero(event)
        if at.size >= 2:
            gaps.extend(np.diff(at).astype(np.float64).tolist())
        seen += 1
        if seen >= TRAIN_BETA_GROUPS:
            break
    if not gaps:
        raise RuntimeError("no OR events in the train sample used for beta")
    return float(np.mean(gaps))


def _accumulate_xtx(
    features: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sw = np.sqrt(np.maximum(weight, 0.0))
    design = np.column_stack([np.ones(features.shape[0]), features])
    weighted = design * sw[:, None]
    xtx = weighted.T @ weighted
    xty = weighted.T @ (target * sw[:, None])
    return xtx, xty


def _solve_ridge(xtx: np.ndarray, xty: np.ndarray, alpha: float) -> np.ndarray:
    penalized = xtx.copy()
    penalty = alpha * np.eye(xtx.shape[0])
    penalty[0, 0] = 0.0
    penalized += penalty
    return np.linalg.solve(penalized, xty)


def _bincount_add(
    store: dict[str, np.ndarray],
    key: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    n_keys: int,
) -> None:
    """Add clipped weighted moments for already-selected rows."""
    yc = np.clip(y, -CLIP, CLIP)
    pc = np.clip(p, -CLIP, CLIP)
    for target_i in range(2):
        w = np.abs(yc[:, target_i])
        yy = yc[:, target_i]
        pp = pc[:, target_i]
        store["w"][target_i] += np.bincount(key, weights=w, minlength=n_keys)
        store["wy"][target_i] += np.bincount(key, weights=w * yy, minlength=n_keys)
        store["wp"][target_i] += np.bincount(key, weights=w * pp, minlength=n_keys)
        store["wyy"][target_i] += np.bincount(key, weights=w * yy * yy, minlength=n_keys)
        store["wpp"][target_i] += np.bincount(key, weights=w * pp * pp, minlength=n_keys)
        store["wyp"][target_i] += np.bincount(key, weights=w * yy * pp, minlength=n_keys)


def _empty_store(n_keys: int) -> dict[str, np.ndarray]:
    shape = (2, n_keys)
    return {name: np.zeros(shape, dtype=np.float64) for name in ("w", "wy", "wp", "wyy", "wpp", "wyp")}


def _cell_table(store: dict[str, np.ndarray], names: list[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(names):
        record: dict[str, float | str] = {"cell": name}
        local = []
        for target_i, target_name in enumerate(("t0", "t1")):
            w = float(store["w"][target_i, index])
            record[f"w_{target_name}"] = w
            if w <= 1e-8:
                record[f"wp_{target_name}"] = 0.0
                local.append(0.0)
                continue
            my = store["wy"][target_i, index] / w
            mp = store["wp"][target_i, index] / w
            vy = store["wyy"][target_i, index] / w - my * my
            vp = store["wpp"][target_i, index] / w - mp * mp
            cov = store["wyp"][target_i, index] / w - my * mp
            if vy <= 1e-12 or vp <= 1e-12:
                wp = 0.0
            else:
                wp = float(np.clip(cov / np.sqrt(vy * vp), -1.0, 1.0))
            record[f"wp_{target_name}"] = wp
            local.append(wp)
        record["wp"] = 0.5 * (local[0] + local[1])
        rows.append(record)
    return rows


def _global_means(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yc = np.clip(y[mask], -CLIP, CLIP)
    pc = np.clip(p[mask], -CLIP, CLIP)
    w = np.abs(yc)
    denom = w.sum(axis=0)
    my = (w * yc).sum(axis=0) / denom
    mp = (w * pc).sum(axis=0) / denom
    return my, mp


def _contribution(
    y: np.ndarray,
    p: np.ndarray,
    key: np.ndarray,
    my: np.ndarray,
    mp: np.ndarray,
    n_keys: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Weight and global-mean covariance numerator per cell, shape (2, n_keys)."""
    yc = np.clip(y, -CLIP, CLIP)
    pc = np.clip(p, -CLIP, CLIP)
    weight = np.zeros((2, n_keys), dtype=np.float64)
    numer = np.zeros((2, n_keys), dtype=np.float64)
    for target_i in range(2):
        w = np.abs(yc[:, target_i])
        weight[target_i] = np.bincount(key, weights=w, minlength=n_keys)
        piece = w * (yc[:, target_i] - my[target_i]) * (pc[:, target_i] - mp[target_i])
        numer[target_i] = np.bincount(key, weights=piece, minlength=n_keys)
    return weight, numer


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache"
    with np.load(cache / "valid_targets.npz") as valid:
        y = valid["y"]
        mask = valid["mask"]
        seq_ix = valid["seq_ix"]
        row_group = valid["row_group"]
        market_class = valid["market_class"]
    parent = np.load(cache / "pred_c20261004_combo_s4_final_fp32_valid.npy", mmap_mode="r")
    head = np.load(cache / "pred_c20261004_combo_head_s4_final_fp32_valid.npy", mmap_mode="r")
    blend = (0.5 * (parent.astype(np.float64) + head.astype(np.float64))).astype(np.float32)
    hold = np.array([is_holdout(int(s)) for s in seq_ix])
    dev = ~hold

    blend_score = _chunk_score(y, blend, mask)
    zero = np.zeros_like(blend)
    zero_score = _chunk_score(y, zero, mask)
    by_class = {}
    for cls in range(4):
        take = market_class == cls
        by_class[str(cls)] = _chunk_score(y[take], blend[take], mask[take])

    quantum = TARGET_QUANTA[market_class].astype(np.float64)
    snapped = np.empty_like(blend, dtype=np.float32)
    sum_snapped = np.empty_like(blend, dtype=np.float32)
    for start in range(0, blend.shape[0], 200):
        stop = min(start + 200, blend.shape[0])
        block = blend[start:stop].astype(np.float64)
        q = quantum[start:stop, None, None]
        snapped[start:stop] = np.clip(q * np.rint(block / q), -CLIP, CLIP).astype(np.float32)
        total = block.sum(axis=-1)
        total_q = (q[..., 0] * np.rint(total / q[..., 0]))
        delta = 0.5 * (total_q - total)
        adjusted = block + delta[..., None]
        sum_snapped[start:stop] = np.clip(adjusted, -CLIP, CLIP).astype(np.float32)
    lattice_score = _chunk_score(y, snapped, mask)
    sum_lattice_score = _chunk_score(y, sum_snapped, mask)

    train = pq.ParquetFile(ROOT / "datasets" / "train.parquet")
    mean_gap = _mean_or_gap(train)
    beta = 1.0 / mean_gap
    decay = float(np.exp(-beta))

    valid_pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    my, mp = _global_means(y, blend, mask)
    n_keys = N_CLOCK * N_AGE
    or_names = [
        f"clock{clock}_age{age}"
        for age in range(N_AGE)
        for clock in range(N_CLOCK)
    ]
    # key layout: age_bin * 4 + clock, matching the name order age-major.
    local_or = _empty_store(n_keys)
    local_and = _empty_store(n_keys)
    numer_w = np.zeros((2, n_keys), dtype=np.float64)
    numer = np.zeros((2, n_keys), dtype=np.float64)
    contingency = np.zeros((2, N_CLOCK), dtype=np.int64)  # [changed, clock]
    a6_counts = {
        "scored": 0,
        "a6_changed": 0,
        "or_event": 0,
        "and_event": 0,
        "a6_and_or": 0,
        "a6_and_and": 0,
        "a6_only": 0,
        "or_only": 0,
        "label_and_a6": 0,
        "label_changed": 0,
    }
    # Ridge on the champion residual, fit/eval split of non-holdout validation.
    d_feat = len(FEATURE_NAMES)
    d_ctrl = 1 + 2
    d_full = 1 + d_feat + 2
    stats = {
        "fit_x": np.zeros((d_full, d_full)),
        "fit_xy": np.zeros((d_full, 2)),
        "fit_rows": 0,
        "eval_y": [],
        "eval_p": [],
        "eval_f": [],
        "eval_wmask": [],
    }
    # Feature-only model fit on a train sample, via sufficient statistics.
    feat_dim = 1 + d_feat
    train_xtx = np.zeros((feat_dim, feat_dim))
    train_xty = np.zeros((feat_dim, 2))
    train_w = np.zeros(d_feat)
    train_ww = np.zeros(d_feat)
    train_weight = 0.0
    train_seen = 0

    def consume_valid(index_in_npz: int, cols: dict[str, np.ndarray]) -> None:
        state = _clock_and_features(cols, decay)
        scored = mask[index_in_npz]
        y_s = y[index_in_npz][scored]
        p_s = blend[index_in_npz][scored]
        clock = state["clock"][scored]
        age_or = _age_bin(state["age_or"])[scored]
        age_and = _age_bin(state["age_and"])[scored]
        key_or = age_or.astype(np.int64) * N_CLOCK + clock.astype(np.int64)
        key_and = age_and.astype(np.int64) * N_CLOCK + clock.astype(np.int64)
        _bincount_add(local_or, key_or, y_s, p_s, n_keys)
        _bincount_add(local_and, key_and, y_s, p_s, n_keys)
        w_or, n_or = _contribution(y_s, p_s, key_or, my, mp, n_keys)
        numer_w[:] += w_or
        numer[:] += n_or
        changed = np.zeros(mask.shape[1], dtype=bool)
        changed[1:] = np.max(np.abs(y[index_in_npz, 1:] - y[index_in_npz, :-1]), axis=1) > 1e-5
        changed_s = changed[scored]
        for clock_id in range(N_CLOCK):
            take = clock == clock_id
            contingency[0, clock_id] += int((~changed_s & take).sum())
            contingency[1, clock_id] += int((changed_s & take).sum())
        a6_counts["scored"] += int(scored.sum())
        a6_s = state["da6"][scored]
        or_s = state["or_ev"][scored]
        and_s = state["and_ev"][scored]
        a6_counts["a6_changed"] += int(a6_s.sum())
        a6_counts["or_event"] += int(or_s.sum())
        a6_counts["and_event"] += int(and_s.sum())
        a6_counts["a6_and_or"] += int((a6_s & or_s).sum())
        a6_counts["a6_and_and"] += int((a6_s & and_s).sum())
        a6_counts["a6_only"] += int((a6_s & ~or_s).sum())
        a6_counts["or_only"] += int((or_s & ~a6_s).sum())
        a6_counts["label_changed"] += int(changed_s.sum())
        a6_counts["label_and_a6"] += int((changed_s & a6_s).sum())
        if hold[index_in_npz]:
            return
        features = state["features"][scored]
        preds = p_s.astype(np.float64)
        design = np.column_stack([features, preds])
        if _split_bit(int(seq_ix[index_in_npz]), "wave1b") == 0:
            sw = np.sqrt(np.maximum(np.abs(np.clip(y_s, -CLIP, CLIP)).mean(axis=1), 0.0))
            weighted = np.column_stack([np.ones(design.shape[0]), design]) * sw[:, None]
            stats["fit_x"] += weighted.T @ weighted
            stats["fit_xy"] += weighted.T @ (y_s.astype(np.float64) * sw[:, None])
            stats["fit_rows"] += int(design.shape[0])
        else:
            stats["eval_y"].append(y_s.astype(np.float32))
            stats["eval_p"].append(p_s.astype(np.float32))
            stats["eval_f"].append(design.astype(np.float32))

    # Train sufficient statistics for the feature-only model, then valid stream.
    for index in range(train.num_row_groups):
        if index in CORRUPT_TRAIN_ROW_GROUPS:
            continue
        table = train.read_row_group(index, columns=[*COLS, "t0", "t1"], use_threads=False)
        cols = {name: table.column(name).to_numpy() for name in COLS}
        target = np.column_stack(
            [table.column("t0").to_numpy(), table.column("t1").to_numpy()]
        )
        state = _clock_and_features(cols, decay)
        rows = slice(99, None)
        feat = state["features"][rows]
        yy = target[rows].astype(np.float64)
        weight = np.abs(np.clip(yy, -CLIP, CLIP)).mean(axis=1)
        xtx, xty = _accumulate_xtx(feat, yy, weight)
        train_xtx += xtx
        train_xty += xty
        sw = weight
        train_w += (feat * sw[:, None]).sum(axis=0)
        train_ww += ((feat * feat) * sw[:, None]).sum(axis=0)
        train_weight += float(sw.sum())
        train_seen += 1
        if train_seen >= TRAIN_RIDGE_GROUPS:
            break

    for npz_index in range(y.shape[0]):
        cols = _read_columns(valid_pf, int(row_group[npz_index]))
        consume_valid(npz_index, cols)
        if (npz_index + 1) % 200 == 0:
            print(f"valid {npz_index + 1}/{y.shape[0]}", flush=True)

    feature_beta = _solve_ridge(train_xtx, train_xty, PRIMARY_ALPHA)
    del train_w, train_ww, train_weight

    eval_y = np.concatenate(stats["eval_y"], axis=0)
    eval_p = np.concatenate(stats["eval_p"], axis=0)
    eval_f = np.concatenate(stats["eval_f"], axis=0).astype(np.float64)
    eval_mask = np.ones(eval_y.shape[0], dtype=bool)
    incumbent_eval = _score(eval_y[:, None, :], eval_p[:, None, :], eval_mask[:, None])

    full_beta = _solve_ridge(stats["fit_x"], stats["fit_xy"], PRIMARY_ALPHA)
    full_pred = np.column_stack([np.ones(eval_f.shape[0]), eval_f]) @ full_beta
    full_pred = np.clip(full_pred, -CLIP, CLIP).astype(np.float32)
    full_eval = _score(eval_y[:, None, :], full_pred[:, None, :], eval_mask[:, None])

    # Control: intercept + the two incumbent predictions, which are the last two columns.
    ctrl_index = [0, d_feat + 1, d_feat + 2]
    # fit_x includes the intercept already at column 0, then features, then preds.
    ctrl_x = stats["fit_x"][np.ix_(ctrl_index, ctrl_index)]
    ctrl_xy = stats["fit_xy"][ctrl_index]
    ctrl_beta = _solve_ridge(ctrl_x, ctrl_xy, PRIMARY_ALPHA)
    ctrl_design = np.column_stack(
        [np.ones(eval_f.shape[0]), eval_f[:, d_feat], eval_f[:, d_feat + 1]]
    )
    ctrl_pred = np.clip(ctrl_design @ ctrl_beta, -CLIP, CLIP).astype(np.float32)
    ctrl_eval = _score(eval_y[:, None, :], ctrl_pred[:, None, :], eval_mask[:, None])

    # Feature-only train model scored on the same eval rows (features are the first d_feat).
    feat_design = np.column_stack([np.ones(eval_f.shape[0]), eval_f[:, :d_feat]])
    feat_pred = np.clip(feat_design @ feature_beta, -CLIP, CLIP).astype(np.float32)
    feat_eval = _score(eval_y[:, None, :], feat_pred[:, None, :], eval_mask[:, None])

    clock_names = ["00_neither", "10_p0_only", "01_p11_only", "11_both"]
    age_names = ["0", "1-7", "8-31", "32-127", "128+"]
    cell_names = [f"{age}/{clock}" for age in age_names for clock in clock_names]
    or_table = _cell_table(local_or, cell_names)
    and_table = _cell_table(local_and, cell_names)
    total_w = numer_w.sum(axis=1)
    total_n = numer.sum(axis=1)
    contribution = []
    for index, name in enumerate(cell_names):
        row: dict[str, float | str] = {"cell": name}
        for target_i, target_name in enumerate(("t0", "t1")):
            w = float(numer_w[target_i, index])
            n = float(numer[target_i, index])
            row[f"weight_share_{target_name}"] = w / total_w[target_i] if total_w[target_i] else 0.0
            row[f"numer_share_{target_name}"] = n / total_n[target_i] if total_n[target_i] else 0.0
            row[f"density_{target_name}"] = n / w if w > 1e-8 else 0.0
        contribution.append(row)

    label_by_clock = []
    for clock_id, name in enumerate(clock_names):
        unchanged = int(contingency[0, clock_id])
        changed = int(contingency[1, clock_id])
        label_by_clock.append(
            {
                "clock": name,
                "unchanged": unchanged,
                "changed": changed,
                "change_rate": changed / (changed + unchanged) if changed + unchanged else 0.0,
            }
        )

    payload = {
        "metric": "global_wp_clip2_v1",
        "blend": "0.5 * (c20261004_combo_s4_final_fp32 + c20261004_combo_head_s4_final_fp32)",
        "seconds": time.time() - started,
        "holdout_sequences": int(hold.sum()),
        "dev_sequences": int(dev.sum()),
        "scores": {
            "blend_full_valid": blend_score,
            "zero_full_valid": zero_score,
            "independent_lattice_snap": lattice_score,
            "sum_lattice_snap": sum_lattice_score,
            "blend_by_class": by_class,
        },
        "hawkes": {
            "mean_or_gap_rows": mean_gap,
            "beta": beta,
            "decay": decay,
            "train_sequences_for_beta": TRAIN_BETA_GROUPS,
            "train_sequences_for_feature_model": train_seen,
        },
        "clock_or_age_local_wp": or_table,
        "clock_and_age_local_wp": and_table,
        "global_mean_numerator_or_age": contribution,
        "label_change_by_clock": label_by_clock,
        "a6": a6_counts,
        "models": {
            "split": "non-holdout valid, sha256(seq_ix:wave1b) % 2, 0 fit / 1 eval",
            "alpha": PRIMARY_ALPHA,
            "fit_rows": stats["fit_rows"],
            "eval_rows": int(eval_y.shape[0]),
            "incumbent_eval_wp": incumbent_eval,
            "affine_incumbent_eval_wp": ctrl_eval,
            "features_plus_incumbent_eval_wp": full_eval,
            "delta_vs_affine": full_eval - ctrl_eval,
            "train_feature_only_on_eval_wp": feat_eval,
            "feature_names": list(FEATURE_NAMES) + list(CONTROL_NAMES),
        },
        "global_weighted_means": {"y": my.tolist(), "p": mp.tolist()},
    }
    (OUT / "wave1_anatomy.json").write_text(json.dumps(payload, indent=2))
    print("WAVE1_DONE", payload["models"]["delta_vs_affine"], payload["scores"]["blend_full_valid"]["wp"], flush=True)


if __name__ == "__main__":
    main()
