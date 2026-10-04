"""Target-structure laws on TRAIN: fit tables on fit components, grade on held-out components.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/de_target_law.py

Targets are integer tick multiples (class quantum). Fit tables use fit-component TRAIN rows only:
  spread table  code -> (class, s) with s = -(t0+t1) on both-nonpositive rows (support >= 3, confidence >= .95);
  move table    (code_prev, code_next) -> (d_ask, d_bid) = (-dt0, +dt1) at interior-to-interior code events
                (support >= 3, confidence >= .80); quote paths are cumulative sums, unknown pairs break them.
Event marker E (feature-only, E-273): a6 == float32 lower endpoint -5.199337482452393.
Laws graded on held-out components with |clip(t)| weights; unknown mass is reported, never imputed.
  H1  no E at row r  =>  (t0, t1) unchanged from r-1
  H2  both targets <= 0  =>  t0 + t1 = -s(code)
  H3  outside rows: loser = -(s + 2 clip(m, 1, winner)), m = loser's first adverse move (variants)
  H4  winner = round(mean mid over the next N quote-move events - entry quote) (N, offset, rounding chosen on fit)
Writes research/campaign_2026-10-01/data_eval/target_law_tables.npz and target_law_train.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numba as nb
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "campaign_2026-10-01" / "data_eval"
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
PRICE = [f"i0_p{i}" for i in range(22)]
ENDPOINT = np.float32(-5.199337482452393)
N_FIT, N_TEST, PER_COMPONENT = 60, 40, 3
NS = tuple(range(1, 13)) + (16, 20, 24, 32)
ROUNDINGS = ("half_up", "half_down", "half_even", "floor", "ceil")
MULT = np.random.default_rng(20261001).integers(1, 2**62, size=22, dtype=np.int64) | 1


def code_hash(x: np.ndarray) -> np.ndarray:
    """Row id of the 22-field i0 price code rounded to 4 decimals (int64 polynomial hash)."""
    q = np.rint(x.astype(np.float64) * 1e4).astype(np.int64)
    with np.errstate(over="ignore"):
        h = (q * MULT[None]).sum(1)
        h ^= h >> np.int64(29)
    return h


def select() -> dict:
    """Component-disjoint fit/held-out TRAIN sequences per class (hash order, no targets read)."""
    manifest = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
    comp = {s["row_group"]: s["component"] for s in manifest["sequences"]["train"]}
    with np.load(ROOT / "cache" / "train_targets.npz") as z:
        rg, cls = z["row_group"].astype(int), z["market_class"].astype(int)
    prior = json.loads((ROOT / "research/campaign_2026-09-30/data_reconstruction/selection.json").read_text())
    prior_comp = {comp[g] for g in prior["fit_train_groups"] + prior["test_train_groups"] if g in comp}
    side = {c: int(hashlib.sha256(f"de-law:{c}".encode()).hexdigest(), 16) % 10 < 6 for c in set(comp.values())}
    order = sorted(rg, key=lambda g: hashlib.sha256(f"de-law-seq:{g}".encode()).hexdigest())
    cls_of = dict(zip(rg, cls, strict=True))
    picked = {"fit": defaultdict(list), "test": defaultdict(list)}
    used = Counter()
    for g in order:
        if g not in comp or g == 528:
            continue
        c = comp[g]
        part = "fit" if side[c] else "test"
        if part == "test" and c in prior_comp:
            continue
        k = cls_of[g]
        if len(picked[part][k]) >= (N_FIT if part == "fit" else N_TEST) or used[c] >= PER_COMPONENT:
            continue
        picked[part][k].append(int(g))
        used[c] += 1
    return {p: {k: v for k, v in sorted(d.items())} for p, d in picked.items()} | {"component": comp}


def read(source: pq.ParquetFile, group: int):
    t = source.read_row_group(group, columns=PRICE + ["a6", "t0", "t1"], use_threads=False)
    x = np.column_stack([t.column(k).to_numpy() for k in PRICE]).astype(np.float32)
    y = np.column_stack([t.column(k).to_numpy() for k in ("t0", "t1")]).astype(np.float64)
    err = np.max(np.abs(np.rint(y[None] / QUANTA[:, None, None]) * QUANTA[:, None, None] - y), axis=(1, 2))
    cls = int(err.argmin())
    if err[cls] > 0.002:
        raise ValueError(f"lattice mismatch in group {group}")
    a6 = t.column("a6").to_numpy().astype(np.float32)
    return code_hash(x), a6 == ENDPOINT, np.rint(y / QUANTA[cls]).astype(np.int64), cls


def accept(counts: dict, support: int, confidence: float) -> dict:
    out = {}
    for key, c in counts.items():
        value, n = c.most_common(1)[0]
        if n >= support and n / c.total() >= confidence:
            out[key] = value
    return out


def fit_tables(fit_seqs) -> tuple[dict, dict]:
    spread, moves = defaultdict(Counter), defaultdict(Counter)
    for code, _, y, cls in fit_seqs:
        interior = (y[:, 0] <= 0) & (y[:, 1] <= 0)
        s = -(y[:, 0] + y[:, 1])
        rows = np.flatnonzero(interior & (s >= 1))[::3]
        for c, v in zip(code[rows].tolist(), s[rows].tolist(), strict=True):
            spread[c][(cls, v)] += 1
        ev = np.flatnonzero(code[1:] != code[:-1]) + 1
        ev = ev[interior[ev] & interior[ev - 1]]
        d = np.stack([-(y[ev, 0] - y[ev - 1, 0]), y[ev, 1] - y[ev - 1, 1]], 1)
        for a, b, m in zip(code[ev - 1].tolist(), code[ev].tolist(), map(tuple, d.tolist()), strict=True):
            moves[(a, b)][m] += 1
    return accept(spread, 3, 0.95), accept(moves, 3, 0.80)


def quote_path(code: np.ndarray, moves: dict):
    """Relative ask/bid ticks per row and segment id (increments at every unknown code transition)."""
    ev = np.flatnonzero(code[1:] != code[:-1]) + 1
    da = np.zeros(len(code), np.int64)
    db = np.zeros(len(code), np.int64)
    unk = np.zeros(len(code), np.int64)
    for e, a, b in zip(ev.tolist(), code[ev - 1].tolist(), code[ev].tolist(), strict=True):
        m = moves.get((a, b))
        if m is None:
            unk[e] = 1
        else:
            da[e], db[e] = m
    return np.cumsum(da), np.cumsum(db), np.cumsum(unk), ev


@nb.njit(cache=True)
def first_cross(level: np.ndarray, sign: int, strict_step: bool) -> np.ndarray:
    """For each row r: first row j > r where sign*(level[j] - level[r]) > 0 (strict_step False), or where
    sign*(level[j] - level[j-1]) > 0 (strict_step True: first adverse step). -1 if none."""
    n = len(level)
    out = np.full(n, -1, np.int64)
    if strict_step:
        nxt = -1
        for j in range(n - 1, 0, -1):
            if sign * (level[j] - level[j - 1]) > 0:
                nxt = j
            out[j - 1] = nxt
        return out
    stack = np.empty(n, np.int64)
    top = 0
    for j in range(n - 1, -1, -1):
        while top > 0 and sign * (level[stack[top - 1]] - level[j]) <= 0:
            top -= 1
        out[j] = stack[top - 1] if top > 0 else -1
        stack[top] = j
        top += 1
    return out


def round_mode(v2: np.ndarray, den: int, mode: str) -> np.ndarray:
    """Round v2/den (exact rationals; v2 integer numerators) to integers."""
    num = v2.astype(np.int64)
    fl = np.floor_divide(num, den)
    rem = num - fl * den
    if mode == "floor":
        return fl
    if mode == "ceil":
        return fl + (rem > 0)
    up = 2 * rem > den
    half = 2 * rem == den
    if mode == "half_up":
        return fl + (up | half)
    if mode == "half_down":
        return fl + up
    return fl + (up | (half & (fl % 2 == 1)))


def h4_predictions(ask, bid, seg, quote_rows, rows, n, offset):
    """2*N*mean future mid (numerator over den=2N) of the next N quote-move rows after r (offset 0 also counts r)."""
    mid2 = ask + bid
    q = quote_rows
    cs = np.r_[0, np.cumsum(mid2[q])]
    start = np.searchsorted(q, rows, side="left" if offset == 0 else "right")
    ok = start + n <= len(q)
    end = np.where(ok, start + n, 0)
    total = cs[np.where(ok, end, 0)] - cs[np.where(ok, start, 0)]
    last = q[np.where(ok, end - 1, 0)]
    ok &= seg[last] == seg[rows]
    return total, ok


def h3_m(ask, bid, seg, rows, long_loser, variant):
    """Loser's first adverse move m (numerator over den 2: half ticks) and known flag."""
    mid2 = ask + bid
    if variant.startswith("mid"):
        lv = mid2
        scale = 1
    else:
        lv = (ask if variant.endswith("ask") else bid) * 2
        scale = 1
    strict = variant.startswith(("mid_step", "step"))
    out_m = np.zeros(len(rows), np.int64)
    ok = np.zeros(len(rows), bool)
    for sign, sel in ((-1, long_loser), (1, ~long_loser)):  # long loses on down moves, short on up moves
        j = first_cross(lv, sign, strict)[rows[sel]]
        good = j >= 0
        jj = np.where(good, j, 0)
        ref = np.where(strict, lv[np.maximum(jj - 1, 0)], lv[rows[sel]])
        mv = sign * (lv[jj] - ref) * scale
        good &= seg[jj] == seg[rows[sel]]
        out_m[sel] = mv
        ok[sel] = good
    return out_m, ok


def weights(y, cls):
    return np.minimum(np.abs(y * QUANTA[cls]), 2.0)


def grade(seqs, spread, moves, collect_fit_choice=None):
    """Return per-class weighted tallies for all laws/variants on these sequences."""
    tally = defaultdict(lambda: np.zeros(3))  # (key) -> [exact weight, known weight, unknown weight]

    def add(key, w, exact, known):
        tally[key] += [float((w * (exact & known)).sum()), float((w * known).sum()), float((w * ~known).sum())]

    for code, endpoint, y, cls in seqs:
        T = len(y)
        r = np.arange(100, T)
        w = weights(y, cls)
        wsum = w.sum(1)
        changed = (y[r] != y[r - 1]).any(1)
        code_ev = code[r] != code[r - 1]
        add((cls, "H1_noE_unchanged"), wsum[r][~endpoint[r]], ~changed[~endpoint[r]], np.ones((~endpoint[r]).sum(), bool))
        add((cls, "H1_nocode_unchanged"), wsum[r][~code_ev], ~changed[~code_ev], np.ones((~code_ev).sum(), bool))
        add((cls, "H1_change_has_E"), wsum[r][changed], endpoint[r][changed], np.ones(changed.sum(), bool))
        add((cls, "H1_change_has_code_event"), wsum[r][changed], code_ev[changed], np.ones(changed.sum(), bool))
        add((cls, "H1_E_has_change"), wsum[r][endpoint[r]], changed[endpoint[r]], np.ones(endpoint[r].sum(), bool))
        rr = np.arange(99, T)
        sp = np.array([spread.get(c, (-1, -1))[1] for c in code[rr].tolist()])
        cls_hat = np.array([spread.get(c, (-1, -1))[0] for c in code[rr].tolist()])
        known_s = sp >= 1
        add((cls, "class_from_code"), wsum[rr], cls_hat == cls, known_s)
        yy = y[rr]
        interior = (yy[:, 0] <= 0) & (yy[:, 1] <= 0)
        add((cls, "H2_interior_sum_eq_minus_spread"), wsum[rr][interior], (-(yy[:, 0] + yy[:, 1]) == sp)[interior], known_s[interior])
        ask, bid, seg, _ = quote_path(code, moves)
        # anchor bid per segment so that ask - bid equals the tabled current spread (modal offset)
        sp_all = np.full(T, -1)
        sp_all[rr] = sp
        offset_all = ask - bid - sp_all
        good_seg = np.zeros(seg.max() + 1, bool)
        shift = np.zeros(seg.max() + 1, np.int64)
        for g in np.unique(seg):
            on = (seg == g) & (sp_all >= 1)
            if on.any():
                vals, cnt = np.unique(offset_all[on], return_counts=True)
                shift[g], good_seg[g] = vals[cnt.argmax()], True
                tally[(cls, "path_spread_consistency")] += [float(cnt.max()), float(cnt.sum()), 0.0]
        bid = bid + shift[seg]
        seg = np.where(good_seg[seg], seg, -1 - np.arange(T))  # rows of unanchored segments never match
        outside = ~interior
        o = rr[outside]
        yo = y[o]
        win_long = yo[:, 0] > 0  # t0 winner: long side wins, short (t1) loses
        W = np.where(win_long, yo[:, 0], yo[:, 1])
        L = np.where(win_long, yo[:, 1], yo[:, 0])
        s_o = sp[outside]
        wl = np.where(win_long, w[o, 1], w[o, 0])
        ww = np.where(win_long, w[o, 0], w[o, 1])
        ks = s_o >= 1
        add((cls, "H3_const_offset2"), wl, L == -(s_o + 2), ks)
        long_loser = ~win_long
        for variant in ("step_ask", "step_bid", "mid_step", "cross_ask", "cross_bid", "mid_cross"):
            m2, okm = h3_m(ask, bid, seg, o, long_loser, variant)  # m2 = 2*m in half ticks
            m2c = np.clip(m2, 2, 2 * W)
            add((cls, f"H3_{variant}"), wl, L == -(s_o + m2c), ks & okm)
            add((cls, f"H3_{variant}_noclip"), wl, L == -(s_o + m2), ks & okm)
            add((cls, f"H3_{variant}_offset2_same_rows"), wl, L == -(s_o + 2), ks & okm)
        quote_rows = np.flatnonzero(endpoint)
        quote_rows = quote_rows[quote_rows > 0]
        sign = np.where(win_long, 1, -1)
        entry2 = np.where(win_long, 2 * ask[o], 2 * bid[o])
        mid2q = (ask + bid)[quote_rows]
        for n in NS:
            for offset in (0, 1):
                tot, okn = h4_predictions(ask, bid, seg, quote_rows, o, n, offset)
                num = sign * (tot - n * entry2)  # 2N * (mean mid - entry), signed for the winner
                for mode in ROUNDINGS:
                    pred = round_mode(num, 2 * n, mode)
                    add((cls, f"H4_N{n}_o{offset}_{mode}"), ww, pred == W, okn)
                # alternatives: mid at the N-th quote event; extreme mid over the next N quote events
                start = np.searchsorted(quote_rows, o, side="left" if offset == 0 else "right")
                if len(mid2q) >= n:
                    win = np.lib.stride_tricks.sliding_window_view(mid2q, n)
                    hi, lo = win.max(1), win.min(1)
                    okw = okn & (start + n <= len(mid2q))
                    st = np.where(okw, start, 0)
                    nth = sign * (mid2q[np.where(okw, start + n - 1, 0)] - entry2)
                    ext = sign * (np.where(win_long, hi[st], lo[st]) - entry2)
                    for mode in ("half_up", "half_down", "floor", "ceil"):
                        add((cls, f"H4nth_N{n}_o{offset}_{mode}"), ww, round_mode(nth, 2, mode) == W, okw)
                        add((cls, f"H4ext_N{n}_o{offset}_{mode}"), ww, round_mode(ext, 2, mode) == W, okw)
                # interior common reference F = round(mean mid): t0 = F - ask, t1 = bid - F
                tot_i, oki = h4_predictions(ask, bid, seg, quote_rows, rr[interior], n, offset)
                for mode in ("half_up", "half_even", "floor", "ceil"):
                    F = round_mode(tot_i, 2 * n, mode)
                    ex = (F - ask[rr[interior]] == yy[interior, 0]) & (bid[rr[interior]] - F == yy[interior, 1])
                    add((cls, f"H4int_N{n}_o{offset}_{mode}"), wsum[rr][interior], ex, oki)
    return tally


def summarize(tally) -> dict:
    out = defaultdict(dict)
    for (cls, key), (ex, kn, un) in tally.items():
        out[key][str(cls)] = {"exact_share_of_known": ex / kn if kn > 0 else None,
                              "unknown_share": un / (kn + un) if kn + un > 0 else None,
                              "exact_share_of_all": ex / (kn + un) if kn + un > 0 else None,
                              "weight_known": kn, "weight_unknown": un}
    for key in out:
        tot = np.sum([[v["weight_known"], v["weight_unknown"]] for v in out[key].values()], 0)
        ex = sum((v["exact_share_of_known"] or 0) * v["weight_known"] for v in out[key].values())
        out[key]["all"] = {"exact_share_of_known": ex / tot[0] if tot[0] else None,
                           "unknown_share": tot[1] / tot.sum() if tot.sum() else None}
    return dict(out)


def main() -> None:
    sel = select()
    source = pq.ParquetFile(ROOT / "datasets" / "train.parquet")
    fit = [read(source, g) for k in range(4) for g in sel["fit"][k]]
    print("fit read", len(fit), flush=True)
    test = [read(source, g) for k in range(4) for g in sel["test"][k]]
    print("test read", len(test), flush=True)
    spread, moves = fit_tables(fit)
    print("tables", len(spread), len(moves), flush=True)
    fit_sum = summarize(grade(fit, spread, moves))
    print("fit graded", flush=True)
    test_sum = summarize(grade(test, spread, moves))
    # choose H4 (N, offset, rounding) per class on fit; H3 variant per class on fit
    choice = {}
    for cls in map(str, range(4)):
        h4 = {k: v[cls]["exact_share_of_known"] or 0 for k, v in fit_sum.items() if k.startswith("H4") and not k.startswith("H4int") and cls in v}
        fam = {}
        for f in ("H4_N", "H4nth_", "H4ext_", "H4int_"):
            cand = {k: v[cls]["exact_share_of_known"] or 0 for k, v in fit_sum.items() if k.startswith(f) and cls in v}
            b = max(cand, key=cand.get)
            fam[f] = {"variant": b, "fit_exact": cand[b], "test": test_sum[b][cls]}
            tn = {"0": 4, "1": 6, "2": 6, "3": 8}[cls]  # teammate-reported N, best offset/rounding on fit
            cand_t = {k: c for k, c in cand.items() if f"N{tn}_" in k}
            if cand_t:
                bt = max(cand_t, key=cand_t.get)
                fam[f + "teammateN"] = {"variant": bt, "fit_exact": cand_t[bt], "test": test_sum[bt][cls]}
        h3 = {k: v[cls]["exact_share_of_known"] or 0 for k, v in fit_sum.items()
              if k.startswith("H3_") and not k.endswith("same_rows") and cls in v}
        best4, best3 = max(h4, key=h4.get), max(h3, key=h3.get)
        choice[cls] = {"H4": best4, "H4_fit_exact": h4[best4], "H4_test": test_sum[best4][cls],
                       "H3": best3, "H3_fit_exact": h3[best3], "H3_test": test_sum[best3][cls], "H4_families": fam}
    comp = sel.pop("component")
    report = {
        "selection": {p: {str(k): v for k, v in d.items()} for p, d in sel.items()},
        "fit_components": sorted({comp[g] for d in sel["fit"].values() for g in d}),
        "test_components": sorted({comp[g] for d in sel["test"].values() for g in d}),
        "n_spread_codes": len(spread), "n_move_pairs": len(moves),
        "fit_selected": choice, "test": test_sum, "fit": fit_sum,
    }
    (OUT / "target_law_train.json").write_text(json.dumps(report, indent=1) + "\n")
    keys = np.array(list(spread.keys()), np.int64)
    np.savez(OUT / "target_law_tables.npz", spread_code=keys,
             spread_class=np.array([spread[k][0] for k in keys.tolist()]),
             spread_ticks=np.array([spread[k][1] for k in keys.tolist()]),
             move_prev=np.array([k[0] for k in moves], np.int64), move_next=np.array([k[1] for k in moves], np.int64),
             move_d=np.array(list(moves.values()), np.int64), mult=MULT)
    for key in ("H1_noE_unchanged", "H1_nocode_unchanged", "H1_change_has_E", "H1_E_has_change", "class_from_code",
                "H2_interior_sum_eq_minus_spread", "H3_const_offset2"):
        print(key, {c: (round(v["exact_share_of_known"], 5) if v["exact_share_of_known"] is not None else None,
                        round(v["unknown_share"], 4) if v["unknown_share"] is not None else None)
                    for c, v in test_sum[key].items()})
    for cls, c in choice.items():
        print(cls, c["H3"], round(c["H3_fit_exact"], 4), round(c["H3_test"]["exact_share_of_known"], 4),
              round(c["H3_test"]["unknown_share"], 4), "|", c["H4"], round(c["H4_fit_exact"], 4),
              round(c["H4_test"]["exact_share_of_known"], 4), round(c["H4_test"]["unknown_share"], 4))
        for f, v in c["H4_families"].items():
            print("   ", f, v["variant"], round(v["fit_exact"], 4), round(v["test"]["exact_share_of_known"], 4),
                  round(v["test"]["unknown_share"], 4))


if __name__ == "__main__":
    sys.exit(main())
