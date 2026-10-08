"""Causal heuristic feature library (research): building blocks for symbolic regression.

Every feature at row t depends only on rows <= t of the same sequence and is an O(1)
recursive update, so any feature that proves useful can be ported to the streaming
predictor unchanged. ``heuristic_features(x)`` returns (T, K) float32 and ``NAMES``.

Families (column indices follow utils.FEATURE_COLUMNS; the code index is an identifier,
not a depth level, so p0/p11 and v0/v11 are used as representative codes):
  recency      log1p rows since the last event of each kind
  rates        EMAs of event indicators at half-lives 8, 64, 512 rows
  held deltas  change of representative codes at the last i0 book event (held until next)
  deviations   x - EMA(x) at half-lives 32 and 256 rows for representative columns
  zscore       online per-sequence z-scores of representative columns
  depth        sums of the 11 bid / 11 ask volume codes, their difference and its EMA drift
  trades       EMAs of trade-volume code sums, trades since the last i0 book event
  cross        i1 book events and i1 code drift since the last i0 book event
  clocks       a-columns: raw, sentinel flags, EMA deviations
  position     log1p(step)
"""

from __future__ import annotations

import numba as nb
import numpy as np

SENTINEL = -5.19934
HL_RATE = (8.0, 64.0, 512.0)
HL_DEV = (32.0, 256.0)

EVENTS = ("book", "bid", "ask", "volbid", "volask", "trade", "i1book", "i1vol", "i1trade")
DEV_COLS = {
    "p0": 0, "p4": 4, "p11": 11, "v0": 22, "v1": 23, "v11": 33, "v12": 34, "dp0": 44, "dv0": 48,
    "i1p0": 52, "i1p11": 63, "i1v0": 74, "i1v11": 85,
    "a0": 104, "a1": 105, "a2": 106, "a3": 107, "a4": 108, "a5": 109, "a6": 110, "a7": 111,
}
Z_COLS = {"v0": 22, "v11": 33, "a0": 104, "a2": 106, "a3": 107, "a4": 108}
HELD_COLS = {"p0": 0, "p4": 4, "p11": 11, "p15": 15, "v0": 22, "v11": 33}


def _names() -> list[str]:
    n = [f"since_{e}" for e in EVENTS]
    n += [f"rate_{e}_hl{int(h)}" for e in ("book", "trade", "i1book", "volbid", "volask") for h in HL_RATE]
    n += ["log_n_book", "book_per_row"]
    n += [f"held_d_{c}" for c in HELD_COLS] + [f"held_sign_{c}" for c in HELD_COLS]
    n += [f"absd_p0_hl{int(h)}" for h in (64.0, 512.0)] + [f"drift_p0_hl{int(h)}" for h in (64.0, 512.0)]
    n += [f"dev_{c}_hl{int(h)}" for c in DEV_COLS for h in HL_DEV]
    n += [f"z_{c}" for c in Z_COLS]
    n += ["depth_bid", "depth_ask", "depth_imb", "depth_imb_dev64", "depth_imb_d_at_book"]
    n += ["tradevol_hl16", "tradevol_hl128", "trades_since_book", "last_dp0"]
    n += ["i1book_since_i0book", "i1p0_drift_since_i0book", "i1p11_drift_since_i0book"]
    n += [f"raw_a{i}" for i in range(8)] + [f"sent_a{i}" for i in (5, 6, 7)]
    n += ["log_step"]
    return n


NAMES = _names()
K = len(NAMES)


@nb.njit(cache=True)
def _alpha(hl):
    return 1.0 - 0.5 ** (1.0 / hl)


@nb.njit(cache=True)
def _compute(x, out, dev_idx, z_idx, held_idx):
    T = x.shape[0]
    n_ev = 9
    last_ev = np.zeros(n_ev, np.int64)
    rates = np.zeros((5, 3))
    a_rate = np.array([_alpha(8.0), _alpha(64.0), _alpha(512.0)])
    a_dev = np.array([_alpha(32.0), _alpha(256.0)])
    ema_dev = np.zeros((len(dev_idx), 2))
    z_n = 0.0
    z_mean = np.zeros(len(z_idx))
    z_m2 = np.zeros(len(z_idx))
    held = np.zeros(len(held_idx))
    absd = np.zeros(2)
    drift = np.zeros(2)
    a_pd = np.array([_alpha(64.0), _alpha(512.0)])
    imb_ema = 0.0
    a64 = _alpha(64.0)
    imb_at_book = 0.0
    imb_d_at_book = 0.0
    tv = np.zeros(2)
    a_tv = np.array([_alpha(16.0), _alpha(128.0)])
    trades_since_book = 0.0
    last_dp0 = 0.0
    i1_since = 0.0
    i1p0_at_book = x[0, 52]
    i1p11_at_book = x[0, 63]
    n_book = 0.0
    for t in range(T):
        row = x[t]
        ev = np.zeros(n_ev, np.bool_)
        if t > 0:
            prev = x[t - 1]
            for j in range(22):
                if row[j] != prev[j]:
                    ev[0] = True
                    break
            ev[1] = row[0] != prev[0]
            ev[2] = row[11] != prev[11]
            for j in range(22, 33):
                if row[j] != prev[j]:
                    ev[3] = True
                    break
            for j in range(33, 44):
                if row[j] != prev[j]:
                    ev[4] = True
                    break
            for j in range(48, 52):
                if row[j] != prev[j]:
                    ev[5] = True
                    break
            for j in range(52, 74):
                if row[j] != prev[j]:
                    ev[6] = True
                    break
            for j in range(74, 96):
                if row[j] != prev[j]:
                    ev[7] = True
                    break
            for j in range(100, 104):
                if row[j] != prev[j]:
                    ev[8] = True
                    break
            if ev[0]:
                n_book += 1.0
                for k in range(len(held_idx)):
                    held[k] = row[held_idx[k]] - prev[held_idx[k]]
                d = row[0] - prev[0]
                for k in range(2):
                    absd[k] += a_pd[k] * (abs(d) - absd[k])
                    drift[k] += a_pd[k] * (d - drift[k])
                i1_since = 0.0
                i1p0_at_book = row[52]
                i1p11_at_book = row[63]
                trades_since_book = 0.0
            else:
                for k in range(2):
                    absd[k] *= 1.0 - a_pd[k]
                    drift[k] *= 1.0 - a_pd[k]
            if ev[6]:
                i1_since += 1.0
            if ev[5]:
                trades_since_book += 1.0
                last_dp0 = row[44]
        for e in range(n_ev):
            if ev[e]:
                last_ev[e] = t
        rate_ev = (ev[0], ev[5], ev[6], ev[3], ev[4])
        for r in range(5):
            v = 1.0 if rate_ev[r] else 0.0
            for h in range(3):
                rates[r, h] += a_rate[h] * (v - rates[r, h])
        for k in range(len(dev_idx)):
            v = row[dev_idx[k]]
            for h in range(2):
                if t == 0:
                    ema_dev[k, h] = v
                else:
                    ema_dev[k, h] += a_dev[h] * (v - ema_dev[k, h])
        z_n += 1.0
        for k in range(len(z_idx)):
            v = row[z_idx[k]]
            dlt = v - z_mean[k]
            z_mean[k] += dlt / z_n
            z_m2[k] += dlt * (v - z_mean[k])
        db = 0.0
        da = 0.0
        for j in range(22, 33):
            db += row[j]
        for j in range(33, 44):
            da += row[j]
        imb = db - da
        if t == 0:
            imb_ema = imb
            imb_at_book = imb
        imb_ema += a64 * (imb - imb_ema)
        if ev[0]:
            imb_d_at_book = imb - imb_at_book
            imb_at_book = imb
        tvol = 0.0
        if ev[5]:
            for j in range(48, 52):
                tvol += row[j]
        for h in range(2):
            tv[h] += a_tv[h] * (tvol - tv[h])
        # write
        c = 0
        for e in range(n_ev):
            out[t, c] = np.log1p(t - last_ev[e])
            c += 1
        for r in range(5):
            for h in range(3):
                out[t, c] = rates[r, h]
                c += 1
        out[t, c] = np.log1p(n_book)
        out[t, c + 1] = n_book / (t + 1.0)
        c += 2
        for k in range(len(held_idx)):
            out[t, c] = held[k]
            c += 1
        for k in range(len(held_idx)):
            out[t, c] = np.sign(held[k])
            c += 1
        for k in range(2):
            out[t, c] = absd[k]
            c += 1
        for k in range(2):
            out[t, c] = drift[k]
            c += 1
        for k in range(len(dev_idx)):
            for h in range(2):
                out[t, c] = row[dev_idx[k]] - ema_dev[k, h]
                c += 1
        for k in range(len(z_idx)):
            sd = np.sqrt(z_m2[k] / z_n) if z_n > 1 else 0.0
            out[t, c] = (row[z_idx[k]] - z_mean[k]) / (sd + 1e-3)
            c += 1
        out[t, c] = db
        out[t, c + 1] = da
        out[t, c + 2] = imb
        out[t, c + 3] = imb - imb_ema
        out[t, c + 4] = imb_d_at_book
        c += 5
        out[t, c] = tv[0]
        out[t, c + 1] = tv[1]
        out[t, c + 2] = trades_since_book
        out[t, c + 3] = last_dp0
        c += 4
        out[t, c] = i1_since
        out[t, c + 1] = row[52] - i1p0_at_book
        out[t, c + 2] = row[63] - i1p11_at_book
        c += 3
        for j in range(8):
            out[t, c] = row[104 + j]
            c += 1
        for j in (109, 110, 111):
            out[t, c] = 1.0 if abs(row[j] - SENTINEL) < 1e-4 else 0.0
            c += 1
        out[t, c] = np.log1p(t)
        c += 1


def heuristic_features(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros((len(x), K), np.float64)
    _compute(x, out, np.array(list(DEV_COLS.values()), np.int64), np.array(list(Z_COLS.values()), np.int64),
             np.array(list(HELD_COLS.values()), np.int64))
    return out.astype(np.float32)
