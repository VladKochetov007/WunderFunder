"""Numba GRU256x2 step (f32 / int16 / dynamic int8) timed against the Rust q8vnni engine.

    taskset -c 2 .venv/bin/python scripts/bench_numba_gru.py --weights ep3.npz --rows rows.f32

Same model and per-row work as native/examples/bench_int8.rs, minus the Rust feature pipeline: the
211 inputs are the 112 raw columns plus 99 event-like columns that are 85% zero (the real
sparsity), because only the zero pattern matters for timing. Prints us/row for
 - the whole sequence loop compiled inside numba (best case for numba), and
 - one Python call per row (what a scorer callback would pay).
Variants: `libm` uses math.exp/erf, `poly` uses the same polynomial exp as the Rust engine so LLVM
can vectorize the gates.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from numba import njit

H = 256


@njit(inline="always", fastmath=True)
def exp_poly(x):
    x = min(40.0, max(-40.0, x))
    n = np.rint(x / 0.6931471805599453)
    r = x - n * 0.6931471805599453
    p = 1.0 / 5040.0
    p = 1.0 / 720.0 + r * p
    p = 1.0 / 120.0 + r * p
    p = 1.0 / 24.0 + r * p
    p = 1.0 / 6.0 + r * p
    p = 0.5 + r * p
    p = 1.0 + r * p
    p = 1.0 + r * p
    return p * 2.0 ** n


@njit(inline="always", fastmath=True)
def sig(x, poly):
    if poly:
        return 1.0 / (1.0 + exp_poly(-x))
    return 1.0 / (1.0 + math.exp(-x))


@njit(inline="always", fastmath=True)
def gelu(x, poly):
    if poly:
        a = abs(x) * 0.7071067811865476
        t = 1.0 / (1.0 + 0.3275911 * a)
        pl = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        e = 1.0 - pl * exp_poly(-a * a)
        return 0.5 * x * (1.0 + (e if x >= 0 else -e))
    return 0.5 * x * (1.0 + math.erf(x * 0.7071067811865476))


@njit(fastmath=True)
def gemv_f32(W, b, x, out):
    for r in range(W.shape[0]):
        acc = 0.0
        for k in range(W.shape[1]):
            acc += W[r, k] * x[k]
        out[r] = acc + b[r]


@njit(fastmath=True)
def quant8(x, q):
    m = 0.0
    for k in range(x.shape[0]):
        m = max(m, abs(x[k]))
    if m == 0.0:
        q[:] = 0
        return 0.0
    inv = 127.0 / m
    for k in range(x.shape[0]):
        q[k] = np.int8(min(127.0, max(-127.0, np.rint(x[k] * inv))))
    return m / 127.0


@njit(fastmath=True)
def gemv_i8(W, ws, q, xs, b, out):
    for r in range(W.shape[0]):
        acc = np.int32(0)
        for k in range(W.shape[1]):
            acc += np.int32(W[r, k]) * np.int32(q[k])
        out[r] = acc * (ws[r] * xs) + b[r]


@njit(fastmath=True)
def quant16(x, q):
    m = 0.0
    for k in range(x.shape[0]):
        m = max(m, abs(x[k]))
    if m == 0.0:
        q[:] = 0
        return 0.0
    inv = 16383.0 / m
    for k in range(x.shape[0]):
        q[k] = np.int16(min(16383.0, max(-16383.0, np.rint(x[k] * inv))))
    return m / 16383.0


@njit(fastmath=True)
def gemv_i16(W, ws, q, xs, b, out):
    for r in range(W.shape[0]):
        acc = 0.0
        for c0 in range(0, W.shape[1], 2):
            # two products fit int32 exactly; flush to float per pair of columns like q16x
            v = np.int32(W[r, c0]) * np.int32(q[c0]) + np.int32(W[r, c0 + 1]) * np.int32(q[c0 + 1])
            acc += np.float32(v)
        out[r] = acc * (ws[r] * xs) + b[r]


@njit(fastmath=True)
def step(mode, poly, x, Wp, bp, W, WS, B, w1h, w1hs, w1z, w1zs, b1, w2, b2, h, z, gi, gh, a, qz, qh, out):
    # input projection: sparse columns, then GELU
    z[:] = bp
    for j in range(x.shape[0]):
        xj = x[j]
        if xj != 0.0:
            for r in range(H):
                z[r] += Wp[j, r] * xj
    for r in range(H):
        z[r] = gelu(z[r], poly)
    inp = z
    for k in range(2):
        if mode == 0:
            gemv_f32(W[2 * k], B[2 * k], inp, gi)
            gemv_f32(W[2 * k + 1], B[2 * k + 1], h[k], gh)
        elif mode == 1:
            xs = quant16(inp, qz)
            gemv_i16(W[2 * k], WS[2 * k], qz, xs, B[2 * k], gi)
            xs = quant16(h[k], qh)
            gemv_i16(W[2 * k + 1], WS[2 * k + 1], qh, xs, B[2 * k + 1], gh)
        else:
            xs = quant8(inp, qz)
            gemv_i8(W[2 * k], WS[2 * k], qz, xs, B[2 * k], gi)
            xs = quant8(h[k], qh)
            gemv_i8(W[2 * k + 1], WS[2 * k + 1], qh, xs, B[2 * k + 1], gh)
        for j in range(H):
            r = sig(gi[j] + gh[j], poly)
            u = sig(gi[H + j] + gh[H + j], poly)
            c = 2.0 * sig(2.0 * (gi[2 * H + j] + r * gh[2 * H + j]), poly) - 1.0
            h[k, j] = (1.0 - u) * c + u * h[k, j]
        inp = h[k]
    # head
    if mode == 0:
        gemv_f32(w1h, b1, h[1], a)
        tmp = np.empty(H, np.float32)
        gemv_f32(w1z, np.zeros(H, np.float32), z, tmp)
    elif mode == 1:
        xs = quant16(h[1], qh)
        gemv_i16(w1h, w1hs, qh, xs, b1, a)
        xs = quant16(z, qz)
        tmp = np.empty(H, np.float32)
        gemv_i16(w1z, w1zs, qz, xs, np.zeros(H, np.float32), tmp)
    else:
        xs = quant8(h[1], qh)
        gemv_i8(w1h, w1hs, qh, xs, b1, a)
        xs = quant8(z, qz)
        tmp = np.empty(H, np.float32)
        gemv_i8(w1z, w1zs, qz, xs, np.zeros(H, np.float32), tmp)
    for j in range(H):
        a[j] = gelu(a[j] + tmp[j], poly)
    for o in range(2):
        s = b2[o]
        for j in range(H):
            s += w2[o, j] * a[j]
        out[o] = s


@njit(fastmath=True)
def run(mode, poly, X, reset_every, params, state):
    Wp, bp, W, WS, B, w1h, w1hs, w1z, w1zs, b1, w2, b2 = params
    h, z, gi, gh, a, qz, qh, out = state
    acc = 0.0
    for t in range(X.shape[0]):
        if t % reset_every == 0:
            h[:] = 0.0
        step(mode, poly, X[t], Wp, bp, W, WS, B, w1h, w1hs, w1z, w1zs, b1, w2, b2, h, z, gi, gh, a, qz, qh, out)
        acc += out[0]
    return acc


def quantize_rows(w, bits):
    qmax = 127 if bits == 8 else 32767
    s = np.abs(w).max(axis=1) / qmax
    s[s == 0] = 1e-30
    dt = np.int8 if bits == 8 else np.int16
    return np.clip(np.rint(w / s[:, None]), -qmax, qmax).astype(dt), s.astype(np.float32)


def build(weights, mode):
    w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
    std, mean = w["std"], w["mean"]
    win = w["inp.weight"] / std[None, :]
    bp = (w["inp.bias"] - win @ mean).astype(np.float32)
    Wp = np.ascontiguousarray(win.T)  # (211, 256) columns
    mats, scales, biases = [], [], []
    for k in range(2):
        for name in ("weight_ih", "weight_hh"):
            m = w[f"gru.{name}_l{k}"]
            biases.append(w[f"gru.{name.replace('weight', 'bias')}_l{k}"])
            if mode == 0:
                mats.append(m)
                scales.append(np.ones(3 * H, np.float32))
            else:
                q, s = quantize_rows(m, 8 if mode == 2 else 16)
                mats.append(q)
                scales.append(s)
    head = w["head.0.weight"]
    parts = []
    for blk in (head[:, :H], head[:, H:]):
        blk = np.ascontiguousarray(blk)
        if mode == 0:
            parts.append((blk, np.ones(H, np.float32)))
        else:
            parts.append(quantize_rows(blk, 8 if mode == 2 else 16))
    params = (
        Wp, bp, np.stack(mats), np.stack(scales), np.stack(biases),
        parts[0][0], parts[0][1], parts[1][0], parts[1][1], w["head.0.bias"],
        np.ascontiguousarray(w["head.2.weight"]), w["head.2.bias"],
    )
    qdt = np.float32 if mode == 0 else (np.int8 if mode == 2 else np.int16)
    state = (
        np.zeros((2, H), np.float32), np.zeros(H, np.float32), np.zeros(3 * H, np.float32),
        np.zeros(3 * H, np.float32), np.zeros(H, np.float32), np.zeros(H, qdt), np.zeros(H, qdt),
        np.zeros(2, np.float32),
    )
    return params, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    weights = dict(np.load(args.weights))
    raw = np.fromfile(args.rows, dtype="<f4").reshape(-1, 112)
    rng = np.random.default_rng(0)
    events = (rng.random((len(raw), 99)) > 0.85) * rng.standard_normal((len(raw), 99))
    X = np.ascontiguousarray(np.concatenate([raw, events], axis=1).astype(np.float32))
    n = len(X)
    for mode, name in ((0, "f32"), (1, "int16"), (2, "int8")):
        params, state = build(weights, mode)
        for poly in (False, True):
            run(mode, poly, X[:200], 20000, params, state)  # compile + warm
            times = []
            for _ in range(args.reps):
                t = time.perf_counter()
                run(mode, poly, X, 20000, params, state)
                times.append((time.perf_counter() - t) / n * 1e6)
            # one numba call per row from Python
            Wp, bp, W, WS, B, w1h, w1hs, w1z, w1zs, b1, w2, b2 = params
            h, z, gi, gh, a, qz, qh, out = state
            h[:] = 0
            t = time.perf_counter()
            for i in range(5000):
                step(mode, poly, X[i], Wp, bp, W, WS, B, w1h, w1hs, w1z, w1zs, b1, w2, b2, h, z, gi, gh, a, qz, qh, out)
            percall = (time.perf_counter() - t) / 5000 * 1e6
            print(f"{name:6s} {'poly' if poly else 'libm'}: loop median {np.median(times):7.2f} us/row  (min {min(times):.2f}); per-call from Python {percall:7.2f} us/row", flush=True)


if __name__ == "__main__":
    main()
