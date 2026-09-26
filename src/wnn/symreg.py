"""SISSO-style symbolic regression for explanation (research).

Given base features X (n, K), a target y and row weights w, build candidate expressions
(level 1: unary transforms of standardized features; level 2: binary combinations of the
best level-1 candidates), screen them by |weighted Pearson| with the current residual, and
select a sparse additive model greedily, refitting weighted least squares after each term
and re-screening against the new residual. Every candidate is evaluated in one pass over
the rows inside numba ``prange`` loops, so ~10^5 candidates x 2*10^5 rows run in seconds.

The metric-aligned choice is w = |clip(y)| (global weighted Pearson); any weights work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numba as nb
import numpy as np

UNARY = ("id", "abs", "sq", "ssqrt", "slog", "tanh", "relu", "nrelu", "sign", "bump")
BINARY = ("add", "sub", "mul", "div", "max", "min", "gate")


@nb.njit(inline="always")
def _unary(op, v):
    if op == 0:
        return v
    if op == 1:
        return abs(v)
    if op == 2:
        return v * v
    if op == 3:
        return np.sign(v) * np.sqrt(abs(v))
    if op == 4:
        return np.sign(v) * np.log1p(abs(v))
    if op == 5:
        return np.tanh(v)
    if op == 6:
        return v if v > 0 else 0.0
    if op == 7:
        return v if v < 0 else 0.0
    if op == 8:
        return np.sign(v)
    return np.exp(-v * v)


@nb.njit(inline="always")
def _binary(op, a, b):
    if op == 0:
        return a + b
    if op == 1:
        return a - b
    if op == 2:
        return a * b
    if op == 3:
        return a / (1.0 + abs(b))
    if op == 4:
        return a if a > b else b
    if op == 5:
        return a if a < b else b
    return a if b > 0 else 0.0


@nb.njit(inline="always")
def _corr_acc(sw, swv, swvv, swvr, swr, swrr):
    mv = swv / sw
    mr = swr / sw
    cov = swvr / sw - mv * mr
    vv = swvv / sw - mv * mv
    vr = swrr / sw - mr * mr
    if vv <= 1e-12 or vr <= 1e-12:
        return 0.0
    return cov / np.sqrt(vv * vr)


@nb.njit(parallel=True, cache=True)
def _screen_unary(Z, r, w, n_ops):
    n, k = Z.shape
    out = np.zeros((k, n_ops))
    sw = 0.0
    swr = 0.0
    swrr = 0.0
    for i in range(n):
        sw += w[i]
        swr += w[i] * r[i]
        swrr += w[i] * r[i] * r[i]
    for j in nb.prange(k):
        for op in range(n_ops):
            swv = 0.0
            swvv = 0.0
            swvr = 0.0
            for i in range(n):
                v = _unary(op, Z[i, j])
                wv = w[i] * v
                swv += wv
                swvv += wv * v
                swvr += wv * r[i]
            out[j, op] = _corr_acc(sw, swv, swvv, swvr, swr, swrr)
    return out


@nb.njit(parallel=True, cache=True)
def _screen_binary(V, r, w, n_ops):
    n, m = V.shape
    out = np.zeros((m, m, n_ops))
    sw = 0.0
    swr = 0.0
    swrr = 0.0
    for i in range(n):
        sw += w[i]
        swr += w[i] * r[i]
        swrr += w[i] * r[i] * r[i]
    for a in nb.prange(m):
        for b in range(m):
            if a == b:
                continue
            for op in range(n_ops):
                if (op == 0 or op == 2 or op == 4 or op == 5) and b < a:
                    continue  # commutative: evaluated once
                swv = 0.0
                swvv = 0.0
                swvr = 0.0
                for i in range(n):
                    v = _binary(op, V[i, a], V[i, b])
                    wv = w[i] * v
                    swv += wv
                    swvv += wv * v
                    swvr += wv * r[i]
                out[a, b, op] = _corr_acc(sw, swv, swvv, swvr, swr, swrr)
    return out


@nb.njit(parallel=True, cache=True)
def _apply_unary(Z, cols, ops):
    n = Z.shape[0]
    out = np.empty((n, len(cols)))
    for c in nb.prange(len(cols)):
        for i in range(n):
            out[i, c] = _unary(ops[c], Z[i, cols[c]])
    return out


@nb.njit(cache=True)
def _apply_binary(a, b, op):
    out = np.empty(len(a))
    for i in range(len(a)):
        out[i] = _binary(op, a[i], b[i])
    return out


_LIB = None


def _native():
    """ctypes handle to csrc/libsymreg_screen.so (built on first use with -O3 -march=native), or None."""
    global _LIB
    if _LIB is None:
        import ctypes
        import subprocess
        from pathlib import Path

        d = Path(__file__).resolve().parent / "csrc"
        so = d / "libsymreg_screen.so"
        src = d / "symreg_screen.cpp"
        try:
            if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
                subprocess.run(["g++", "-O3", "-march=native", "-ffast-math", "-fopenmp", "-shared", "-fPIC", str(src), "-o", str(so)],
                               check=True, capture_output=True)
            lib = ctypes.CDLL(str(so))
            f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
            f64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
            lib.screen_unary.argtypes = [f32, ctypes.c_int64, ctypes.c_int64, f32, f32, ctypes.c_int, f64]
            lib.screen_binary.argtypes = [f32, ctypes.c_int64, ctypes.c_int64, f32, f32, f64]
            _LIB = lib
        except (OSError, subprocess.CalledProcessError):
            _LIB = False
    return _LIB or None


def screen_unary(Z, r, w, n_ops):
    lib = _native()
    if lib is None:
        return _screen_unary(Z, r, w, n_ops)
    zc = np.ascontiguousarray(Z.T, dtype=np.float32)
    out = np.zeros((Z.shape[1], n_ops))
    lib.screen_unary(zc, Z.shape[0], Z.shape[1], np.ascontiguousarray(r, np.float32), np.ascontiguousarray(w, np.float32), n_ops, out)
    return out


def screen_binary(V, r, w, n_ops):
    lib = _native()
    if lib is None or n_ops != 7:
        return _screen_binary(V, r, w, n_ops)
    vc = np.ascontiguousarray(V.T, dtype=np.float32)
    m = V.shape[1]
    out = np.zeros((m, m, 7))
    lib.screen_binary(vc, V.shape[0], m, np.ascontiguousarray(r, np.float32), np.ascontiguousarray(w, np.float32), out)
    return out


def wpearson(y, p, w) -> float:
    sw = w.sum()
    my, mp = (w * y).sum() / sw, (w * p).sum() / sw
    cov = (w * (y - my) * (p - mp)).sum() / sw
    vy, vp = (w * (y - my) ** 2).sum() / sw, (w * (p - mp) ** 2).sum() / sw
    return float(cov / np.sqrt(vy * vp)) if vy > 0 and vp > 0 else 0.0


@dataclass
class Standardizer:
    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X, w):
        lo, hi = np.nanpercentile(X, 0.5, axis=0), np.nanpercentile(X, 99.5, axis=0)
        Xc = np.clip(X, lo, hi)
        sw = w.sum()
        mean = (w[:, None] * Xc).sum(0) / sw
        std = np.sqrt((w[:, None] * (Xc - mean) ** 2).sum(0) / sw)
        return cls(lo, hi, mean, np.where(std > 1e-9, std, 1.0))

    def __call__(self, X):
        return ((np.clip(X, self.lo, self.hi) - self.mean) / self.std).astype(np.float64)


@dataclass
class Term:
    expr: str
    spec: tuple  # ("u", col, op) or ("b", specA, specB, op)
    score: float


@dataclass
class SymRegResult:
    terms: list[Term] = field(default_factory=list)
    coef: np.ndarray | None = None
    wp_train: list[float] = field(default_factory=list)
    wp_val: list[float] = field(default_factory=list)
    std: Standardizer | None = None


def predict_terms(res: SymRegResult, X, n_terms: int | None = None) -> np.ndarray:
    """Term values (n, n_terms) for new rows, using the fit-time standardization."""
    Z = res.std(X)
    terms = res.terms[: n_terms or len(res.terms)]
    return np.column_stack([_eval_spec(t.spec, Z) for t in terms])


def _eval_spec(spec, Z):
    if spec[0] == "u":
        return _apply_unary(Z, np.array([spec[1]]), np.array([spec[2]]))[:, 0]
    return _apply_binary(_eval_spec(spec[1], Z), _eval_spec(spec[2], Z), spec[3])


def _spec_str(spec, names):
    if spec[0] == "u":
        op = UNARY[spec[2]]
        return names[spec[1]] if op == "id" else f"{op}({names[spec[1]]})"
    a, b = _spec_str(spec[1], names), _spec_str(spec[2], names)
    return {"add": f"({a} + {b})", "sub": f"({a} - {b})", "mul": f"({a} * {b})", "div": f"({a} / (1+|{b}|))",
            "max": f"max({a}, {b})", "min": f"min({a}, {b})", "gate": f"({a} if {b}>0)"}[BINARY[spec[3]]]


def _rescore(specs, Z, r, w):
    """Exact |weighted Pearson| of each candidate spec with r on all rows."""
    out = []
    for sp in specs:
        out.append(abs(wpearson(r, _eval_spec(sp, Z), w)))
    return out


def fit(X, names, y, w, X_val=None, y_val=None, w_val=None, n_terms=5, top_m=60, level2=True, pool_base=True,
        screen_rows=30_000, verbose=True, seed=0):
    """Greedy SISSO: returns SymRegResult with expressions and cumulative train/val WP.

    The level-2 pool holds the top_m screened level-1 candidates and, with ``pool_base``, every
    base feature untransformed, so pure interactions whose factors have no marginal correlation
    with the residual are still reachable.
    """
    std = Standardizer.fit(X, w)
    Z = std(X)
    Zv = std(X_val) if X_val is not None else None
    y = y.astype(np.float64)
    w = w.astype(np.float64)
    res = SymRegResult(std=std)
    design = [np.ones(len(y))]
    design_v = [np.ones(len(y_val))] if X_val is not None else None
    r = y.copy()
    for it in range(n_terms):
        s1 = np.abs(screen_unary(Z, r, w, len(UNARY)))
        flat = np.argsort(-s1.ravel())[:top_m]
        cands = [(("u", int(j // len(UNARY)), int(j % len(UNARY))), float(s1.ravel()[j])) for j in flat]
        if level2:
            pool = [c[0] for c in cands]
            if pool_base:
                seen = set(pool)
                pool += [("u", j, 0) for j in range(Z.shape[1]) if ("u", j, 0) not in seen]
            # screen pairs on a row subsample, then rescore the best on all rows
            sub = np.arange(len(r)) if len(r) <= screen_rows else np.sort(
                np.random.default_rng(seed + it).choice(len(r), screen_rows, replace=False))
            V = np.column_stack([_eval_spec(c, Z[sub]) for c in pool])
            s2 = np.abs(screen_binary(V, r[sub], w[sub], len(BINARY)))
            m = len(pool)
            flat2 = np.argsort(-s2.ravel())[:3 * top_m]
            specs = []
            for j in flat2:
                a, rem = divmod(int(j), m * len(BINARY))
                b, op = divmod(rem, len(BINARY))
                specs.append(("b", pool[a], pool[b], op))
            for sp, sc in zip(specs, _rescore(specs, Z, r, w)):
                cands.append((sp, sc))
        best = max(cands, key=lambda c: c[1])
        term = Term(_spec_str(best[0], names), best[0], best[1])
        res.terms.append(term)
        design.append(_eval_spec(best[0], Z))
        D = np.column_stack(design)
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(D * sw[:, None], y * sw, rcond=None)[0]
        fitted = D @ beta
        r = y - fitted
        res.coef = beta
        res.wp_train.append(wpearson(y, fitted, w))
        if X_val is not None:
            design_v.append(_eval_spec(best[0], Zv))
            res.wp_val.append(wpearson(y_val.astype(np.float64), np.column_stack(design_v) @ beta, w_val.astype(np.float64)))
        if verbose:
            v = f" val {res.wp_val[-1]:.4f}" if res.wp_val else ""
            print(f"term {it + 1}: {term.expr}  |corr| {best[1]:.4f}  cum train {res.wp_train[-1]:.4f}{v}", flush=True)
    return res
