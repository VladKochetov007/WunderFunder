"""Causal blend/stack policies for two (or three) members.  All are per-row functions of the row's own predictions,
causal features (heur_valid: since_*, rates, ...), per-sequence EWMA state started at row 0, and a class estimate fixed from the first 99 rows."""
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

torch.set_num_threads(2)
GRID = np.round(np.arange(0, 1.0001, 0.05), 3)
HLS = [50, 100, 200, 500]

def pearson_k(y, p, k):
    w = np.abs(y[:, k]).astype(np.float64); yy = y[:, k].astype(np.float64); pp = np.clip(p[:, k].astype(np.float64), -2, 2)
    W = w.sum(); my = (w * yy).sum() / W; mp = (w * pp).sum() / W
    c = (w * yy * pp).sum() / W - my * mp
    return c / np.sqrt(((w * yy * yy).sum() / W - my * my) * ((w * pp * pp).sum() / W - mp * mp))

def wp(y, p):
    return 0.5 * (pearson_k(y, p, 0) + pearson_k(y, p, 1))

class D:
    """Flat scored-row data for one pair."""
    def __init__(self, flat, pa, pb, ew, warm, idx_seq):
        self.n = len(pa)
        self.y, self.pa, self.pb = flat["y"], pa, pb
        self.seq = flat["seq"]; self.feats = flat["feats"]
        self.dE, self.pE = ew["dE"], ew["pE"]
        self.warm = warm; self.cls = flat["market_class"]; self.comp = flat["comp"]
    def sub(self, seqmask):
        r = seqmask[self.seq]
        o = object.__new__(D)
        for k in ("y", "pa", "pb", "seq", "feats", "dE", "pE"):
            setattr(o, k, getattr(self, k)[r])
        o.n = int(r.sum()); o.warm, o.cls, o.comp = self.warm, self.cls, self.comp
        o.rowmask = r
        return o

# ---------------- policy 1 / 2: static weights ----------------
def fit_static(d, per_target):
    ws = []
    for k in range(2):
        best = max(GRID, key=lambda w: pearson_k(d.y, w * d.pa + (1 - w) * d.pb, k))
        ws.append(best)
    if not per_target:
        w = max(GRID, key=lambda w: wp(d.y, w * d.pa + (1 - w) * d.pb)); ws = [w, w]
    return np.array(ws)
def apply_static(ws, d):
    return ws[None, :] * d.pa + (1 - ws[None, :]) * d.pb

# ---------------- policy 3: ridge stacker ----------------
def ridge_design(d, level):
    cols = [d.pa, d.pb] if level == 0 else [d.pa, d.pb, d.pa - d.pb, np.abs(d.pa - d.pb)]
    X = [c[:, k] for k in range(1) for c in cols]
    return cols
def design_k(d, k, level, mu=None, sd=None):
    base = [d.pa[:, k], d.pb[:, k]]
    if level >= 1:
        base += [d.pa[:, k] - d.pb[:, k], np.abs(d.pa[:, k] - d.pb[:, k])]
    if level >= 2:
        F = d.feats.copy(); F[:, 0] = np.log1p(F[:, 0]); F[:, 1] = np.log1p(F[:, 1])
        if mu is None: mu, sd = F.mean(0), F.std(0) + 1e-6
        Z = (F - mu) / sd
        pm = 0.5 * (d.pa[:, k] + d.pb[:, k]); dd = d.pa[:, k] - d.pb[:, k]
        base += list(Z.T) + [Z[:, j] * pm for j in range(Z.shape[1])] + [Z[:, j] * dd for j in range(Z.shape[1])]
        base += [d.dE[:, k, 1], d.pE[:, k, 1], d.dE[:, k, 1] * dd]
    return np.stack(base, 1).astype(np.float64), mu, sd

def wridge(X, y, w, alpha):
    mu = np.average(X, axis=0, weights=w); ym = np.average(y, weights=w)
    Xc = X - mu; sc = np.sqrt(np.average(Xc ** 2, axis=0, weights=w)) + 1e-9
    Xs = Xc / sc
    A = (Xs * w[:, None]).T @ Xs / w.sum() + alpha * np.eye(X.shape[1])
    b = (Xs * w[:, None]).T @ (y - ym) / w.sum()
    beta = np.linalg.solve(A, b) / sc
    return beta, ym - mu @ beta

def fit_ridge(d, level, alphas=(1e-4, 1e-3, 1e-2, 1e-1)):
    out = []
    for k in range(2):
        X, mu, sd = design_k(d, k, level)
        y = np.clip(d.y[:, k], -2, 2).astype(np.float64); w = np.abs(y)
        # inner 2-fold over sequences by component
        comps = np.unique(d.comp[d.seq]); rng = np.random.default_rng(1)
        csel = d.comp[d.seq]; half = np.isin(csel, rng.permutation(np.unique(csel))[: len(np.unique(csel)) // 2])
        sc = {}
        for a in alphas:
            s = 0
            for tr, te in ((half, ~half), (~half, half)):
                beta, b0 = wridge(X[tr], y[tr], w[tr], a)
                p = X[te] @ beta + b0
                pp = np.zeros((te.sum(), 2)); pp[:, k] = p
                s += pearson_k(d.y[te], pp, k)
            sc[a] = s / 2
        a = max(sc, key=sc.get)
        beta, b0 = wridge(X, y, w, a)
        out.append((beta, b0, mu, sd, a))
    return out
def apply_ridge(par, d, level):
    p = np.zeros((d.n, 2), np.float32)
    for k in range(2):
        beta, b0, mu, sd, _ = par[k]
        X, _, _ = design_k(d, k, level, mu, sd)
        p[:, k] = X @ beta + b0
    return p

# ---------------- policy 4: causal gate ----------------
def gate_feats(d, k, hl_i):
    F0 = np.log1p(d.feats[:, 0])
    return np.stack([d.dE[:, k, hl_i], np.abs(0.5 * (d.pa[:, k] + d.pb[:, k])), F0], 1).astype(np.float64)
def fit_gate(d, hl_i, l2=1e-3, iters=60):
    pars = []
    for k in range(2):
        Z = gate_feats(d, k, hl_i); mu, sd = Z.mean(0), Z.std(0) + 1e-9; Zs = torch.tensor((Z - mu) / sd)
        y = torch.tensor(d.y[:, k].astype(np.float64)); w = y.abs(); W = w.sum()
        pa = torch.tensor(np.clip(d.pa[:, k], -2, 2).astype(np.float64)); pb = torch.tensor(np.clip(d.pb[:, k], -2, 2).astype(np.float64))
        th = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        def loss():
            g = torch.sigmoid(th[0] + Zs @ th[1:]); p = g * pa + (1 - g) * pb
            my = (w * y).sum() / W; mp = (w * p).sum() / W
            c = (w * y * p).sum() / W - my * mp
            vy = (w * y * y).sum() / W - my * my; vp = (w * p * p).sum() / W - mp * mp
            return -c / torch.sqrt(vy * vp) + l2 * (th[1:] ** 2).sum()
        opt = torch.optim.LBFGS([th], lr=1, max_iter=iters, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = loss(); l.backward(); return l
        opt.step(closure)
        pars.append((th.detach().numpy().copy(), mu, sd))
    return pars
def apply_gate(par, d, hl_i):
    p = np.zeros((d.n, 2), np.float32)
    for k in range(2):
        th, mu, sd = par[k]
        Z = (gate_feats(d, k, hl_i) - mu) / sd
        g = 1 / (1 + np.exp(-(th[0] + Z @ th[1:])))
        p[:, k] = g * d.pa[:, k] + (1 - g) * d.pb[:, k]
    return p

# ---------------- classes from first 99 rows ----------------
def fit_classifier(d_all_warm, cls, train_seq):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    est = np.full(len(cls), -1)
    idx = np.nonzero(train_seq)[0]
    est[idx] = cross_val_predict(m, d_all_warm[idx], cls[idx], cv=5)   # noisy labels for the fit, as at deployment
    m.fit(d_all_warm[idx], cls[idx])
    rest = np.nonzero(~train_seq)[0]
    if len(rest): est[rest] = m.predict(d_all_warm[rest])
    return est, m

# ---------------- policy 5/6: lookup tables by coordinate ascent on the global metric ----------------
def cell_ids(d, est_cls, edges, use_age, use_vol):
    cid = est_cls[d.seq].astype(np.int64)
    if use_age:
        a = np.log1p(d.feats[:, 0]); cid = cid * 4 + np.searchsorted(edges["age"], a)
    if use_vol:
        v = d.pE[:, 0, 1] + d.pE[:, 1, 1]; cid = cid * 4 + np.searchsorted(edges["vol"], v)
    return cid
def fit_table(d, est_cls, use_age, use_vol, wgrid=np.round(np.arange(0, 1.0001, 0.1), 2), min_mass=0.002, sweeps=3):
    edges = {"age": np.quantile(np.log1p(d.feats[:, 0]), [.25, .5, .75]), "vol": np.quantile(d.pE[:, 0, 1] + d.pE[:, 1, 1], [.25, .5, .75])}
    cid = cell_ids(d, est_cls, edges, use_age, use_vol); nc = cid.max() + 1
    tables = np.full((2, nc), 0.5)
    for k in range(2):
        y = d.y[:, k].astype(np.float64); w = np.abs(y); pa = np.clip(d.pa[:, k], -2, 2).astype(np.float64); pb = np.clip(d.pb[:, k], -2, 2).astype(np.float64)
        def cs(v): return np.bincount(cid, weights=v, minlength=nc)
        Wc, Sy, Syy = cs(w), cs(w * y), cs(w * y * y)
        Sa, Sb, Saa, Sbb, Sab, Sya, Syb = cs(w * pa), cs(w * pb), cs(w * pa * pa), cs(w * pb * pb), cs(w * pa * pb), cs(w * y * pa), cs(w * y * pb)
        W = Wc.sum(); my = Sy.sum() / W; vy = Syy.sum() / W - my * my
        mass = Wc / W
        def score_tab(t):
            Sp = (t * Sa + (1 - t) * Sb).sum(); Spp = (t * t * Saa + (1 - t) ** 2 * Sbb + 2 * t * (1 - t) * Sab).sum()
            Syp = (t * Sya + (1 - t) * Syb).sum()
            mp = Sp / W; return (Syp / W - my * mp) / np.sqrt(vy * (Spp / W - mp * mp))
        t = tables[k]
        for _ in range(sweeps):
            for c in range(nc):
                if mass[c] < min_mass: continue
                best, bw = -9, t[c]
                for g in wgrid:
                    t[c] = g; s = score_tab(t)
                    if s > best: best, bw = s, g
                t[c] = bw
    return {"tables": tables, "edges": edges, "use_age": use_age, "use_vol": use_vol}
def apply_table(par, d, est_cls):
    cid = cell_ids(d, est_cls, par["edges"], par["use_age"], par["use_vol"])
    t = par["tables"][:, cid].T
    return t * d.pa + (1 - t) * d.pb
