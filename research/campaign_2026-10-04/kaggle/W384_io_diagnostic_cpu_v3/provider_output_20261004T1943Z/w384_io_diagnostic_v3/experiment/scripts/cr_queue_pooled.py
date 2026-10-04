"""Queue-race falsifier: deployable pooled test of the saved class-3 scores (no affine recalibration of p).

    .venv/bin/python scripts/cr_queue_pooled.py PROBE.scores.npz VALID.npz SURPRISE_ROWS.npz OUT.json

Class-3 rows only: p'_j = p_j + beta_j (z_j - mean_fit z_j), optionally rescaled so the |y|-weighted sd of p'_j in
class 3 equals that of p_j ("keep_sd"). beta_j is chosen on a grid to maximise the POOLED Global WP of target j on the
other three folds (all classes), then applied to the held fold. Reports pooled and class-3 deltas vs raw p with a
chain-component bootstrap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_queue_probe import CLIP, seq_moments, wp, wp1  # noqa: E402
from wnn.evaluation import paired_component_bootstrap  # noqa: E402
from wnn.splits import connected_components  # noqa: E402

GRID = np.round(np.arange(0.0, 0.62, 0.02), 2)


def wsd(x, w):
    m = (w * x).sum() / w.sum()
    return np.sqrt((w * (x - m) ** 2).sum() / w.sum())


def remap(p, z, w, beta, zbar, keep_sd):
    q = p + beta * (z - zbar)
    if keep_sd:
        m = (w * p).sum() / w.sum()
        q = m + (q - (w * q).sum() / w.sum()) * wsd(p, w) / wsd(q, w)
    return q


def main() -> None:
    sc_path, va_path, rows_path, out_path = map(Path, sys.argv[1:5])
    sc = np.load(sc_path)
    r = np.load(rows_path)
    seq = r["seq"].astype(np.int64)
    s3 = r["cls"] == 3
    fold = r["seq_fold"][seq]
    Y = np.column_stack([r["y0"], r["y1"]]).astype(np.float64)
    P = np.column_stack([r["ft0"], r["ft1"]]).astype(np.float64)
    quiet = np.zeros(len(Y), bool)
    quiet[s3] = np.load(va_path)["nc_until_E"] >= 32
    ch = np.load(ROOT / "cache/chains.npz")
    comp = connected_components(ch["next"])
    vcomp = {int(g): int(c) for s_, g, c in zip(ch["split"], ch["row_group"], comp, strict=True) if s_ == "valid"}
    useq, mref = seq_moments(seq, Y, P)
    comps = np.array([vcomp[int(g)] for g in useq])
    u3, mref3 = seq_moments(seq[s3], Y[s3], P[s3])
    comps3 = np.array([vcomp[int(g)] for g in u3])
    out = {"grid": GRID.tolist()}
    for feats in ("control", "hist"):
        for keep_sd in (False, True):
            Pn = P.copy()
            betas = []
            for j in (0, 1):
                z = np.zeros(len(Y))
                z[s3] = sc[f"{feats}_r{j}"]
                w = np.abs(np.clip(Y[:, j], -CLIP, CLIP))
                for k in range(4):
                    f, a = fold != k, fold == k
                    zbar = z[s3 & f].mean()
                    best, bbest = -np.inf, 0.0
                    for beta in GRID:
                        q = P[f, j].copy()
                        m3 = s3[f]
                        q[m3] = remap(P[f, j][m3], z[f][m3], w[f][m3], beta, zbar, keep_sd)
                        v = wp1(Y[f, j], q)
                        if v > best:
                            best, bbest = v, beta
                    m3 = s3 & a
                    Pn[m3, j] = remap(P[m3, j], z[m3], w[m3], bbest, zbar, keep_sd)
                    betas.append(float(bbest))
            name = f"{feats}_{'keepsd' if keep_sd else 'add'}"
            _, mn = seq_moments(seq, Y, Pn)
            _, mn3 = seq_moments(seq[s3], Y[s3], Pn[s3])
            d = {"betas_t0_t1_by_fold": betas, "d_all": wp(Y, Pn) - wp(Y, P),
                 "d_c3": wp(Y[s3], Pn[s3]) - wp(Y[s3], P[s3]),
                 "d_c3_quiet32": wp(Y[quiet], Pn[quiet]) - wp(Y[quiet], P[quiet]),
                 "d_all_by_fold": [wp(Y[fold == k], Pn[fold == k]) - wp(Y[fold == k], P[fold == k]) for k in range(4)],
                 "boot_all": paired_component_bootstrap(mn, mref, comps, n_boot=1000, centered=True),
                 "boot_c3": paired_component_bootstrap(mn3, mref3, comps3, n_boot=1000, centered=True)}
            out[name] = d
            print(f"{name}: betas {betas} d_all {d['d_all']:+.5f} (p<=0 {d['boot_all']['p_le_0']:.3f}, "
                  f"ci {np.round(d['boot_all']['ci95'], 5).tolist()}) d_c3 {d['d_c3']:+.5f} "
                  f"(p<=0 {d['boot_c3']['p_le_0']:.3f}) quiet {d['d_c3_quiet32']:+.5f} "
                  f"folds {np.round(d['d_all_by_fold'], 5).tolist()}", flush=True)
    Path(out_path).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
