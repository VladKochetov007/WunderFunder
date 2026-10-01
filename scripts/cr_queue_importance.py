"""Queue-race falsifier: which in-run history groups carry the class-3 gain beyond the GRU (permutation by group).

    WNN_CPU_PCT=25 WNN_MEM_MAX=12G scripts/limited .venv/bin/python scripts/cr_queue_importance.py TRAIN.npz \
        VALID.npz SURPRISE_ROWS.npz OUT.json

Refits the history t0/t1 regressors of cr_queue_probe.py on the same TRAIN half (CR_SUB=2 rows), then on VALID
class-3 scored rows permutes one feature group at a time (rows shuffled jointly within the group) and records the
partial correlation beyond p and the cross-fitted calibrated class-3 delta vs the a + b p remap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_queue_probe import HGB, blend, partial, wp  # noqa: E402

GROUPS = {
    "current_row": None,
    "timing": lambda n: n.startswith(("h_since", "h_prev_run", "h_nE_")),
    "vol_since_run_start": lambda n: n.startswith(("h_dVrun", "h_pathV", "h_nchgV", "h_netV")),
    "vol_at_last_E": lambda n: n.startswith("h_dVatE"),
    "px_at_last_E": lambda n: n.startswith(("h_dPatE", "h_dPbE", "h_dPaE")),
    "px_since_run_start": lambda n: n.startswith("h_dPrun"),
    "trades_in_run": lambda n: n.startswith(("h_ncc_run", "h_ntr_run", "h_sumdp", "h_sumdv")),
    "last_vol_change": lambda n: n.startswith("h_lastdV"),
    "i1_in_run": lambda n: n.startswith(("h_npx1", "h_ntr1")),
}


def main() -> None:
    tr_path, va_path, rows_path, out_path = map(Path, sys.argv[1:5])
    z = np.load(tr_path)
    sub = slice(None, None, 2)
    XH = np.concatenate([z["X"][sub], z["H"][sub]], 1)
    Y = z["y"][sub].astype(np.float64)
    comp = z["comp"][sub]
    names = [*range(112), *map(str, z["hist_names"])]
    del z
    rng = np.random.default_rng(0)
    uc = np.unique(comp)
    half_a = set(rng.permutation(uc)[: len(uc) // 2].tolist())
    fit = np.array([c in half_a for c in comp])
    models = []
    for j in (0, 1):
        s = fit & (Y[:, j] != 0)
        models.append(HistGradientBoostingRegressor(**HGB).fit(XH[s], Y[s, j], sample_weight=np.abs(Y[s, j])))
        print("fit", j, flush=True)
    del XH
    v = np.load(va_path)
    XHv = np.concatenate([v["X"], v["H"]], 1)
    Yv = v["y"].astype(np.float64)
    quiet = v["nc_until_E"] >= 32
    r = np.load(rows_path)
    s3 = r["cls"] == 3
    P = np.column_stack([r["ft0"], r["ft1"]]).astype(np.float64)[s3]
    fold = r["seq_fold"][r["seq"].astype(np.int64)][s3]
    cal0, _ = blend(Yv, P, {}, fold, [], "cal")

    def evaluate(Xm):
        Z = {"reg": np.column_stack([m.predict(Xm) for m in models])}
        Pn, _ = blend(Yv, P, Z, fold, ["reg"], "cal")
        return {"partial": float(np.mean([partial(Yv[:, j], P[:, j], Z["reg"][:, j]) for j in (0, 1)])),
                "partial_quiet32": float(np.mean([partial(Yv[quiet, j], P[quiet, j], Z["reg"][quiet, j])
                                                  for j in (0, 1)])),
                "d_c3": wp(Yv, Pn) - wp(Yv, cal0), "d_c3_quiet32": wp(Yv[quiet], Pn[quiet]) - wp(Yv[quiet], cal0[quiet])}

    out = {"full": evaluate(XHv)}
    print("full", out["full"], flush=True)
    for g, f in GROUPS.items():
        cols = list(range(112)) if f is None else [i for i, n in enumerate(names) if isinstance(n, str) and f(n)]
        Xp = XHv.copy()
        Xp[:, cols] = XHv[np.ix_(rng.permutation(len(Xp)), cols)]
        out[g] = {"n_cols": len(cols), **evaluate(Xp)}
        del Xp
        print(g, out[g], flush=True)
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
