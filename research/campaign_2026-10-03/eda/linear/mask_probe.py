"""What drives is_scored? Binned P(scored) by a-features; runs; within vs between sequence decomposition of corr."""
import sys, numpy as np
sys.path.insert(0, "research/campaign_2026-10-03/eda/linear")
from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=200)
pf = pfile("valid")
rgs = np.random.default_rng(3).choice(pf.metadata.num_row_groups, 120, replace=False)
A = []; S = []; Q = []; ST = []; Y = []; P3 = []
for rg in rgs:
    X, Yy, ex = load(pf, rg, extra=["is_scored", "need_prediction", "step_in_seq"])
    m = ex["need_prediction"]
    A.append(X[m][:, 104:112]); S.append(ex["is_scored"][m]); ST.append(ex["step_in_seq"][m]); Y.append(Yy[m])
    P3.append(X[m][:, [3, 15, 17, 20]])
    nz = np.abs(Yy[Yy != 0]); Q.append(np.full(m.sum(), nz.min()))
A = np.concatenate(A); S = np.concatenate(S); Q = np.round(np.concatenate(Q), 3); ST = np.concatenate(ST); Y = np.concatenate(Y)
print("overall scored", S.mean())
for j in range(8):
    x = A[:, j]
    qs = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    b = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, len(qs) - 2)
    print(f"a{j}: P(scored|decile)", np.array([S[b == i].mean() for i in range(len(qs) - 1)]), " edges", qs[[0, 5, -1]])
# joint a2,a3 table
for qv in np.unique(Q):
    m = Q == qv
    print("q", qv, "scored", S[m].mean())
# scored vs |y| and y sign
print("mean y0 scored/unscored", Y[S, 0].mean(), Y[~S, 0].mean(), " mean |y0|", np.abs(Y[S, 0]).mean(), np.abs(Y[~S, 0]).mean())
print("mean y1 scored/unscored", Y[S, 1].mean(), Y[~S, 1].mean(), " mean |y1|", np.abs(Y[S, 1]).mean(), np.abs(Y[~S, 1]).mean())
# a3 threshold sharpness: P(scored) vs fine a3 bins
x = A[:, 3]; qs = np.quantile(x, np.linspace(0, 1, 41))
b = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, 39)
print("a3 40-bin P(scored):", np.array([S[b == i].mean() for i in range(40)]))
