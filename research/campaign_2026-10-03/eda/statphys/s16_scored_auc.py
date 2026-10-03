# Which observables (causal) predict is_scored? univariate AUC of features, |dF|, and simple event indicators
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
from scipy.stats import rankdata
pf = pq.ParquetFile(VALID); fn = feat_names(pf)
rng = np.random.default_rng(9)
Xs, Ds, Ss, Us = [], [], [], []
for rg in rng.choice(pf.metadata.num_row_groups, 30, replace=False):
    X, Y, u, ex = load(pf, rg, extra=('is_scored',))
    s = ex['is_scored'].astype(bool)
    D = np.abs(np.diff(X, axis=0, prepend=X[:1]))
    Xs.append(X[99:]); Ds.append(D[99:]); Ss.append(s[99:]); Us.append(np.full(len(s) - 99, u))
X = np.concatenate(Xs); D = np.concatenate(Ds); s = np.concatenate(Ss); U = np.concatenate(Us)
def auc(x, s):
    r = rankdata(x); n1 = s.sum(); n0 = len(s) - n1
    return (r[s].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
res = [(fn[j], auc(X[:, j], s)) for j in range(112)] + [('|d|' + fn[j], auc(D[:, j], s)) for j in range(112)]
res.sort(key=lambda r: -abs(r[1] - 0.5))
for r in res[:25]: print(r[0], round(r[1], 4))
for u in np.unique(U):
    m = U == u
    rr = [(fn[j], auc(X[m, j], s[m])) for j in range(104, 112)] + [('|d|' + fn[j], auc(D[m, j], s[m])) for j in range(104, 112)]
    rr.sort(key=lambda r: -abs(r[1] - 0.5))
    print('class', round(u, 4), [(a, round(b, 3)) for a, b in rr[:6]])
