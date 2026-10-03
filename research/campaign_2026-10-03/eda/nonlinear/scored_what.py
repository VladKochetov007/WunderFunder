# What distinguishes is_scored rows? Single-feature AUCs (level, |diff|), plus target-based stats (non-causal, descriptive)
import numpy as np, sys
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *
from sklearn.metrics import roc_auc_score
ids = seq_ids("valid", 40, seed=1)
Xs=[];S=[];T=[];U=[];G=[]
for rg in ids[:25]:
    d = load("valid", rg)
    x = np.stack([d[c] for c in FEATS], 1)[99:]
    dx = np.abs(np.diff(np.stack([d[c] for c in FEATS], 1), axis=0))[98:]
    s = d["is_scored"][99:]
    u = unit_of(d["t0"])
    t = np.stack([d["t0"], d["t1"]], 1)[99:] / u
    # target change indicators (non-causal descriptive)
    dt = np.vstack([np.zeros((1,2)), np.abs(np.diff(t, axis=0))])
    Xs.append(np.hstack([x, dx])); S.append(s); T.append(np.hstack([t, dt])); U.append(np.full(len(s), u)); G.append(np.full(len(s), rg))
X=np.concatenate(Xs); S=np.concatenate(S); T=np.concatenate(T); U=np.concatenate(U); G=np.concatenate(G)
print("P(scored)", S.mean())
names = FEATS + ["|d|"+f for f in FEATS]
auc = []
for j in range(X.shape[1]):
    if X[:, j].std() == 0: continue
    a = roc_auc_score(S, X[:, j]); auc.append((abs(a - .5), a, names[j]))
auc.sort(reverse=True)
for a in auc[:25]: print("%-12s AUC %.3f" % (a[2], a[1]))
for j, nm in enumerate(["k0", "k1", "|dk0|", "|dk1|"]):
    print(nm, "AUC %.3f" % roc_auc_score(S, T[:, j]), " AUC(|.|) %.3f" % roc_auc_score(S, np.abs(T[:, j])))
# within-sequence: scored rows: target nonzero? distance from mode
for rg in np.unique(G)[:8]:
    m = G == rg; k0 = T[m, 0]; s = S[m]
    print(rg, "frac |k0|>=2: scored %.3f  unscored %.3f" % ((np.abs(k0[s]) >= 2).mean(), (np.abs(k0[~s]) >= 2).mean()))
