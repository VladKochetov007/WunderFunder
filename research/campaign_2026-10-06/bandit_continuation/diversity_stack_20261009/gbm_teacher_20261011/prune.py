import sys, numpy as np
d = np.load(sys.argv[1]); X = d["X"].astype(np.float32); names = list(d["names"])
Z = (X - X.mean(0)) / (X.std(0) + 1e-6); sd = X.std(0)
C = np.abs(np.corrcoef(Z.T)); np.nan_to_num(C, copy=False)
keep = []
for j in range(len(names)):
    if sd[j] < 1e-8: continue
    if not keep or C[j, keep].max() < 0.9: keep.append(j)
np.save(sys.argv[2], np.array(keep)); open(sys.argv[3], "w").write("\n".join(names[j] for j in keep))
print("kept", len(keep), "of", len(names))
