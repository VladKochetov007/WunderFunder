"""Zero-fit: teacher residual correlations and Var(y - t)/Var(y) by |y| decile on a train sample."""
import numpy as np

C = "/home/vlad/development/wunder/research_tmp/chrono_root/cache/"
z = np.load(C + "train_targets.npz")
idx = np.sort(np.random.default_rng(0).choice(z["y"].shape[0], 400, replace=False))
y = z["y"][idx].astype(np.float64).clip(-2, 2)
m = np.isfinite(y).all(-1) & (np.abs(y).sum(-1) > 0)  # train mask array is empty in this cache
names = ["tt29rep_fbc1k_s5", "tt29rep_fin256_s3", "tt29rep_fin256_s4"]
t = [np.load(C + f"pred_{n}_train.npy", mmap_mode="r")[idx].astype(np.float64) for n in names]
print("shapes", y.shape, t[0].shape, "rows", int(m.sum()))
yy = y[m].ravel()
tt = [a[m].ravel() for a in t]
w = np.abs(yy)


def wc(a, b):
    a = a - np.average(a, weights=w)
    b = b - np.average(b, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


for n, a in zip(names, tt):
    print(n, "wp", round(wc(a, yy), 5), "resid wstd", round(float(np.sqrt(np.average((yy - a) ** 2, weights=w))), 4))
mean3 = sum(tt) / 3
print("mean3 wp", round(wc(mean3, yy), 5))
for i in range(3):
    for j in range(i + 1, 3):
        print("pred corr", i, j, round(wc(tt[i], tt[j]), 4), "resid corr", round(wc(yy - tt[i], yy - tt[j]), 4))
q = np.quantile(np.abs(yy), np.linspace(0, 1, 11))
for k in range(10):
    s = (np.abs(yy) >= q[k]) & (np.abs(yy) <= q[k + 1])
    print("decile", k, "|y|<=", round(q[k + 1], 3), "E(y-t)^2", round(float(((yy - tt[0])[s] ** 2).mean()), 4),
          "E y^2", round(float((yy[s] ** 2).mean()), 4), "E t^2", round(float((tt[0][s] ** 2).mean()), 4))
