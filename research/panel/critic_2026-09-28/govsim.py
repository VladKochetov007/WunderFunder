"""Faithful re-implementation of solution_ensemble._govern under per-sequence timing noise."""
import numpy as np
B, E, L = 52 * 60, 39.4e6, 20_000
def run(mu_scale, sigma, rng, nseq=1970, O=2e-6, F=1.4e-6, m=(18e-6, 18e-6, 18e-6, 19.5e-6), total_rows=None):
    k = 4; t = 0.0; rows = 0; seqs = 0; win_t0 = win_rows = 0; win_step = 0.0; drops = []
    for s in range(nseq):
        if s > 0:  # boundary: _govern(now)
            seqs += 1
            if seqs == 1:
                win_t0, win_rows, win_step = t, rows, 0.0
            else:
                rw = rows - win_rows; rem = E - rows
                if k > 1 and rw >= 20_000 and rem > 0:
                    allowed = (B - t) / rem; per_row = (t - win_t0) / rw
                    if per_row > allowed:
                        pm = win_step / rw / k; oh = per_row - pm * k
                        kn = max(1, min(k - 1, int((allowed - oh) // pm) if pm > 0 else 1))
                        drops.append((s, k, kn)); k = kn; win_t0, win_rows, win_step = t, rows, 0.0
        noise = np.exp(sigma * rng.standard_normal())
        step = (F + sum(m[:k])) * mu_scale * noise
        cold = 1.3 if s == 0 else 1.0
        t += L * (O * mu_scale * noise + step) * cold; win_step += L * step; rows += L
    return drops, t / 60
rng = np.random.default_rng(0)
base = 2e-6 + 1.4e-6 + 18e-6 * 3 + 19.5e-6
print(f"nominal K4 per_row={base*1e6:.1f}us -> {base*E/60:.1f} min; governor threshold={B/E*1e6:.1f}us/row")
for scale in [0.95, 1.0, 1.03, 1.05, 1.10]:
    for sigma in [0.01, 0.03, 0.06]:
        res = [run(scale, sigma, rng) for _ in range(200)]
        p = np.mean([len(d) > 0 for d, _ in res]); kfin = np.mean([d[-1][2] if d else 4 for d, _ in res]); tmax = max(t for _, t in res)
        print(f"speed x{scale:.2f} sigma={sigma:.2f}: P(any drop)={p:.2f} mean final k={kfin:.2f} max total={tmax:.1f} min")
# private set larger than EXPECTED_ROWS
for mult in [1.0, 1.2, 1.4]:
    d, t = run(1.0, 0.03, rng, nseq=int(1970 * mult))
    print(f"test rows x{mult}: drops={d[:3]} total={t:.1f} min")
