"""Critic: SP2 sensitivity grid over prior, transfer slope and route weight (numbers for expectation_v1.md)."""
import json
import numpy as np

N = 400_000
rng = np.random.default_rng(7)
H_SLOPES = {"pair": (0.657, 0.239, 0.00064), "single": (0.506, 0.20, 0.00054)}
rows = []
for prior in [(0.0003, 0.0006), (0.0, 0.0005), (None, None)]:
    for tau in [(0.75, 0.35), (0.8, 0.12), (0.5, 0.3)]:
        for wH, scen in [(0.5, "pair"), (0.5, "single"), (0.0, "pair"), (1.0, "pair")]:
            o, s = 0.001326, 0.000714
            if prior[0] is None:
                mu, sd = o, s
            else:
                k = prior[1] ** 2 / (prior[1] ** 2 + s ** 2)
                mu, sd = prior[0] + k * (o - prior[0]), np.sqrt(k) * s
            b, bsd, esd = H_SLOPES[scen]
            h = rng.normal(b, bsd, N) * 0.000379 + rng.normal(0, esd, N)
            f = rng.normal(*tau, N) * rng.normal(mu, sd, N) + rng.normal(0, 0.0006, N)
            d = 0.6622 + rng.uniform(-5e-5, 5e-5, N) + np.where(rng.random(N) < wH, h, f)
            rows.append({"prior": prior, "tau": tau, "w_holdout": wH, "ema6_scenario": scen,
                         "mean": round(float(d.mean()), 5), "p10": round(float(np.quantile(d, .1)), 5),
                         "p90": round(float(np.quantile(d, .9)), 5), "P>0.6641": round(float((d > 0.6641).mean()), 3),
                         "P>=0.6660": round(float((d >= 0.6660).mean()), 4)})
print(json.dumps(rows))
