# cr_surprise — residual forensics (final, saved by controller from the agent's message)

Basis: clean OOF since-FT pair, all 1873 valid sequences, Global WP 0.6895. Class 3 = 47.7% of weight, 60.5% of 1−WP.
Numbers: results.json; figure event_time.png; falsifiers hazard/tier/rawprobe_falsifier.json.
Verdict: loss is concentrated, but 18 cross-fitted remaps and 5 falsifiers all score ≤ 0 — no causally observable fix.

| # | surprise | at stake | falsifier result |
|---|---|---|---|
| 1 | Information arrives late inside a run: corr(run target, p at run start) 0.105 in class 3 vs 0.81 at run end (class 0: 0.70 vs 0.87). Class-3 rows ≥32 before next E = 50% of class-3 weight, 72% of its loss, WP 0.02–0.33. | non-causal until_E remap +0.048 class-3 | causal hazard predictable (R² 0.61) but class×hazard-decile remap −0.0012: model already prices hazard |
| 2 | Per-sequence trend level in class 3 (seq corr of D level 0.53; sd of seq-mean D 0.43 y vs 0.15 p) | oracle +0.029 global | causal drift (20–8000 rows) |corr| ≤ 0.025; linear augmentation −0.004 class 3 |
| 3 | Class-3 tier: same p for (0,−1) and (+1,−3) — direction found, not whether price moves through; winner rows = 86% of class-3 loss | — | AUC 0.666 p only / 0.679 summaries / 0.660 raw 112 features; raw-book residual MLP −0.011 |
| 4 | 29% of class-3 Var(p) is within-run, zero covariance | run-mean oracle +0.025 | causal mean −0.019; F-space/y-space EMA lose |
| 5 | E-row over-shift (slope 1.37 vs 1.025) | ≤ +0.0001 | fold-inconsistent |
| 6 | Mask avoids quote-move bursts in class 3 (3% scored on E rows vs 21% quiet) | — | propensity cache already reproduces it |
| 7 | Class-3 per-sequence S level anti-correlated with truth (−0.14) | oracle +0.0044 class 3 | S-shift cross-fit ≤ 0 |

Also: seed disagreement carries no confidence signal; parent beats FT in class 3 at since_E 32–512 (already used by RSP2 routing).
Where to dig: history-based decoding of the class-3 queue race (which queue depletes first; does price move through) early
in quiet runs. Volume codes are mixed, not depth-ordered; every post-hoc/summary-feature route is exhausted.
