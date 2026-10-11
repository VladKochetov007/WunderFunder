# Cache-only distillation headroom diagnostics (fold 1, 2026-10-11 06:05)
Script `scripts/distill_ceiling_diag.py` (run on Vast B host from banked predictions; single = muonall_h384_e3 s3 0.67021; ensemble = 3 muonall_h384 seeds 0.67254; parent = ctrl_s3 h256 0.66559). Motivated by Grok-4.7 method 1/2/5 falsifiers (ideation/grok_distill.md).

| Probe | WP | Meaning |
|---|---|---|
| single | 0.67021 | |
| ensemble (3 seeds) | 0.67254 | distillation ceiling for this teacher set (+0.00233) |
| single with ensemble's run-level swapped in (true-label-run oracle) | 0.67175 | **66% of ensemble gain (+0.00154) is the per-run level** |
| ensemble delta on SUM channel (p0+p1) only | 0.67039 | +0.0002 |
| ensemble delta on DIFFERENCE channel (p0-p1) only | **0.67212** | **+0.0019 = 82% of ensemble gain lives in the difference channel** |
| run-averaged single / ensemble (non-causal oracle) | 0.70500 / 0.70677 | +0.0018, irrelevant to serving |
| weighted corr(ens-parent, y-parent) | 0.1316 | the diverse-teacher delta is predictable from labels (not pure cancellation) |

## Consequences
1. Ensemble gain is mostly (a) difference-mode (b) run-level. This matches residual EDA #5 (difference mode under-dispersed, 35% of label variance) and #2. A distillation loss that up-weights the difference channel (and run-level) targets the right 80%; plain MSE on (t0,t1) splits equally between the sum channel (nothing to gain) and difference.
2. Cached train-row targets: Allen-Zhu & Li (2012.09816) / lit. diagnosis: members memorise train rows so ensemble disagreement on train is small; the signal is on unseen paths -> OOD/augmented arms (w14_*) are the principled test; measure disagreement train vs valid once train caches exist (w15 jobs create them).
3. Student pipeline facts: warm start, 2 epochs, batch 16, target = 0.8125 * mean teacher; dsq_none ≈ dsq_t256 - 0.0008; most gain is extra low-LR training of parent.

## Hypotheses (appended to research/hypotheses.jsonl)
H-366 difference-channel-weighted distillation (weight diff channel xk in teacher term); H-367 run-level distillation target (event-run mean of ensemble); H-368 ensemble-minus-parent residual target weighted by disagreement; H-369 Pearson (correlation) distillation loss; H-370 long low-LR tail (6-12 epochs <=1e-4) on cached targets (w15 arms); H-371 privileged-lookahead teacher R^2 check; H-372 teacher-ensemble of SAME best recipe (3 muonall seeds) beats diverse weaker members as teacher set; H-373 hidden-state low-rank match.
