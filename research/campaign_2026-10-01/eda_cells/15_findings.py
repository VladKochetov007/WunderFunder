_g = {c: round(v["median_persistence"], 3) for c, v in GARCH.items()}
mo.md(f"""
## Findings (clean OOF, 4 folds; SOTA = R4D analog, reference = incumbent-recipe pair)

**Where the metric lives.** Class 3 carries ~48% of the metric weight at WP ≈ {SUMMARY[3]['WP SOTA (R4D analog)']:.3f}; classes 0–2 sit at 0.74–0.81.
SOTA gains are largest in class 3 (+{SUMMARY[3]['delta']:.4f}), then classes 2/1/0. The public test is estimated to hold ~46% class-3 sequences, so class 3 dominates public score.

**Target dynamics (event time).** Classes 0–2: target direction D is a persistent AR-like process (ACF 0.65–0.84 at lag 1, PACF cuts after ~2 runs) — consecutive runs share most of their future path.
Class 3 is a different regime: S alternates sign every run (ACF ≈ −0.9 at lag 1) and D oscillates with period ≈ 4–6 runs — quote **flicker cycles** between two levels.

**Residuals.** In event time the SOTA residual stays autocorrelated for 10+ runs (lag-1 0.46–0.75); class 3 keeps periodic peaks at lags 4/8/12/16/20.
This is overlapping-horizon structure: a run's label depends on many later quote moves (maturity ≥ 30 events, E-317), so these correlations are **not causally exploitable** (feature-reconstructed label feedback ≤ 0).
In row time the residual ACF decays to ~0 by 1,000 rows.

**Volatility (ARCH/GARCH).** ARCH-LM rejects homoscedasticity in 100% of sampled sequences. Median GARCH(1,1) persistence α+β of the event-time residual: {_g}.
Class 3 is near-integrated (β ≈ 0.82): residual magnitude regimes last long. Earlier per-sequence scale/level oracles turned out to be overfit when cross-fitted (E-314), so a causal scale model must come from observable activity, not past residuals.

**Calibration.** |y|-weighted reliability is close to the diagonal for every class and target (slight shrink at the extremes in class 2), so recalibration has no headroom (consistent with E-288).

**Event position.** WP falls steeply with rows until the next quote move (non-causal): ~0.88 within 8 rows of a move, ≤0.3 beyond 64 rows. In class 3 most metric weight sits 16–256 rows before the next move, where WP is 0.1–0.5 — the long queue-race runs.
WP by rows since the last move (causal) is comparatively flat in class 3, so the model does not know *when* it is in a long run.

**Between vs within sequences.** >90% of the pooled covariance is within sequences. Class 3 has both weak between-sequence correlation (≈0.52) and weak within (≈0.43); y's between-sequence variance share (≈0.11–0.12) is twice the prediction's (≈0.06) — the model under-disperses sequence-level means in class 3.

### Where to dig (creative-research readout)
1. **Distillation strength** is the only lever with a large recent effect (α 0.7 vs 0.5: +0.0023 single model, fold 0) — denoising class-3 labels whose realized values are dominated by long overlapping futures. Confirm and push α / teacher quality.
2. **Class-3 flicker regime as a representation**: explicit cycle phase/period features (event-time position within the flicker cycle) may let the GRU separate flicker runs from genuine queue races; cheap to test as inputs.
3. **Under-dispersed class-3 sequence level**: a causal, activity-based class-3 level/scale state (not past residuals) — test only with cross-fitting; prior oracle results warn of overfit.
""")
