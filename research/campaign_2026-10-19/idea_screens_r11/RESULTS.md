# R11 result: fixed initial-gradient importance sampling

The importance-sampled arm finished below the uniform control on the prespecified endpoint. On the accepted, exposed R3 F0 panel, canonical pooled `global_wp_clip2_v1` over 32 components and 59,436 original scored rows was **0.48819247** for uniform and **0.45491058** for importance sampling, for `importance - uniform = -0.03328189`.

| Official score | Uniform | Importance | Importance − uniform |
|---|---:|---:|---:|
| Target 0 (`t0`) | 0.49824180 | 0.46162526 | -0.03661653 |
| Target 1 (`t1`) | 0.47814314 | 0.44819590 | -0.02994725 |
| Pooled WP | 0.48819247 | 0.45491058 | -0.03328189 |

Both target-specific differences were negative. The component summaries were mixed (importance scored higher on 8 components and lower on 24). In the 32 descriptive leave-one-component-out comparisons, the pooled difference remained negative for every omission, ranging from **-0.03821993** to **-0.02696733**. These are sensitivity checks on the same exposed panel, not independent replications or inferential intervals.

Both arms completed all 756/756 updates with zero nonfinite events, using the same initialization and fit-only normalization. Each used 381,696 with-replacement draws. The fixed initial norm-only uncentered second moment was 650.7560 for uniform and 187.0809 for the importance distribution (ratio **0.28748**); the realized importance effective sample size was 16,850 of 31,808 rows. This is a fixed-`theta0` uncentered norm diagnostic, not a variance estimate for the stochastic gradient or the training trajectory. The importance arm also had fewer clipped updates (150 vs. 446). Neither diagnostic establishes why its final score was lower: importance correction is unbiased only for the preclip gradient at fixed parameters, while clipping and AdamW alter the updates nonlinearly.

The analysis helper recorded **61.66 s wall / 30.62 s CPU**; unit-level timing includes a separate startup scope. The measured fixed-probability/correction/common-uniform setup was 0.077 s wall; the analytic norm pass was 0.142 s wall. Per-arm wall time includes inverse-CDF index generation, correction lookup, and optimizer setup: uniform 7.104 s (2.687 s setup, 4.417 s training) and importance 5.301 s (0.088 s setup, 5.213 s training). This matched-update experiment does not establish equal-FLOP or equal-wall-clock efficiency.

This closes the **specific fixed initial-gradient importance sampler** on this panel. No retuning, second model, or follow-up variant is warranted from this result. The conclusion is conditional on the fixed seed, architecture, 756-update budget, and previously exposed/adaptive R3 panel; it is not a fresh-holdout, generalization, or public-score result and does not reject all importance-sampling methods. R7 scores were not used as an R11 baseline.

V1 stopped at the CLI import-inventory gate before output initialization, array reads, or model work; V2 corrected the `python -m` module-identity check and ran the same scientific protocol. Captured RESULT identity: SHA256 `0c8d01161ac3aee5b2e49744ce846caed1e0cdd314126e1ad998d7a8b1c375b1`.
