# E-515 result: disagreement as an error-location signal

## Outcome

The one-time exposed-development screen passed its preregistered gate for the `ab12` candidate on target 0. The disagreement/error association is consistent across all three chain folds, and the high-disagreement quartile carries more anchor squared-error surrogate on these chains. This is a promising signal for a future training-allocation test, not evidence that merging the weak model or changing student sampling improves the official score.

The other five candidate-target comparisons did not pass. `ab12` target 1 has positive rank association but its error-excess interval crosses zero and the Holm-adjusted matched-random test does not pass. The `ab48` and baseline patterns are target-dependent or negative.

## Registered evidence

- Population: 432 groups across 123 complete chains and three frozen outer chain folds.
- Source manifest SHA-256: `8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd`; partition SHA-256: `e9e0cb9391789f54e34c1d02836d02c336c5d3a13bde6f9473196bbbc71da1c2`.
- The source was previously exposed fold-0 validation data. These intervals and p-values are conditional on the fixed candidates, observed ranking, and exposed split; they are not fresh confirmation.
- The screen used 2,000 stratified chain-bootstrap replicates and 5,000 matched-random permutations per comparison, with Holm correction across six comparisons.
- The endpoint here is a clipped, label-magnitude-weighted squared-error surrogate. E-515 did not compute an official WP score, fit a neural model, or produce a public submission.

| Candidate | Target | Equal-fold Spearman (95% chain bootstrap) | Top-quartile error excess (95% chain bootstrap) | Holm-adjusted p | Gate |
|---|---:|---:|---:|---:|---|
| ab12 | 0 | 0.507 [0.355, 0.629] | +0.488 [+0.284, +0.694] | 0.0012 | PASS |
| ab12 | 1 | 0.296 [0.120, 0.451] | +0.072 [-0.101, +0.251] | 0.8854 | no |
| ab48 | 0 | -0.106 [-0.310, 0.093] | +0.020 [-0.176, +0.233] | 1.0000 | no |
| ab48 | 1 | -0.278 [-0.455, -0.085] | -0.192 [-0.327, -0.051] | 1.0000 | no |
| baseline | 0 | -0.351 [-0.523, -0.146] | -0.359 [-0.509, -0.209] | 1.0000 | no |
| baseline | 1 | 0.372 [0.204, 0.517] | +0.132 [-0.019, +0.296] | 0.3539 | no |

## Bandit decision

Keep H-204 as a live but bounded direction; do not give it a numerical prior update from this screen. The next training experiment cannot use these validation-chain disagreements as a sampling schedule. It first needs ancestry-clean group-OOF disagreement predictions over supplied training groups, with every sampled group excluded from candidate fitting, checkpoint selection, and teacher propensity. Then preregister one paired, equal-compute guided/uniform/sham student comparison using identical architecture, seed pairing, and official `global_wp_clip2_v1` evaluation. If that OOF cache is not available, allocate effort to producing it only after the higher-priority H-206 stack result and delayed tree reflection.

## Human-readable figures and raw artifacts

- [Disagreement vs. anchor error by chain](plots/disagreement_vs_anchor_error.png)
- [Top-quartile error-excess intervals](plots/disagreement_top_quartile_error_excess.png)
- [Machine-readable screen result](E515_RESULT.json)
- [Per-chain diagnostics](E515_CHAIN_DIAGNOSTICS.csv)
- [Preregistration](PRE_REGISTRATION.md)
