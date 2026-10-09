# E-517 result: fixed rowwise gate on the exposed E-514 pool

**Artifact/calculation integrity: pass. Preregistered scientific screen: fail.**
Two independent Luna-max review sessions reconstructed the result from the
registered source, fold, code and prediction artifacts. Neither reviewer
changed project files or launched a fit.

The exact E-514 manifest, chain partition, row-group ordering, ancestry/cache
checks and saved prediction arrays matched their registered hashes. The
population was 432 previously exposed row groups across 123 complete chains,
with 41 chains per fold. Both saved E-517 OOF arrays reconstruct exactly and
are byte-identical.

| Endpoint | Pooled official WP | Target 0 | Target 1 |
|---|---:|---:|---:|
| C1 anchor | 0.6832264635 | 0.6733560416 | 0.6930968854 |
| 75th-percentile disagreement gate | 0.6832264635 | 0.6733560416 | 0.6930968854 |
| Matched global ab12 blend | 0.6832264635 | 0.6733560416 | 0.6930968854 |

All gated and global coefficients were zero in all three folds. Their paired,
fold-stratified complete-chain 95% intervals versus the anchor and one another
are exactly `[0, 0]` over 2,000 replicates. The official fold WPs were
0.7534652370, 0.6454556540 and 0.6372393614. The candidate was routed on
15.74%, 31.21% and 29.27% of scored rows; those rows carried 13.52%, 25.75%
and 24.78% of the official metric weight. No fold selected positive model
mass, so none of the four preregistered pass conditions holds.

The result closes this exact ab12/75th-percentile gate with its registered
weighted-squared-error coefficient fit. It does not show that every weak model
or every use of model diversity has zero value. E-514 had already tested
several fixed/global prediction-level blends on a coarse coefficient grid with
the exact official score and selected zero contribution. This distinction and
the lower priority of output-only blending are preserved in the tree; any
further same-pool test requires a separately registered, materially changed
question. This exposed screen earns no numerical bandit update, public claim
or score forecast.

Independent review caveats: the per-chain CSV is present but its digest was
not embedded in the result JSON; this did not affect the verified decision.
The E-517 bootstrap seed records the hash of `sensors` stdout, but the raw
stdout was not preserved, so a reviewer could not regenerate that derivation.
The zero-delta OOF predictions imply every paired bootstrap delta is zero; the
first reviewer also replayed all registered replicates. No new OOM kill was
observed during the bounded audits.

Artifacts: [frozen result](E517_RESULT.json),
[preregistration](GATED_BLEND_PREREGISTRATION_E517.md),
[independent pre-score executor review](E517_INDEPENDENT_PRE_SCORE_REVIEW_20261007.md),
[first independent recomputation](E517_INDEPENDENT_EVIDENCE_REVIEW_20261007.md),
[second independent audit](E517_SECOND_INDEPENDENT_AUDIT_20261007.md),
[fold plot](plots/E517_FOLD_GATE_RESULTS.png).
