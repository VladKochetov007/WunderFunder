# E-515 preregistration: disagreement as a weak-model error locator

Registered 2026-10-07 07:42 Europe/Kyiv, before running this diagnostic. This
is a **no-fit, exposed-development screen** for H-204. It tests whether model
disagreement marks chains where the anchor's errors are larger. It cannot show
that those errors are learnable, improve the official metric, or transfer to
the public leaderboard.

## Bandit placement and question

The arm is `arm:disagreement-guided-training`, under the optimization family.
The exact E-514 direct-blend pool and scalar rule remain closed. E-514's
independently reviewed source manifest identifies held-out predictions for the
same fixed candidate pool on the 432 pure fold-0 groups across 123 connected
chains. H-204 asks a different question: can that disagreement act as a
training-allocation signal even if the weak predictor is not served?

The allocation cost is a short CPU-only scan of existing, out-of-fit
predictions. The execution window closes for local experiments at 10:00
Europe/Kyiv. No GPU, TPU, model fitting, public submission, or new weights are
part of E-515. All groups and candidate families were previously exposed; the
result is development evidence only.

## Frozen inputs

- Source manifest: `conditional_weak_portfolio_E514/SOURCE_MANIFEST_E514_V3.json`,
  SHA-256 `8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd`.
- Fold-0 chain partition: `conditional_weak_portfolio_E514/E514_CHAIN_PARTITION.json`,
  SHA-256 `e9e0cb9391789f54e34c1d02836d02c336c5d3a13bde6f9473196bbbc71da1c2`.
- Target cache: `cache/valid_targets.npz`, SHA-256
  `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`.
- Anchor: arithmetic mean of the two frozen E-511 C1v2 epoch-1 predictions,
  seeds 3 and 4. The source manifest records both prediction hashes and
  ancestry; neither checkpoint is used as a new initializer.
- Candidate pool and scales, fixed before analysis: `ab12` at 1.0, `ab48` at
  1.0, and the supplied baseline at 0.25. Their prediction hashes are bound in
  the same V3 source manifest. Do not select or remove candidates by these
  screen results.
- Population: the existing 432 pure fold-0 groups, grouped into 123 complete
  chains, and the already frozen three outer chain folds (41 chains each).
  Every selected group is out of the anchor and candidate fit sets, but all
  folds are exposed and this is not confirmation.

The notebook must verify the source-manifest and partition digests, expected
prediction/target shapes, row-group order, fold purity, and all prediction
hashes before computing diagnostics. A failed identity or membership check
stops before any score-derived statistic is produced.

## Fixed diagnostic definitions

For each chain and each target separately, use rows in the official scored
mask after the 99-row warmup. Clip labels and predictions to ±2, and use
`abs(clipped_label)` as the row weight, matching the official mask and weights.
The official evaluator is `global_wp_clip2_v1`; E-515 does **not** compute or
claim an official score.

- Anchor error surrogate: within-chain weighted mean squared residual of the
  clipped anchor from the clipped label.
- Candidate disagreement: within-chain weighted mean squared difference
  between the clipped anchor and the clipped, served-scale candidate output.
- Statistical unit: one connected chain. Do not treat rows as independent.
- Primary association: Spearman correlation between chain-level disagreement
  and anchor-error surrogate, computed within each frozen outer fold and
  summarized by the equal-fold mean Fisher-z correlation.
- Companion contrast: within each outer fold, compare mean anchor-error
  surrogate for the top disagreement quartile of chains with the remaining
  chains; average the three fold contrasts equally. Set each fold's top count
  to `ceil(0.25 * chain_count)`. If disagreement values tie at the cutoff,
  sort by `(disagreement, canonical chain_id)` ascending and select the last
  `count` rows; this makes the tie break deterministic and invariant to input
  row order. The matched random control preserves the resulting fixed count.
- Uncertainty: 2,000 chain bootstrap replicates, resampling chains within each
  outer fold. The association interval resamples all chains in each frozen
  fold. The error-excess interval resamples separately within the fixed
  high-disagreement and remaining-chain strata so neither group vanishes in a
  replicate; it is conditional on the fitted models and observed ranking.
- Matched random control: 5,000 permutations of the high-disagreement labels
  within each outer fold, preserving that fold's top-quartile count. Use the
  one-sided randomization p-value for positive error excess, then Holm-adjust
  the six candidate-by-target comparisons.

### Screen gate

Continue H-204 to a separately preregistered, equal-compute guided/uniform/sham
student design only if at least one candidate-target pair satisfies all of:

1. positive fold-specific Spearman correlation in at least two of the three
   outer folds;
2. the stratified chain-bootstrap 95% lower bound for equal-fold association
   is above zero;
3. the stratified bootstrap 95% lower bound for top-quartile error excess is
   above zero; and
4. the Holm-adjusted matched-random one-sided p-value is below 0.05.

Passing means only “disagreement is a promising error-location signal on these
exposed groups.” It does not authorize a training run or assert learnability.
If no pair passes, close H-204 for this fixed pool and do not repeatedly vary
strata, target, metric proxy, or candidate subset on these same chains. A
future candidate source can reopen the family through a separately frozen
protocol.

## Reproducibility and resources

The predeclared seed is `2985631662`, the first eight hex digits of the
2026-10-07 `sensors` stdout SHA-256
`b1f51fae294caebcf5326b25262933c55b97f1993b744f12f4b613a93bd472fd`.
The analysis runs through the active Marimo workflow with 1 CPU core, at most
2,500,000,000 bytes for any new process, no swap, and no GPU. It writes a
machine-readable result, chain summary, human-readable report, and figures to
this lineage's `plots/` folder. No local fit is involved.

Pre-score implementation audit at 08:02 Europe/Kyiv: a synthetic test showed
that an unstratified bootstrap can omit one side of the fixed top-quartile
contrast and produce an undefined replicate. The frozen estimator therefore
resamples high-disagreement and remaining chains separately for that contrast;
the Spearman bootstrap still resamples all chains within each outer fold. No
source-data statistic had been read when this amendment was made.

Pre-score tie-rule amendment at 08:14 Europe/Kyiv: an independent code review
found that the top-quartile cutoff did not state how equal disagreement values
were selected. The fixed-count cutoff now uses ascending canonical chain ID as
its secondary key, and a synthetic order-invariance test covers tied values.
No source-data statistic had been read when this amendment was made.
