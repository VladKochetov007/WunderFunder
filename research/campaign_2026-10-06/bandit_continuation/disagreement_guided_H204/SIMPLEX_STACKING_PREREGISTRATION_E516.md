# E-516 preregistration: target-wise calibrated weak-model stacking

Registered 2026-10-07 07:49 Europe/Kyiv, before calculating E-516 coefficients
or endpoint scores. This is a **prediction-only, exposed-development test**;
no neural model is trained and no public-score claim is permitted.

## Research question and bandit placement

Can weak models with complementary prediction errors improve the C1 anchor if
their outputs receive separately calibrated, target-specific convex weights?
E-514 tested the same fixed candidate outputs through a globally scalar blend
and calibration-selected member sets. Every chosen scalar weight was zero.
E-516 changes the combination rule: jointly fit a simplex of weights for the
anchor and three weak predictors, separately for each target, and fit one
weighted intercept. It does not choose candidates by raw prediction
correlation. E-516 belongs to the new H-206 target-wise weak-model stacking
arm; it is not an implementation of H-203's residual-corrector hypothesis.

This experiment deliberately revisits the same candidate pool and exposed
fold-0 chains after E-514, in response to the user's new direction to test
whether low-scoring, less-correlated models can help under a different merge
rule. That instruction is a narrow override of E-514's prior “do not widen on
fold 0” stop for **one** prediction-only exploratory screen. It does not make
the analysis fresh, independent, confirmatory, or eligible for a numerical
bandit reward. The shared pool and outcomes were already inspected; selecting
this follow-up after E-514 is another exposure source. Do not treat a pass as
proof of complementarity or public transfer.

H-206 is a separate method hypothesis under the ensemble family, linked to
H-202 for provenance. It uses only predictions already present in the reviewed
E-514 source manifest. E-516 earns no numeric bandit prior update and does not
authorize model fitting or a submission. If it fails, close this exact
target-wise simplex recipe on this pool. If it passes, the next evidence must
come from materially new ancestry-audited candidates and genuinely unexposed
confirmation; do not retune on these chains.

## Frozen inputs and population

- Source manifest: `conditional_weak_portfolio_E514/SOURCE_MANIFEST_E514_V3.json`,
  SHA-256 `8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd`.
- Chain partition: `conditional_weak_portfolio_E514/E514_CHAIN_PARTITION.json`,
  SHA-256 `e9e0cb9391789f54e34c1d02836d02c336c5d3a13bde6f9473196bbbc71da1c2`.
- Targets: `cache/valid_targets.npz`, SHA-256
  `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`.
- Anchor: arithmetic mean of the frozen E-511 C1v2 seed-3/seed-4 epoch-1
  fold-0 predictions.
- Weak members, fixed: `ab12` at 1.0, `ab48` at 1.0, and organizer baseline
  at 0.25. The anchor is also a simplex member at scale 1.0.
- Population: all 432 pure fold-0 groups across 123 complete chains and the
  same three frozen outer chain folds used by E-514. Fit blend coefficients on
  the other two folds and score each chain once in its held-out fold.

Before any score statistic, code must verify the source/partition digests,
target and chain-cache hashes, every prediction and sidecar hash, full
prediction row order, source tensor shapes/dtypes, the 432-group/123-chain
population, chain purity, and 41 chains per outer fold. A failed identity gate
stops the run.

The machine-readable result must carry `experiment_id=E-516`,
`hypothesis_id=H-206`, the bandit arm ID, and the verified source-manifest and
chain-partition SHA-256 digests so its statistics remain bound to this
registration.

## Fixed calibration and endpoint

For each outer fold and target, fit weights `w` on the four fixed source
predictions by minimizing the official clipped-label-magnitude-weighted
squared error on calibration chains, with an intercept and constraints
`w_i >= 0`, `sum(w_i) = 1`. The regression inputs and labels are clipped to
±2, use the official scored mask after the 99-row warmup, and use
`abs(clipped_label)` as weight. Calibration minimizes weighted squared error
on the affine blend before clipping; the final held-out endpoint is clipped to
±2. A fixed ridge penalty of `1e-8 * max(trace(covariance) / 4, 1e-12)` times
`0.5 * ||w - e_anchor||²` selects the anchor-closest solution when the
quadratic problem is numerically tied. No ridge, threshold, pool, or candidate
choice is selected from held-out scores.

Apply the calibration intercept and weights to the four source outputs for
the held-out chains; clip the complete prediction to ±2. The primary endpoint
is pooled `global_wp_clip2_v1` over all once-held-out predictions. Also report
the two target scores, outer-fold scores, each fold's fixed weights/intercept,
and weak-member mass. Compare the stack with the unchanged anchor. E-514's
equal-weak and scalar-screen results remain separate historical controls; they
are not refit or retuned.

Uncertainty is a 2,000-replicate paired bootstrap over complete chains,
stratified by frozen outer fold and conditional on each fold's fitted blend.
The bootstrap reports the difference in official pooled WP between the OOF
stack and OOF anchor. It does not include model-training, blend-selection,
candidate-pool, or public-transfer uncertainty.

## Decision rule and stopping

The exact E-516 recipe is a promising development result only if (a) total weak
weight is positive in at least two outer folds, (b) pooled OOF WP exceeds the
anchor, and (c) the paired chain-bootstrap 95% interval has a positive lower
bound. Failing any condition closes this fitted-simplex recipe for this fixed
pool and exposed chains; do not widen the feature set, simplex constraints or
regularization on these groups. A pass can motivate one materially new
ancestry-audited candidate cohort or a separately preregistered held-out
confirmation when genuinely unexposed data exist. It is not a public gain or
submission-ready claim.

## Reproducibility and resources

Use one CPU core, at most 2,500,000,000 bytes for any new process, no swap, and
no GPU. The analysis uses memory-mapped prediction sources, streams sufficient
statistics by group, and writes result JSON, chain summaries and plots to this
lineage's `plots/` directory. The random seed for the paired bootstrap is
`2985631662`, derived from the recorded `sensors` output hash in E-515.


Pre-score execution repair at 08:28 Europe/Kyiv: the first notebook call failed on project-package import before any E-516 fit or statistic. The project `src/` path was added inside the live E-516 Marimo cell using code mode, then only that cell was rerun. This plumbing repair did not alter the frozen experiment. Details and the failed attempt are recorded in `E516_IMPORT_PATH_REPAIR_20261007.md` and `E516_IMPORT_PATH_FAILURE_20261007.json`.
