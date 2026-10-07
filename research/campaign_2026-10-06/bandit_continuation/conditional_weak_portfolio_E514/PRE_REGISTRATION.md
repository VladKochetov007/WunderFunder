# E-514: conditional-signal screen for a weak-model portfolio

Status: preregistered before scoring the E-511 C1 seed pair. Prediction-only analysis; no new weights are fitted in E-514.

## Research question and change from E-513

Can a predeclared pool of weak, differently sourced predictions improve a strong anchor when members are chosen for low prediction correlation or positive target signal conditional on the anchor? E-513 averaged a fixed weak trio, but post-score review found that the anchor and two members had propensity fit/checkpoint-selection ancestry on the evaluated folds. E-513 therefore supplies no clean OOF score evidence and is not used to choose E-514 candidates or weights.

The new intervention is a **calibration-only member screen followed by an unchanged chain-held-out evaluation**. It separates the idea “choose the least correlated weak predictor” from the repair “require conditional target signal as well.” The official endpoint remains `global_wp_clip2_v1`; correlation statistics are only candidate-screening diagnostics.

## Frozen anchor and candidate pool

The anchor `A` is the raw arithmetic mean of the two independently trained E-511 C1v2 fold-0 predictions at their frozen epoch 1:

- `bd26_c1v2_s3_f0`
- `bd26_c1v2_s4_f0`

The weak candidate pool, in fixed tie-break order, is:

1. `ab12_h128_plain_fallback` — one-epoch, seed-3, train-only GRU without propensity weights.
2. `ab48_basehead_f0` — one-epoch fold-0 head-only fine-tune from the supplied baseline, without propensity weights.
3. `baseline*0.25` — the organizer baseline at the established 0.25 output scale.

Their known weak standalone score or low correlation is a reason to test the user’s idea, not a positive prior. The C1 pair has a separate frozen comparison in E-511; E-514 adds a new, predeclared weak-pool endpoint. All source arrays are legacy files without complete modern prediction sidecars, so E-514 must bind their byte hashes and confirm exact shape and `valid_targets.npz` row-group order before any E-514 score is calculated. C1 prediction sidecars must additionally bind the frozen checkpoints and row groups.

The ancestry gate must confirm: (a) both C1 models use the audited fold-0 split, fit-only `c20260929_prop_f0`, fixed epoch selection, and no evaluation during fit; (b) `ab12` is train-only and has no propensity or mask-derived input; (c) `ab48` excludes fold-0 evaluation groups from fitting, has no propensity input, and its single-epoch checkpoint is fixed rather than chosen among several scored epochs; and (d) the baseline source is the supplied organizer artifact and was not fit on validation groups. If any ancestry or row-order fact cannot be verified, stop before E-514 score calculation and preserve the failure.

The fold-0 chain fold and all its groups have prior exposure. A further split of these chains improves separation of this E-514 calibration procedure from its internal score calculation; it does **not** create fresh or confirmatory data.

## Calibration and evaluation

Use only fold-0 valid groups whose connected chain contains no training group. Assign complete chain components to three outer folds by a shuffled permutation of unique chain IDs using seed `474419703`, derived from the first eight hex digits of `sha256(sensors output)` = `1c4711f79523231ba740478d4922bd893afbe980441de5be7c3b27c109ec2b15`. Persist the chain-to-outer-fold mapping and group counts before calculating any prediction metric. Each outer fold is evaluated once; its other two folds are calibration data. Tie breaks follow the fixed candidate order above.

For each calibration partition, choose and evaluate these predeclared strategies:

- **Anchor:** `A` alone (`lambda=0`).
- **Weak mean:** `C = (P_ab12 + P_ab48 + 0.25 P_baseline) / 3`.
- **Lowest-correlation member:** choose the candidate with the smallest arithmetic mean of the two absolute official-weighted per-target prediction correlations with `A`, using official mask, warmup and clipped values.
- **Conditional-signal portfolio:** residualize each candidate and each clipped target against the intercept and `A` under the official clipped-label-magnitude weights. Retain candidates with positive mean partial correlation across the two targets; equal-average the retained served-scale predictions. If none passes, set `C=A` and `lambda=0` for this arm.
- **Best single weak control:** use calibration-set official WP to select from the same candidate pool and lambda grid.

For each non-anchor strategy with a cohort `C`, select `lambda` only on the calibration chains from `[0, 0.05, 0.10, 0.15, 0.20, 0.25]` for `(1-lambda)A + lambda C`. Evaluate that exact member set and lambda unchanged on the outer chains. Concatenate the three once-held-out predictions per strategy and calculate the official metric on the complete out-of-fold chain set. Also report member-selection/lambda frequencies and per-target scores. Do not search members or weights on the concatenated held-out result and then report the selected result as OOF.

Uncertainty is a paired bootstrap over the held-out connected chains, conditional on the fold-specific calibration decisions. It does not include model-training, candidate-pool, checkpoint-selection, or public-transfer uncertainty. Compare the decorrelation-only and conditional-signal arms with both the anchor and best-single control; a positive cohort-versus-anchor value alone is insufficient.

## Stop and continuation rules

- Stop before scoring if any source hash, row-group alignment, fit membership, propensity ancestry, or fixed-checkpoint-selection check fails.
- Continue to a distinct clean-group or Kaggle cross-fit only if the conditional-signal portfolio improves the anchor and best-single control on the concatenated once-held-out predictions, the paired-chain uncertainty is compatible with a positive change, and no one outer fold drives the result. The next rung needs separately frozen fold-1 C1 predictions with the matching clean fold-1 ancestry; do not reuse E-513's propensity sources.
- If the lowest-correlation-only arm wins without conditional-signal qualification, inspect scale, bias, target-channel and chain composition before designing a separately registered child; do not call low correlation alone the mechanism.
- If both weak-pool strategies fail, lower this exact portfolio-screen arm's priority. Do not widen the candidate/lambda search on fold 0. Preserve teacher-target, residual-representation and distillation descendants as separate roles for weak models.

## Artifacts and interpretation

Persist a frozen source/prediction manifest, chain partition, replayable notebook, `E514_RESULT.json`, concise report, and figures under this directory's `plots/`. This is an exposed-fold exploration screen, not independent confirmation, a public-score forecast, or a direct submission. A candidate still needs full serving, package, and appropriate external evaluation checks.

## Before-score execution amendment

Before any E-514 comparison score, the audit method was amended to replace legacy run-card scalar replay with exact checkpoint-bound prediction exports and row-group sidecars. The candidate pool and comparison protocol are unchanged. See [PRE_SCORE_AUDIT_AMENDMENT_20261007.md](PRE_SCORE_AUDIT_AMENDMENT_20261007.md) for the observed cache discrepancy, corrected exports, and contained resource-limit incident.
