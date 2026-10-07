# E-513 post-score provenance correction

Date: 2026-10-07. This correction supplements the frozen preregistration and original result; it does not rewrite either artifact.

## Disposition

E-513's metric calculation, row-order reconstruction, frozen source hashes, and four run-card point-score reproductions replayed successfully. Its preregistered out-of-fold ancestry gate did not. Scores in `E513_RESULT.json` and `REPORT.md` are therefore descriptive exposed-fold results, not clean held-out evidence. This is a provenance failure, not a falsification of weak-model averaging.

The omitted gate was upstream propensity ancestry. The anchor and TD-target members depend on `prop_v1`: 166/432 pure fold-0 groups and 155/399 pure fold-1 groups entered its fitting split, while 178/432 and 163/399 entered its devB checkpoint-selection split. The `ab53_p2` members depend on `prop_v2_f{k}`; their launcher selected checkpoints by devB `is_scored` AUC on the same fold later treated as evaluation. The mask exposure shaped propensity weights used in model fitting; it was not direct use of held-out return labels. Nevertheless it violates the declared fit/selection ancestry gate. See the line-cited independent review in [INDEPENDENT_REVIEW_MEMO_20261007.md](INDEPENDENT_REVIEW_MEMO_20261007.md).

The notebook also deviated from the preregistered bootstrap seed on the 0→1 direction by adding the evaluation fold to the manifest-derived seed. The exact preregistered seed is 4,001,575,692. Replaying with that seed leaves every interval crossing zero; the cohort delta versus the single-member control changes sign across directions. This seed correction does not repair the ancestry defect or change the allocation decision.

## What remains useful

- Preserve the fixed point estimates, plots, source hashes, code receipt, and independent replay as historical diagnostics only.
- Do not use E-513 as OOF evidence, a public forecast, a submission candidate, or a numerical bandit reward.
- Close this exact three-member cohort and blend on these exposed folds. Do not widen its lambda grid or search more weights on the same data.
- Keep the broader mechanism open. Future weak-member screens must recursively audit each member's data fit, mask/propensity source, checkpoint selection, model selection, and row alignment before any score calculation.
- Treat E-177's partner-scan figures as historical, exposure-aware hypotheses until the relevant member ancestry is audited; do not silently transfer its numeric results into current bandit priors.
- A new direction should test conditional contribution, not raw decorrelation alone: on calibration groups, select from a predeclared pool using predictor covariance together with the candidate's conditional target association after the anchor; average the selected members with a fixed rule and evaluate unchanged on disjoint chain groups. Recompute the official pooled clipped weighted-Pearson metric on each evaluation group set. Previously inspected groups remain exposed, so this can guide exploration but cannot be called fresh confirmation.

## Separate E-511 status

The same independent review found `c20260929_prop_f0`, C1 seed-3, its initializer, and the two teachers used by C1 to have fold-0 fit-only membership and fixed checkpoint selection with evaluation withheld. That ancestry is clean for the frozen fold membership but the fold itself is previously exposed. Seed-4 remains a separate running attempt until its terminal receipt and provenance are verified; the pair is not yet a graded result.
