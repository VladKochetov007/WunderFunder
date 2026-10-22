---
kind: review
status: conditional_small_fixed_mixture_signal
created_utc: 2026-10-22
updated_utc: 2026-10-22
hypothesis_ids: [H-PLATEAU-I1-HISTORY-20261020-R1]
experiment_ids: [E-PLATEAU-I1-20261021-R1]
evidence_paths:
  - research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/i1/grade_i1_panel.py
  - research/campaign_2026-10-21/next_cycle_r1/I1_PANEL_ADMISSION_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/i1/PREFIT_RECIPE_AUDIT_R1.json
  - research/unified/reviews/STRONG_PARENT_F0_F1_WARM_INIT_CATALOG_20261022_R1.md
---

# Independent review of the completed i1 history screen

The terminal grade supports a small, fixed 90/10 B/C mixture contribution on
both tested cells. It does not support the history model as a standalone
replacement, establish a history-specific mechanism, or qualify a submission.
The two-cell consistency is enough to retain the already conditional four-fit
strong-parent follow-up as a bounded research screen, provided its warm-start
parity, exposure, and serving gaps remain explicit. Root owns fit admission and
allocation.

This review reads the committed grade JSON, frozen recipes, small source
receipts, and source code only. I did not open prediction, target, or checkpoint
arrays, run a grader or tests, load a model, or perform new fits. I did not
independently regrade the fixed predictions.

## Evidence and fixed comparison

The six-fit screen compares A (all i1 inputs masked), B (history masked), and C
(history active) on F0/seed1001 and F2/seed1002. Each arm has two fits, for six
new fits total, with no teacher or reused-model fits. The saved grade reports
468 F0 groups / 141 connected components and 443 F2 groups / 144 components;
it records complete float32 prediction coverage and the fixed official
`global_wp_clip2_v1` score. Its provenance checks require the common initial
parameter and normalization identities within each cell, fixed epoch-2
checkpoints, complete three-epoch update counts, exact fit/evaluation
membership, and prediction sidecar/checkpoint correspondence.

The screen's original grader already fixed the two mixtures at `0.9*B + 0.1*A`
and `0.9*B + 0.1*C` before outcomes (`4078ae085`). The terminal grader does not
search coefficients. The admission receipt pins the training source revision
`b1b680d4a700d353f3e6db4f761f6670bd02fc7f`; source hashes for the trainer,
predictor, feature construction, model/input, preprocessing, evaluator, and
metric match that historical tree. The terminal grade's grader, scorer,
evaluation, and metric hashes also match their recorded source files. The
target cache is pinned by the grader to SHA-256
`dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`; this
review did not read or hash that payload.

| Cell | C − B | Fixed 90/10 B+C − B (95% component interval) | Fixed 90/10 B+A − B (95% component interval) |
|---|---:|---:|---:|
| F0 / seed 1001 | −0.003537 [−0.007283, −0.000066] | +0.00035953 [−0.00001187, +0.00070893] | −0.00002308 [−0.00030904, +0.00026620] |
| F2 / seed 1002 | −0.003903 [−0.005818, −0.002287] | +0.00027145 [+0.00008223, +0.00043398] | +0.00054446 [+0.00031366, +0.00075637] |

The unweighted mean of the two B/C mixture point deltas is `+0.00031549`.
This is a descriptive two-cell average, not a pooled confidence estimate. B/C
mixture deltas are positive for both targets in both cells: F0 `+0.00052218`
and `+0.00019688`; F2 `+0.00037163` and `+0.00017128`. They are also positive
in each of the four fixed position bins in both cells. Those target/bin values
are point estimates without intervals and are correlated with the overall
score, not separate replications.

The uncertainty is conditional on the fitted predictions: 2,000 resamples of
source-qualified connected components, seed 271828. F0's B/C mixture interval
just crosses zero; F2's interval is positive. There is one fit per arm in each
fold/seed cell, so these intervals do not quantify training variability. Fold
and seed also change together. Both folds were previously inspected
development data, not fresh confirmation. The grader reports no across-cell
interval, and the bin/target point deltas do not add one.

## Interpretation and follow-up value

C loses to B alone in both cells, while the frozen B/C mixture is slightly
positive. That pattern is compatible with a weak complementary correction, but
the gain is small and not unique to history: the B/A mixture is flat on F0 and
larger than B/C on F2. A fixed mixture can benefit from model diversity or
prediction geometry without proving that the appended history carries a
portable mechanism. The fact that all B/C target/bin point deltas are positive
is encouraging consistency, not additional independent evidence.

The cold A/B/C runs share an initial-parameter hash within each cell, but that
does not mean they share the same initial function. C exposes nonzero randomly
initialized projection weights on the appended history columns; B masks those
inputs after normalization. The treatment therefore changes both history
exposure and the effective initial function/optimization path. The cold screen
cannot isolate those effects.

The existing four-fit plan is a reasonable next bounded test of the warm-start
question: one control and one history treatment in each of F0 and F1, compared
within fold. Keep the fixed 90/10 endpoint and scoring contract frozen, verify
that the zero-extended child reproduces its parent before any fit, and treat
results as exposed development evidence. F0 and F1 parents have different
ancestry and historical precision, so their deltas should not be pooled as
independent replications. This screen warrants spending the existing
conditional allocation to investigate a strong-parent contribution; it does
not warrant promotion or an expected public-score claim.

The warm code in `cc9af1c68745289a71e30aa45e5242af87a3f5bf` is default-off and
narrows extension to a standard 211-input `none` parent and 367-input
`i1_history` child. It zero-extends the input projection, retains the inherited
211 means/scales, and has a `fit-appended` mode that fits only the extra 156
normalization slots from the first two fitting-loader batches, retaining every
seventh flattened row. The helper saves and restores the parent mean/std and
checks their tensor identities. It rejects resume-state training and unsupported
parent/input combinations. Source inspection supports these intended
invariants, but it is not an initial-output parity measurement; preserve that as
a pre-fit gate. The current warm provenance records the estimator and fit
membership, not an identity for the exact two normalization batches.

There is a sampler detail to freeze for this follow-up. With the known seeded
`EpochOrderSampler`, normalization starts at epoch −1 and `prepare_epoch`
resets it to epoch 0 before fitting. With `batch_order=random`, the loader uses
shuffle and the normalization pass consumes the first two batches before epoch
0, advancing its RNG state. The existing strong-parent recipes use random
shuffle. If they remain the basis, the follow-up must either isolate/restore the
sampler RNG around normalization or explicitly freeze this new order and record
the sampled normalization batches. Do not describe it as the historical random
order otherwise.

There is also a packaging boundary: the existing native callback/export path
does not accept the 367-column `i1_history` layout. The all-zero-history B form
has a possible 211-column projection, but conversion and native parity are
unimplemented; active C needs a history feature builder and state/reset path.
Thus the measured PyTorch blend is not a currently packageable endpoint. Any
four-fit result remains research evidence until the active-history native path
or an explicitly supported conversion is implemented and evaluated.

Finally, this review did not run tests. The root's reported broad compatibility
run found one preexisting `test_fixed_checkpoints` failure: the test patches
`train_rnn.market_class`, while `evaluate` resolves to
`wnn.training.evaluation.evaluate`, which uses its own imported `market_class`.
The warm commit changes neither that evaluator nor the failing test; the source
diff from the preceding grade commit confirms the mismatch is outside the warm
change. It is a test-fixture issue, not evidence against the completed grade.

## Sources

- [Terminal six-fit grade](../../campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json) and [fixed grader](../../campaign_2026-10-21/next_cycle_r1/i1/grade_i1_panel.py).
- [Frozen screen admission](../../campaign_2026-10-21/next_cycle_r1/I1_PANEL_ADMISSION_R1.json), [prefit audit](../../campaign_2026-10-21/next_cycle_r1/i1/PREFIT_RECIPE_AUDIT_R1.json), and [screen packet](../../campaign_2026-10-21/next_cycle_r1/i1/README.md).
- [Conditional strong-parent catalog](STRONG_PARENT_F0_F1_WARM_INIT_CATALOG_20261022_R1.md) and [native-serving compatibility review](I1_DYNAMICS_NATIVE_SERVING_COMPATIBILITY_REVIEW_20261022_R1.md).
- Warm source reviewed at commit `cc9af1c68745289a71e30aa45e5242af87a3f5bf`, especially `src/wnn/training/i1_warm_normalization.py`, `data_loaders.py`, `epoch.py`, and `tests/test_i1_warm_normalization.py` (source only).
