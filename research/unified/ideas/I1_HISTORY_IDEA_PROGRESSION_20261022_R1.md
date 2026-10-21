---
kind: idea
status: inconclusive
created_utc: 2026-10-21T22:57:52Z
updated_utc: 2026-10-21T22:57:52Z
hypothesis_ids: [H-PLATEAU-I1-INCREMENTALITY-20261020-R1, H-PLATEAU-I1-HISTORY-20261020-R1, H-I1-HISTORY-FUNCTION-PRESERVING-20261022-R1]
experiment_ids: [E-PLATEAU-I1-20261021-R1]
evidence_paths:
  - research/campaign_2026-10-21/next_cycle_r1/i1/interim_f0_abc/RESULT_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/RESULT_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/I1_F2_AB_POINTS_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/i1/runtime/panel_r2.log
---

# From useful i1 inputs to a function-preserving history test

## Question or claim

Can explicit causal history contribute information that the GRU does not retain,
even when the history model scores worse alone? Preserve the sequence of
observations and changing explanations, rather than recording only a winner.
This snapshot was requested by the user after the intermediate-results discussion.

## Mechanism and method

The frozen screen uses cold H512×1, the same 367 input slots, 10% of fitting
components as complete sequences, three epochs and fixed checkpoint 2. Cells
are F0/seed1001 and F2/seed1002. A masks standard and derived i1 channels;
B uses standard inputs with history slots masked; C activates residuals from
causal EMA half-lives 32/512 and slow change energy. There are six new fits,
two per arm, no teacher fits and no reused fitted controls. Metric and fixed
90%B/10%C mixture are unchanged; coefficients were not searched.

## Evidence and interpretation

1. **The original information question:** perhaps i1 contributes little at this
   fidelity. Removing it instead costs 0.01590144 on F0 and 0.01110844 on F2.
   Standard i1 is useful in both cells; this does not show which history
   representation will help or establish a broad capacity conclusion.
2. **The first apparent contradiction:** F0 C−B is −0.00353711, yet the fixed
   90%B/10%C mixture gains +0.00035953. Its conditional paired-component 95%
   interval is [−0.00001187,+0.00070893]; target deltas are +0.00052218 and
   +0.00019688. A weaker model can supply a useful correction. All four overall
   position-bin point deltas improve, but early t1 bins lose. This is one
   exposed cell, not an established ensemble improvement.
3. **A distribution explanation was checked:** the same F0 mixture gains
   +0.00036506 before prediction clipping and +0.00035953 under official
   clipping. Clipping slightly reduces the gain. The correction has positive
   target covariance after accounting for B for both targets. This weakens
   clipping-only and pure scalar-shrinkage explanations; it does not prove
   that temporal information caused the correction. This decomposition reuses
   the same fits and supplies no independent replication or extra uncertainty
   interval.
4. **A source-level confound changes the next test:** identical stored initial
   weights do not mean identical initial functions. C activates 156 appended
   random input columns that B masks after normalization. History exposure
   and the resulting optimization path therefore change together. The cold
   result may reflect useful information, a different training trajectory, or
   both; no conditioning effect has been measured.
5. **Latest standalone observation:** F2 C has now scored 0.6802047933 versus
   B 0.6841082148, C−B −0.0039034215 (t0 −0.0029051558;
   t1 −0.0049016872). Both history models are weaker alone. At this snapshot,
   the complete-panel grader is still running: the F2 fixed mixture and
   paired intervals are pending. Do not transfer the F0 mixture conclusion
   to F2 or select a mechanism from F0 alone.

## Limits and validity

All evaluation folds are exposed. Component intervals condition on the fitted
models and do not quantify seed variation, fold variation or public transfer;
two fold/seed cells cannot separate these axes. The screen is a small cold
training comparison, not evidence of a production improvement. Strong parents
have different initializer/cache ancestry and need within-cell matched controls.

Separate endpoint checks also matter: replaying TL3 on the current native host
leaves the H512 replacement gap near −0.00736292. A fixed FP32 counterfactual
on eight exposed components gains only +0.00004035 over native CP5 and leaves
−0.00328280 versus TL3 on that sample. Runtime/precision differences therefore
do not explain those measured gaps; unknown TL3 ancestry and member utility
remain alternatives. Do not infer a public score or architecture verdict.

## Decision and next step

The new falsifier is **function-preserving history introduction**: extend a
strong 211-input parent to 367 slots with zero appended projection weights,
retain its original normalization bitwise, and give both arms identical,
explicit fitting-only statistics for the appended slots. Verify parent-output
parity before fitting. Then compare masked/unmasked history with identical
sampling, objective and two-epoch continuation. This distinguishes incremental
history learning from an immediate random change to the inherited predictor.

Conditional implementation is prepared on commit
`cc9af1c68745289a71e30aa45e5242af87a3f5bf`: default-off normalization option,
restricted inheritance guards and synthetic tests. Static checks pass; tests
and independent source review are still pending. No warm fit is admitted by
this note. Selection waits for the complete two-cell panel and fixed endpoint
contribution; any follow-up uses the existing four-fit F0/F1 allocation, zero
new teachers, sustained resource profiling and no history/auxiliary combination.

Serving is a separate gate: active history is not wired into the native raw-row
callback, and the current native layout rejects 367 inputs. A useful screen
must still earn strong-control improvement and actual complete endpoint
quality/runtime before production. Better auxiliary prediction alone likewise
does not earn a larger dynamics run.

## Source records

- [Accepted cycle](../../../specs/plans/2026-10-21-next-local-kaggle-cycle.md)
  and [frontier](../../campaign_2026-10-21/next_cycle_r1/FRONTIER.md).
- [F0 paired grade](../../campaign_2026-10-21/next_cycle_r1/i1/interim_f0_abc/RESULT_R1.json),
  [fixed-mixture geometry](../../campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/RESULT_R1.json),
  [F2 A/B identities](../../campaign_2026-10-21/next_cycle_r1/I1_F2_AB_POINTS_R1.json),
  [live frozen-panel log](../../campaign_2026-10-21/next_cycle_r1/i1/runtime/panel_r2.log).
- [Independent interpretation and initial-function review](../reviews/I1_HISTORY_FIXED_MIX_GEOMETRY_REVIEW_20261022_R1.md),
  [warm-parent catalog](../reviews/STRONG_PARENT_F0_F1_WARM_INIT_CATALOG_20261022_R1.md),
  [native serving audit](../reviews/I1_DYNAMICS_NATIVE_SERVING_COMPATIBILITY_REVIEW_20261022_R1.md).
- [Full-fold current-host grade](../../campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_GRADE_R1.json)
  and [eight-component precision diagnostic](../../campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_PRECISION_PROBE_R2.json).

Append subsequent panel grades and interpretation changes as dated evidence;
do not silently replace this intermediate reasoning or historical recipes.
