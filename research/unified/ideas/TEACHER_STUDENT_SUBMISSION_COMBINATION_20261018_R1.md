---
kind: idea
status: proposed
created_utc: 2026-10-18T08:30:52Z
updated_utc: 2026-10-18T08:30:52Z
hypothesis_ids:
  - H-SUBMISSION-TEACHER-STUDENT-INTEGRATION-20261018-R1
experiment_ids:
  - E-SUBMISSION-TEACHER-STUDENT-INTEGRATION-20261018-R1
evidence_paths:
  - ../../campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md
  - ../../campaign_2026-10-18/online_features_r1/tl3_diagnostics_r1/README.md
  - ../../campaign_2026-10-18/online_features_r1/tl3_fixed_blend_f2_r1/README.md
---

# Build the next candidate through teacher-to-student transfer

## Question and hypothesis

Can causal state features add value inside teacher training and then transfer into one compact, deterministic GRU student more effectively than combining the teacher predictions at inference?

**Hypothesis:** retaining the raw input path while adding one causal learned state representation to a source-qualified teacher may improve the student target. A student that absorbs complementary information during training could preserve a useful fraction of it without deploying another full model or relying on a validation-selected ensemble coefficient.

## Why this is the best supported combination direction so far

The completed model-space mixture evidence is consistently unfavorable for the specific TL3/R3 pair. On exposed F0, TL3 scored 0.701818 WP, compared with 0.698202 for fixed 75/25 and 0.693740 for fixed 50/50. On exposed F2, TL3 scored 0.723887, compared with 0.722678 and 0.720950. The F2 paired component intervals for both mixtures versus TL3 were wholly negative. An input-only warmup router also lost to its parent on all four folds, and the causal lagged-pseudo-label gate added only +0.00004795 over the fixed blend with a zero-crossing interval. These results argue against making another TL3/R3 mixture from the same exposed cache.

They do not show that all ensembles or teachers are useless. They show that this R3 member is not demonstrated to improve the served TL3 endpoint through fixed output averaging or the tested gates. The more defensible integration point is the training target/representation, where a teacher may impart conditional information to a student even when it is a weaker standalone predictor.

## Proposed candidate construction after explicit resumption

1. **Anchor the deliverable.** Keep the strongest publicly confirmed package as the serving reference. The user reports a public team best of 0.66832; the exact archive and ancestry must be recorded before using it as a clean-fold source. Do not infer a public score by adding a local delta.
2. **Use source-clean teacher predictions.** For research comparison, create per-cell teachers with fitting membership excluding each evaluated component. Treat the existing TL3 ZIP as a submission/reference endpoint until its teacher and initializer ancestry are verified; do not silently use possibly all-data predictions as held-out targets.
3. **Add one causal representation at a time.** Start from an existing published causal-state handoff, especially [PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17), which already specifies raw-retaining local-trend teacher arms and matched standard-GRU student-cache replacement. Treat the innovation-conditioned local draft and [predictive-state PR #18](https://github.com/VladKochetov007/WunderFunder/pull/18) as separate, not stacked, targets until each has profile, clean source registry and teacher evidence.
4. **Compare complete student endpoints.** Keep an ordinary-cache student control and replace exactly one predeclared teacher family for each treatment. Match student architecture, fitting membership, seed, target scale, cache coefficient, schedule, checkpoint rule and serving conversion. A component can pay off through student quality even if its teacher score is lower; grade the student itself.
5. **Use checkpoint averaging as a separate, low-serving-cost arm.** Existing [PR #13](https://github.com/VladKochetov007/WunderFunder/pull/13) covers averaging checkpoints from one trajectory. Compare it after freezing the training recipe; do not confuse it with averaging unrelated seeds or teachers.
6. **Package only the frozen winner.** Compare the fixed student against the best reference with source-qualified cross-fold and seed evidence, then run exact-package deterministic CPU and size/runtime checks. Preserve public feedback as transfer evidence, not coefficient-selection feedback.

## What not to combine from the present evidence

- Do not average R3 geometry output with TL3 at 25% or 50%; both tested weights lose on two exposed panels. A new coefficient fitted on these panels would be adaptive and would not create clean evidence.
- Do not append the 39 groupwise Kalman statistics directly to a ridge/feature vector: that exact proxy lost by −0.013093 WP on nonofficial train rows. The two recurrent follow-ups changed sign across seeds and do not show stable gain.
- Do not route experts from the first 99 rows with the tested warmup gate; it underperformed. Do not treat the previous-row-label proxy as mature test-time feedback.
- Do not stack all geometry, moments, dictionary, clock and latent-observer features into one unidentifiable mega-arm. Use matched controls and explicit interaction nodes so an endpoint change remains interpretable.

## Current status

This is a proposal formed from existing records only. It does not admit a fit, score, blend export, package, upload or PR. The user's stop at 2026-10-18 08:27 UTC remains authoritative. Both geometry ZIPs were already submitted according to the user; they are not candidates to repeat. Existing published PRs cover the large causal-state, predictive-state, and trajectory-averaging ideas. The local innovation-gain code branch is unpushed and has no B300 evidence.

## Cheapest discriminating study once reopened

Reuse PR #17's matched teacher/student route rather than creating a new campaign. First bind a clean per-cell teacher registry and pass the actual-device profile and causal/reset checks. Then compare the ordinary cache against exactly one causal teacher replacement in paired fold/seed cells. Promote only when the standard-GRU student endpoint improves with stable evidence, and separately report the teacher score, student score, runtime, archive size and ancestry. All current folds are exposed, so a fresh source-qualified or prospective confirmation is still required for transfer claims.
