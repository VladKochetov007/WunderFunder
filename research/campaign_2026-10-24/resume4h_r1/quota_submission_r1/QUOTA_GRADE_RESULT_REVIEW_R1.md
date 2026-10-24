---
kind: review
status: complete
created_utc: 2026-10-24T12:53:29Z
updated_utc: 2026-10-24T12:53:29Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/QUOTA_GRADE_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/GRADE_ADMISSION_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/GRADE_EXECUTION_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/CROSS_REPLAY_GATE_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/GRADE_PACKAGES_RECIPE_DRAFT_R1.json
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/SELECTION_R1.json
---

# Independent scalar review — quota submission grade R1

## Scope

Reviewed the frozen scalar result, admission, execution counters, selection, and cross-replay receipt. The grade SHA is `a67c728bc08b0256dd7eb0785393461491a27fb1f61325801746397c21c6e56a`; its recipe pin is `ab4c66354e33e9d13b82af63f0c0df4f534e6aa2de12852788aace6310d251cc`. This was a metadata and reported-number audit only: I did not open prediction or target arrays, run the grader, or rerun native inference.

## Arithmetic and coverage

The endpoint and contrast arithmetic agrees to rounding. On F0, baseline/S3/S3+S4 WP are 0.6502113693 / 0.6537244401 / 0.6539535466. The reported S3−baseline, S3+S4−baseline, and S3+S4−S3 differences are +0.003513071, +0.003742177, and +0.000229106. On F2, the corresponding endpoint scores are 0.7117529216 / 0.7117236101 / 0.7119583710, with differences −0.000029312, +0.000205449, and +0.000234761.

The reported descriptive equal-weight F0/F2 means also recompute from those fold contrasts: +0.001741880 for S3−baseline, +0.001973813 for S3+S4−baseline, and +0.000231934 for S3+S4−S3. The first two fold-specific intervals are positive on F0 but span zero on F2; both pair-versus-S3 intervals span zero. These are conditional intervals for fixed predictions, not a pooled significance result.

Coverage is 32 source-qualified components and 32 complete sequences in each of F0 and F2, 64 input identities total. The cross-replay metadata gate reports matching ordered feature and `need_prediction` signatures across all three ZIPs. The reported metric is the unchanged `global_wp_clip2_v1`, with official mask, row index at least 99, clipping at ±2, and absolute clipped-target-magnitude weights. There are no new fits; competition upload is false and expected public gain is null.

## Integrity and execution

The grade result binds the admitted recipe, selection, replay plan, package receipts, source manifest and target cache. Admission authorizes the target-quality read for this exact grade. The target cache pre/post SHA is unchanged at `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`; all three prediction SHA values also match their pre-grade identities. The grade exited successfully in 3.545 seconds under a 400,000,000-byte, one-core, zero-swap cgroup; peak memory was 399,998,976 bytes, with zero OOM and zero swap. Peak was only 1,024 bytes below the configured memory ceiling, so this receipt supports this completed run and does not imply extra memory headroom.

## Interpretation

The comparison supplies promising but exposed, sample-based evidence that the S3 and S3+S4 packages improve on the baseline in the F0 stratum; F2 is essentially flat for S3 and slightly positive for S3+S4, with broad conditional intervals. S3+S4 is numerically above S3 in both strata, but neither paired interval excludes zero. The equal-weight two-stratum summary is descriptive only.

This is not clean cross-validation or a public-score forecast. Supplied validation is repeatedly exposed, H390 S3/S4 used train and validation, TL3 ancestry is unknown, and each fold is a 32-component sample rather than the full fold. The result does not justify claiming a stable cross-fold gain or selecting from these numbers alone.

No scalar arithmetic, coverage, or result-identity inconsistency was found. The folder README still contains a pre-grade statement that labels have not been read; it should be updated to point to this completed grade and its explicit exposure limits.
