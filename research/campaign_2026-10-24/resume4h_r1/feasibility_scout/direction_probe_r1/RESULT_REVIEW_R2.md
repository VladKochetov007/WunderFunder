---
kind: review
status: bounded_pass_with_reproduction_limit
created_utc: 2026-10-24T13:47:34Z
updated_utc: 2026-10-24T13:47:34Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1, H-ASTRA24-FUND-01]
experiment_ids: [E-20261024-fixed-direction-probe-r2]
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_RESULT_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_COMPONENT_MOMENTS_R2.npz
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/EXECUTION_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_RECIPE_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/direction_probe_math.py
---

# Independent result review — direction probe R2

## Disposition

The fixed-prediction result is arithmetically consistent with the independent moment checks that completed. I found no score or projection blocker. The actor's 2,000-draw bootstrap intervals remain **not independently reproduced** in this review: an earlier reviewer calculation used the wrong packed dimension for the eight-variable alpha-zero table, and the corrected verbose command did not return a captured compact receipt before the batchstream hold. Treat the interval values as actor-produced conditional uncertainty, not independently checked output.

This review does not establish a new model, clean validation, a causal training explanation, or a deployable correction. The operation used zero fits and exposed development folds.

## Identity and execution

Hashes below were computed from current bytes:

- Recipe: `a275cdd402aa8e22e181922dd3de0e4f1dee06a5a58fe579709c7f557fdc9a29`.
- Source manifest: `7265890a44ebbcdd46ac8030a3050ad0fb2a90d2642020db4f514077554859a1`.
- Runner: `f303dc5ff359b97d2257ffd2c82ee1f0783604b3cdcbf325c9500348a1027447`.
- Math: `7253701f9cf3bed4668a729a3a8b5447f1a4da6edfe50ddabae1bfd044c3d24e`.
- Result JSON: `cf7dbcc3da251fd77fff4a498115be9d12709fd293667c12c98ca507bef52413`.
- Component moments: `84c01f89f4c1426aebe64c375294cdf4475416424e95db4a13344b3b183539a9`.

The admitted R2 scorer completed with return code 0 in 90.434 seconds. Its receipt records a 399,998,976-byte peak under the 400,000,000-byte limit, zero OOM events and zero swap, with 12,028 memory-max events. The earlier R1 invocation failed before source preflights because it indexed lowercase `inputs.f0` / `inputs.f1` while its recipe used uppercase keys. That is preserved execution evidence, not a negative scientific result; R2's label-free fixture and later score execution are separate successful records.

I read the scalar result and the small aggregated component-moment NPZ only. No row-level prediction arrays or target payloads were opened. The latter moments are already derived, label-weighted evidence and do not make the folds fresh.

## Independent checks

The R2 component schema has eight alpha-zero variables and three alpha-0.1 variables. Using the matching dimensions, the independent covariance reconstruction differed by at most `2.31e-14`; moment-derived per-target score checks differed from reported values by at most `4.4e-11`. Direct weighted Pearson derivative arithmetic on the moment covariance matched reported alpha-zero and alpha-0.1 point derivatives within `2.8e-17`. All checked point branches are regular. The independent calibration Gram ranks were 2 for calibration-only and 3 after adding common A where prescribed; projection derivative checks agreed within `2.8e-17`. These checks support the reported point geometry and rank interpretation, not the bootstrap calculation.

The saved result also records exact reproduction of the existing official parent and fixed-float32 alpha-0.1 grade fields for all reported endpoints, using `global_wp_clip2_v1` and 808,863 F0 / 812,167 F1 selected rows. Those exact reproduction comparisons are in the actor's result receipt; I did not independently re-read prediction or target arrays. The distinction between real-arithmetic derivatives and the frozen float32 alpha-0.1 endpoint is correctly retained.

The first reviewer moment helper call failed before aggregation because I used a nonexistent helper name. A second call failed in the reviewer harness before calculation. A later point-check script produced the checks above, but its bootstrap offset treated the alpha-zero table as three-dimensional; its alpha-zero bootstrap comparison is invalid and superseded. The corrected full calculation returned an oversized output that the tool truncated, so it supplies no captured interval verification. These reviewer failures are recorded here rather than presented as data failures.

## Scientific reading and limits

The result supports a narrow descriptive statement: on exposed F0, the fixed alpha-0.1 shrink endpoints improve the parent by about `+0.00013137` WP, while on exposed F1 the fixed A endpoint is about `−0.00000470` WP from its parent. F0's positive local derivative is not the measured endpoint delta; F1's alpha-0.1 local derivative is negative. The reported conditional component-bootstrap intervals are consistent with this uncertainty, but their numeric reconstruction is the review gap above.

Projection ranks and calibration/A tangent shares describe weighted geometry of the saved predictions and exposed labels. They do not identify which inputs or training process caused the relationship, and evaluation-label-weighted projection coefficients are not serving coefficients. Fold and seed change together, prior endpoint grading used the same rows, and bootstrap resampling conditions on the fixed models and component identities. No public-score forecast follows.

The [R1 execution receipt](EXECUTION_R1.json) preserves the earlier key error. The [R2 result](DIRECTION_PROBE_RESULT_R2.json), [execution receipt](EXECUTION_R2.json), [recipe](DIRECTION_PROBE_RECIPE_R2.json), and [component moments](DIRECTION_PROBE_COMPONENT_MOMENTS_R2.npz) support the checks and limitations above. No additional result run is authorized by this review.
