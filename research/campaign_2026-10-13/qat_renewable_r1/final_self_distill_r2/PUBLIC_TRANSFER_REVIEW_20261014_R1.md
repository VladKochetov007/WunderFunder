---
kind: result
status: complete
created_utc: 2026-10-13T22:55:30.861028+00:00
updated_utc: 2026-10-13T22:55:30.861028+00:00
hypothesis_ids: [H-QAT-LSQ-20261013]
experiment_ids: [E-QAT-FINAL-SELF-DISTILL-20261013-R2]
evidence_paths: [PUBLIC_RESULT_20261014_R1.json, RELEASE_R2.json]
---

# QAT public transfer: compression preserved most quality, no leaderboard gain

## Question or claim

The user reports the frozen QAT ZIP scored **0.6680**, portal `00fpe72k`, in **54m26s / 655MB**. It trails the original fixed blend by **0.0001** at displayed precision and takes **4m10s longer**. This completes the requested public probe as user-reported evidence; it does not establish a stronger submission.

## Mechanism and method

The existing equal-weight four-member blend replaces its fourth member with the stationary-original-teacher seed3 learned-W4 student. Three team members and all 0.25 coefficients are unchanged. Recurrent codes/scales export exactly, first head stays Q8, state stays FP32, and Q4 expands into byte kernels once at load. This is weight-only QAT; native A8 rounding remains outside training. Local release SHA `e2f78e09b0147b37c74b5809b9aa38f48d1ee6b83d656483b14eeb609aac22b3` remains unchanged; portal payload SHA is not available.

## Evidence and interpretation

| Fixed endpoint | Portal ID | Displayed public score | Duration | Memory |
|---|---|---:|---|---:|
| QAT learned-W4 | 00fpe72k | 0.6680 | 54m26s | 655MB |
| Seed6 cross-team Q8 | 0mbzevhv | 0.6679 | 46m51s | 650MB |
| E | 0j1f5ipt | 0.6675 | 51m22s | 651MB |
| Original xteam_h4 | 1hl88h25 | 0.6681 | 50m16s | 650MB |

QAT is +0.0001 versus seed 6 and +0.0005 versus E, but −0.0001 versus its original-blend comparator and −0.0002 versus previously reported pair_ftt2 best 0.6682. These are distinct endpoints, not independent QAT training replications. There is no new team best.

The package is 13,461,304 bytes versus original 16,308,160 bytes (**17.46% smaller**). Whole-package savings include native-library changes. Public memory is 5 MB higher, consistent with no demonstrated resident-memory saving. Runtime is 250 seconds (**8.29%**) longer, leaving only 334 seconds (**5m34s**) to the 60-minute ceiling. Archive compression did not provide observed CPU headroom on this submission.

The short local CPU0/Python3.11 profile gave 87.564 versus 89.678 µs percallback (ratio 0.9764); public duration ratio is 1.0829, the opposite direction. Thus the local short timing cannot justify a competition-host speed claim. Different library bytes, weights, hosts/load and measurement envelopes prevent attributing the extra time solely to QAT. Future serving comparisons need longer representative native replay and explicit runtime controls.

The fitted-data diagnostic had QAT−original−0.000050093, QAT−matchedW4PTQ+0.000083316, and QAT−floatingQ8−0.000158385. Public and local point comparisons both show no improvement over original. Their magnitudes are not interchangeable; no local delta is added to a public score. No public matched W4PTQ or floating-continuation control was submitted.

## Limits and validity

One user-transcribed scored submission; no independently queried portal or uploaded-payload hash. Scores are rounded to four decimals, with no statistical interval or significance claim. Public comparison does not prove equivalence or refute the broader QAT family. Original final parents fit all supplied validation, so local final grades remain exposed fitted diagnostics. Preserve the reported timestamp `2026-10-13 00:56` literally: it conflicts with the filename date/known release chronology, timezone is unspecified, and its meaning is unresolved. Reported quota3/5 and91total are observations from this transcript, not a fresh provider quota query.

## Decision and next step

Mark this exact public probe completed and **not promoted as a quality/runtime winner**. Keep the checked ZIP, weights, negatives and matched controls. QAT currently earns archive compression, not a demonstrated leaderboard, RAM or CPU improvement. Do not fit blend coefficients to these public rows or repeat this unchanged probe for significance.

Continue meaningful controlled research: source-qualified F0/F3 and fitting seeds3/4 (finalists all four folds), longer/full-data learned-scale comparisons against fixed scales and floating/PTQ controls, and activation-aware QAT only after carried-state/native parity and throughput admission. Do not add another full member based on archive savings: this observed endpoint has only 5m34s deadline headroom. A packedINT4 execution path must earn measured CPU benefits separately. No universal family stop follows from this one small negative transfer result.

## Source records

[Exact public receipt](PUBLIC_RESULT_20261014_R1.json), [frozen release](RELEASE_R2.json), [local matched runtime](MATCHED_CALLBACK_RUNTIME_R2.json), [independent fitted diagnostic grade](../../../unified/reviews/QAT_FINAL_SELF_DISTILL_BLEND_GRADE_R4.md), [prior public references](../../team_models_r1/PUBLIC_LB_UPDATE_20261013_R2.md), [delivery package](../../../../submissions/10-14-qat-r1/README.md).
