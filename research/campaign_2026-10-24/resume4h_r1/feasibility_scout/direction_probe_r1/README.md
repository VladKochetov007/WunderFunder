# Fixed prediction direction probe

This is a no-fit diagnostic of score geometry around saved, already exposed endpoints. It does not establish clean cross-validation, a causal training explanation, a serving coefficient, or a public-score forecast.

## R1 failure record

The admitted R1 command stopped before either reused preflight ran. Its first recipe lookup expected lowercase `inputs.f0` / `inputs.f1`, while the frozen recipe uses uppercase `inputs.F0` / `inputs.F1`. The 0.219-second failure and traceback are preserved in [execution receipt](EXECUTION_R1.json) and [execution log](EXECUTION_R1.log). No prediction arrays or target labels were read.

The exact R1 runner, recipe, source manifest, and admission are preserved under [r1_snapshot](r1_snapshot/). Their SHA-256 values match the R1 recipe and admission pins.

## R2 repaired packet

- [R2 recipe](DIRECTION_PROBE_RECIPE_R2.json) SHA-256: `a275cdd402aa8e22e181922dd3de0e4f1dee06a5a58fe579709c7f557fdc9a29`.
- [R2 source manifest](DIRECTION_PROBE_SOURCE_MANIFEST_R2.json) SHA-256: `7265890a44ebbcdd46ac8030a3050ad0fb2a90d2642020db4f514077554859a1` (33 sources, including the immutable R1 packet and failure receipt).
- [Producer/consumer contract audit](PRODUCER_CONSUMER_CONTRACT_R2.md) maps every runner-consumed input key and PreparedF0/PreparedF1 field to the actual preflight and grade schemas.
- [Independent R2 code review](CODE_REVIEW_R2.md) corrects the too-broad R1 startup assessment and confirms the repaired label-free input-preparation path against the actual recipe, producers, source pins, and fixture receipt; the review predates the separately admitted score pass.

R2 fixes the input-key casing, cross-binds every recipe/grade/source-manifest/preflight path and digest to the top-level source map, and provides an admission-gated `--preflight-only` exit. This mode runs both existing label-free ancestry preflights, rechecks prediction file hashes and NPY headers, checks the pinned target archive hash and size, prints a receipt, and returns before `run_probe`. It does not decode target labels or prediction values.

F0 contains five unique endpoints over 468 source-qualified groups and 141 components; A lag-16 is an exact alias of A lag-1. F1 A is evaluated separately over 457 groups and 149 components. Both folds are exposed development evidence. The official metric remains `global_wp_clip2_v1`. Fixed alpha 0.1 predictions use the existing float32 operation; derivatives use real arithmetic. Alpha-zero projections use the shared parent base, and all analysis is descriptive.

The real label-free R2 preflight passed in 6.2 seconds. Its [receipt](DIRECTION_PROBE_PREFLIGHT_R2.json) has SHA-256 `81df5e2689ef4c5def60aa34bdd545ac28e63d34f10aa94abe6ff4b466b24fc0`. It verified both saved preflight reports, the 33-source map and input path/hash bindings, all eight prediction hashes and NPY headers, the target archive SHA/size, and the frozen group/component identities. It records that no label or prediction values were read and that the scoring path was not called.

## R2 label-free preflight command

Run only after root issues the R2 admission SHA. The bounded resources are one CPU core, 400,000,000 bytes, no swap, physical-disk scratch, and 60 seconds for this preflight-only pass.

```sh
env WNN_MEM_MAX=400000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_RUNTIME_MAX=60s WNN_SCOPE_UNIT=wnn-direction-probe-r2-preflight-20261024.scope WNN_TMP_ROOT=/home/vlad/development/wunder/research_tmp scripts/limited uv run python research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/run_direction_probe.py --recipe research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_RECIPE_R2.json --recipe-sha256 a275cdd402aa8e22e181922dd3de0e4f1dee06a5a58fe579709c7f557fdc9a29 --admission-sha256 <root-issued-admission-sha256> --preflight-only
```

The R2 score pass has now completed under separate root admission. Its [result note](RESULTS_R2.md) summarizes fixed-endpoint scores and conditional diagnostics; the [independent result review](RESULT_REVIEW_R2.md) confirms point-moment arithmetic and projection ranks, while marking bootstrap interval reproduction as incomplete. See the exact [result JSON](DIRECTION_PROBE_RESULT_R2.json), [execution receipt](EXECUTION_R2.json), and [component-moment artifact](DIRECTION_PROBE_COMPONENT_MOMENTS_R2.npz). This is exposed, fixed-prediction evidence with zero new fits.

The [R2 independent runner review](CODE_REVIEW_R2.md) covers only label-free input preparation and source/schema consistency. It is distinct from the result review and does not independently grade the saved predictions.

## Synthetic math validation

[R5 synthetic receipt](DIRECTION_PROBE_SYNTHETIC_R5.json) is the current fixture. It uses generated arrays only and passed under 150 MB, one core, and 60 seconds. Float64 finite-difference errors at step `1e-6` were `3.6e-8` at alpha 0 and `3.7e-8` at alpha 0.1. Raw and centered moment parity against direct rows was within `2.3e-16`; the component fixture exercised a calibration-plus-A rank change from 1 to 2. It checks both open and stays-closed prediction-SD epsilon boundaries and final correlation clipping. At the exact epsilon boundary, an opening direction has no right derivative; a stays-closed direction has zero derivative, and both boundary cases are excluded from regular projections.

Earlier [R1](DIRECTION_PROBE_SYNTHETIC_R1.json), [R2](DIRECTION_PROBE_SYNTHETIC_R2.json), [R3](DIRECTION_PROBE_SYNTHETIC_R3.json), and [R4](DIRECTION_PROBE_SYNTHETIC_R4.json) receipts are iteration history. R5 supersedes them for the exact prediction-SD epsilon closed-branch behavior.

## Completed R2 result

- [Scientific findings](RESULTS_R2.md) separate variance overlap from score contribution: ordinary calibration explains <0.1% of F0’s score derivative, while B/C largely share the common A direction. Residual derivative intervals span zero; F1 is much weaker. No new fit, serving coefficient or public forecast.
- [Full result](DIRECTION_PROBE_RESULT_R2.json) preserves all per-target estimates, conditional bootstrap intervals, clipping transitions and exact reproduction of prior official scores. [Component moments](DIRECTION_PROBE_COMPONENT_MOMENTS_R2.npz) retain complete additive/centered tables for independent reconstruction; they are exposed derived data, not fresh validation.
- [Root admission](DIRECTION_PROBE_ADMISSION_R2.json), [execution receipt](EXECUTION_R2.json) and [empty successful log](EXECUTION_R2.log) bind the actual run: one CPU,400MBcap, zero OOM/swap, unchanged target/prediction hashes.
- [Independent result review](RESULT_REVIEW_R2.md) verifies pooled moments, derivatives and projection ranks numerically. Its independent bootstrap reconstruction remained incomplete after reviewer-script errors; actor intervals are not represented as independently revalidated. The source/mathematical fixture remains separately documented.
