---
kind: experiment
status: admitted
created_utc: 2026-10-13T00:08:00Z
hypothesis_ids: [H-STRATEGY-AVERAGING-20261013]
experiment_ids: [E-STRATEGY-SWA-F0-20261013-R1]
evidence_paths: [CONSTRUCTION_RECEIPT_R1.json, S3_MANIFEST_R1.json, S4_MANIFEST_R1.json, EXPORT_AND_GRADE_COMMANDS_R1.json]
---

# Fixed trajectory averages on clean F0 controls

## Question and method

Does the equal parameter mean of the predeclared epochs 3, 4 and 5 improve the same seed's final checkpoint without additional fitting or inference cost? This is the inexpensive fixed-checkpoint arm of S37 from the [first-principles assessment](../../../unified/ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md). It is separate from EMA-only retraining and from consistency training. It does not reproduce classical SWA's usual constant or cyclical learning-rate tail: these checkpoints come from the existing six-epoch OneCycle trajectories.

The [seed-3 manifest](S3_MANIFEST_R1.json) and [seed-4 manifest](S4_MANIFEST_R1.json) each bind three checkpoints from one own seed-specific training trajectory. Epochs 4 and 5 continue the actual epoch-3 state. Each average retains its own seed's identical normalization buffers. No averaging occurs across seeds. Floating parameters are accumulated in FP64 and cast to their original dtype. The [constructor](../../../../scripts/average_trajectory_checkpoints.py) rejects differing scientific arguments, ancestry, tensor schemas, normalization, missing values or inconsistent epoch-3 continuation records. It writes a new run and explicit weight-derivation provenance without changing any parent artifact.

## Evidence and limits

The [construction receipt](CONSTRUCTION_RECEIPT_R1.json) records three passing correctness tests and successful construction from both actual producer trajectories under a 1.2 GB CPU support bound. Each output checkpoint is 3,984,863 bytes; weights remain in ignored `artifacts/runs/` storage at the receipt's exact paths and hashes. The receipt also corrects an approximate timestamp assigned too far ahead in the original manifests; their hash-bound contents are preserved.

No predictions or quality scores exist for these averages yet. Compare seed 3 against its CP5 control (0.68115536) and seed 4 against its CP5 control (0.68125842), then report the mean of paired changes. These student seeds share one teacher cohort. F0's 468 groups and 141 components are repeatedly exposed, so this is discovery evidence, not pristine confirmation or a public forecast. Seed agreement does not measure teacher-training variation. The averaging choice was fixed before its outcomes; do not select a different epoch subset after looking at scores.

## Delivery

Root owns this arm through prediction and grading. [Exact export and grade vectors](EXPORT_AND_GRADE_COMMANDS_R1.json) use the same frozen FP32 feature-only consumer as the controls, with matching default output filenames. The [independent review](../reviews/trajectory_average_f0_r1/README.md) passed actual tensor means, continuation ancestry and strict loading through that consumer. GPU export still requires fresh shared-controller admission. The current amplitude worker chain retains the single local GPU lane. Each full-valid prediction payload is 299,680,000 bytes before the NPY header; use physical storage and bounded actual-producer readers. The grader must retain the official pooled metric, original-label weights and fixed scored mask. A supported gain would still require other folds and actual serving validation before promotion.

See the [campaign](../README.md), [coverage register](../COVERAGE.json), and append-only [experiment ledger](../../../experiments.jsonl).

[Controller vectors](EXPORT_CONTROLLER_COMMANDS_R1.json) wrap the independently reviewed exports with a ten-minute timeout, the existing5.5GB export reservation, exact absolute contract and shared GPU admission. The [binding preflight](EXPORT_CONTROLLER_PREFLIGHT_R1.json) recomputes the immutable source manifest and verifies the actual model/consumer hashes. Root owns two sequential exports after the amplitude chain; no fit is admitted. Each launch still needs current memory and physical-storage checks.

Both reviewedmeans exported and graded: [s3](grade_s3_r1/clean_t3_f0_control_s3_swa345_strategy_r1.grade.json) .6811988703, delta+.000043510; [s4](grade_s4_r1/clean_t3_f0_control_s4_swa345_strategy_r1.grade.json) .6813158771, delta+.000057455. Meanpairedchange+.000050483 onthe same exposedF0. [Paired uncertainty/cost](../first_results_analysis_r1/README.md) supportscheapfollow-up, not completeendpointpromotion.

[Grade folder and provenance](grade_s3_r1/README.md).

[Grade folder and provenance](grade_s4_r1/README.md).
