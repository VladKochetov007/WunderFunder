# E450: fixed seed-4 replication and equal seed bag

**Public result received:** fixed seed-4 ZIP scored **0.6660**, submission
`2e5j6bnt`, 2026-10-04 23:48 portal clock, 37m07s /626 MB. Its SHA-256 matches
the archive below. This supersedes the historical public/runtime-pending
statements in this report; the equal seed bag is still unsubmitted. See
[the reflection](PUBLIC_RESULTS_REFLECTION_20261004_LATE.md).

The seed-4 parent completed all four epochs, selected fixed epoch 3, then
completed the same seed-3 head fine-tune at fixed epoch 0. Both full-validation
FP32 exports finished. All four local jobs returned zero with verified teardown
and no new worker OOM events. Parent peak observed VRAM was 4,484,759,552 bytes;
peak cgroup RAM, including charged file cache, was 22,672,068,608 bytes.

The frozen paired screen completed in 30.77 seconds, with 1,222,365,184-byte peak
RAM. Evaluation uses the same 368 historically exposed excluded sequences and
249 components. These results are not an untouched validation estimate.

| Fixed predictor | Global WP |
| --- | ---: |
| Seed-3 equal parent/head, public-confirmed recipe | 0.699822159 |
| Seed-4 equal parent/head | 0.701694557 |
| Equal seed-3/seed-4 bag | 0.701420130 |
| Seed-4 parent alone | 0.700363274 |
| Seed-4 fine-tuned head alone | 0.701577995 |

| Preregistered comparison | Difference | 95% paired component interval |
| --- | ---: | --- |
| Seed 4 minus seed 3 | +0.001872398 | [-0.000975354, +0.006931454] |
| Equal seed bag minus seed 3 | +0.001597972 | [-0.000039884, +0.004622049] |
| Equal seed bag minus seed 4 | -0.000274427 | [-0.002340905, +0.000929490] |
| Seed-4 head minus parent | +0.001214721 | [-0.000153328, +0.002491309] |

All intervals include zero. The seed-4 versus seed-3 point difference is mostly
associated with class 0 (+0.006151173); class 3 is slightly lower (-0.000250381).
The bag raises class-3 WP by about +0.000405672 versus seed 3. Class correlations
do not add linearly to global WP. These intervals condition on fitted models;
they do not establish a training-seed distribution or selection correction.

The seed-4 package build is a separate bounded CPU attempt. Manual-upload
readiness still requires actual callback/parity/reset/causality and runtime
evidence. Its public score is unmeasured. The public-confirmed seed-3 score
0.6655 is an empirical anchor, not a calibrated narrow forecast for seed 4.
The fixed two-seed bag must meet CPU runtime limits or serve as a teacher for a
separately registered student; its descriptive score alone does not authorize
deployment.

Raw evidence: `checks/e450_report_seed4_bag_fp32_v1.json`, frozen design
`checks/e450_screen_seed4_bag_fp32_v1.json`, sealed scorer specification
`specs/e450_score_seed4_bag_fp32_v1.sealed.json`, all four E450 fit/export worker
records, and `E450_SEED4_REPLICATION_PREREG.md`. Marimo-generated figures and
source hashes are under `eda/seed4_replication/`.

## Actual seed-4 native package checks and supplied-file scorecard

The actual new seed-4 ZIP (`submissions/c20261004_combo_s4_i16/submission.zip`, SHA256 `c4dc258b529a52267a5e108ef0936f7919221cbdd17913270c8eb3482557f94a`) completed 20 full streams /400,000 rows with v2/v3 bitwise agreement, exact targets/masks, reset A-B-A and future-suffix causality checks. Native i16 versus native f32 maximum error was 0.0000978410 and WP change +0.0000000286. Root inspected callback logs confirming actual i16 precision. The bounded worker finished in164.15s with580,894,720B peak RAM, no GPU/OOM and verified teardown. This establishes parity; quiet CPU timing remains pending. Initial never-run verifier source and root fixture fixes are preserved separately.

The fixed FP32 whole supplied-file scores are seed3 **0.700503744**, seed4 **0.700771292**, equalbag **0.701106873**; see `VALIDATION_SCORECARD.md` for the fitted/excluded split. These are descriptive reporting-only results, with no public score for the new candidates.

Actual raw checks: `checks/e450_seed4_native_parity_v1/readiness.json`, `runner/attempts/e450-seed4-native-parity-v1.json`, `checks/e450_full_valid_fixed_models_v1.json`. Timing is durably queued behind E426 in `wnn-timing-window-v1.service`; E425 is ordered after the timing window. A missing quiet window is recorded as deferred, never a runtime pass.
