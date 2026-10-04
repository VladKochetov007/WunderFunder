# E426 event-clock EMA: fixed-checkpoint scientific result

Actual result recorded 2026-10-04T20:11:24.136583+00:00. All three four-epoch fits, fixed epoch-3 FP32 full-VALID exports, and paired fold-0 score completed with successful teardown and no new OOM.

| Clock | Fold-0 pooled WP |
| --- | ---: |
| constant | 0.681527822 |
| aligned | 0.681481895 |
| shuffled | 0.681400532 |

| Contrast | Difference | 95% component interval |
| --- | ---: | --- |
| aligned minus constant | -0.000045927 | [-0.000164898, +0.000071328] |
| aligned minus shuffled | +0.000081363 | [-0.000095696, +0.000305810] |
| shuffled minus constant | -0.000127290 | [-0.000315019, +0.000018401] |

The event-aligned consistency-teacher clock did not outperform the constant-clock control in this screen. All three paired intervals include zero. This tests one exposed fold, seed3, the preregistered clock definition and existing consistency strength; it does not reject event-time representation changes or establish a public-score effect. The evaluation EMA remained fixed and independent from the varied consistency EMA. All arms completed29,840attempts,29,828successfulupdates,12AMPskips and29,840EMAupdates.

The score uses468evaluation sequences,141connected components,2,000paired replicates and seed271828; component intervals are conditional on these fitted models. Teacher-lineage exposure and historical fold selection remain explicit limitations. No epoch or weights were selected from these scores.

Raw score: `staging/e426_clock_comparison/results/e426_clock_screen_v2.json`. Exact score runner: `runner/attempts/e426-clock-fold0-paired-score-v2.json` (14.387s,451MBRAM,noGPU/OOM). Provenance: `checks/e426_fit_provenance_preflight_v1.json`. Plot rendered through live marimo cellBLRt: `eda/e426_event_ema/paired_clock_screen.png` and `.svg`.

Root repaired orchestration separately from scientific code. Queuev2 was stopped when a successful historical transient service had been garbage-collected; queuev3 accepts only its exact successful runner record, verified teardown, artifact hashes and noOOM. Earlier failed source fixtures and stopped controller are preserved. No scientific fit/export was retried.
