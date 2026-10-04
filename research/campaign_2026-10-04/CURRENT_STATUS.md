# Current status — paused by user

As of 2026-10-04T20:59:37.247427+00:00. The research goal is **paused**. No new fits, Kaggle pushes, timing trials, retries or research iterations are authorized. `NO_NEW_RESEARCH_ITERATIONS.json` and `runner/PAUSE.json` are the new pause records. Preserve older unrelated pause/failure markers.

## Completed drain

E425's existing four-epoch soft-routing fit, fixed epoch-3 FP32 export and matched reporting all finished successfully. Soft routing scored **0.696672177** on the historically exposed 368 excluded validation sequences. Differences against learned hard routing /uniform routing /matched GRU were **+0.000225892 /+0.000171562 /+0.000579733**; all 95% paired component intervals include zero. All four experts received prediction gradients on successful updates. No new OOM events. See [E425_DRAIN_RESULT.md](E425_DRAIN_RESULT.md).

E464's waiting controller was intentionally stopped before any of its eight head fits started. This is a user pause, not a scientific failure. E427 and loaded-host timing v4 remain unsealed source-only work. W384 v4 failed before any GPU work; v5 is held and its last guard refinement is untested. No additional research is running locally.

## Existing remote job

The already-submitted E463 TPU real-data preflight remains **QUEUED**. Its exact handle is `vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1-runner`. A read-only fifteen-minute collector observes this handle and downloads terminal outputs; it cannot launch or retry experiments. Status/results go to [E463_DRAIN_RESULT.md](E463_DRAIN_RESULT.md) and `checks/e463_pause_drain_collection.json`. The current collector files may update after this snapshot; those future updates are local until committed. No TPU result is claimed while queued.

## Incumbent and artifacts

**Latest user-reported public best is 0.6660**, fixed seed-4 combo i16 ZIP, submission `2e5j6bnt`, runtime **37m07s**, memory **626 MB**. This reaches the numerical target at displayed portal precision, +0.0005 over seed 3 and +0.0019 over old D256. Seed-3 parent/head endpoints scored **0.6641 /0.6654**. See [the results and reflection](PUBLIC_RESULTS_REFLECTION_20261004_LATE.md) and [interactive climb](eda/leaderboard_climb/leaderboard_climb.html). Research remains paused.

Seed-4 cached complete supplied-validation WP is **0.700771292**, excluded 368 WP **0.701694557**. Seed 3's actual complete-file native callback scored **0.700503766**. Most supplied validation was fitted, and the excluded sequences are historically exposed. Complete-recipe four-fold CV remains the seed-3 result **0.693658483** versus matched control **0.691244104**; no seed-4 CV is claimed. See [VALIDATION_SCORECARD.md](VALIDATION_SCORECARD.md).

The original q8r ZIP failed public prerun (submission `1yv88h4s`): its native library requires `libpython3.13.so.1.0`, absent from the Python 3.11 scorer. **Both original q8r/q8x ZIPs are withdrawn.** Historical local parity does not establish scorer compatibility. A bounded packaging repair is permitted during the research pause; it does not resume research. See [E462_PACKAGING_INCIDENT_20261004.md](E462_PACKAGING_INCIDENT_20261004.md).

The fixed head-only/parent-only endpoints, seed-4 ZIP and seed-3 ZIP passed offline Python 3.11 callback checks on all three CPU builds. Their actual portal runtimes now provide deployment evidence; seed 4 no longer has a pending public/runtime gate. Repaired q8 ZIPs passed v2/v3 compatibility with bitwise archived-output equivalence; they remain withheld until runtime approval. The earlier [submission shortlist](SUBMISSION_SHORTLIST_20261004T2041Z.md) is superseded by these measured results and the withdrawn q8 notice. No research iteration was started to record these outcomes.

## Resources at pause

No active local GPU attempts or live local reservations. Local charged GPU-job time is **11.0373 hours**, leaving **12.9627 hours** in the 24-hour local block. Conservative total device-GPU usage is at most **26.5130 hours** against the 80-hour allowance, leaving at least **53.4870 hours** before any future allocation. This deliberately overcounts provider GPU wall seconds as two allocated devices and is not exact campaign attribution.

Typed Kaggle quota remaining: GPU **22.2622 hours**, TPU **19.9626 hours**. The existing queued TPU job may consume TPU quota if assigned. Provider quota and device-GPU hours are different budgets. Raw snapshot: `checks/COMPUTE_AT_PAUSE_20261004.json`.

## Resume only on explicit user request

Read [RETROSPECTIVE_2026-10-04.md](RETROSPECTIVE_2026-10-04.md) and [FURTHER_RESEARCH_PLAN_PAUSED.md](FURTHER_RESEARCH_PLAN_PAUSED.md). Source-only drafts are not admission evidence. On resumption, refresh provider/local state and budget, remove only this pause's markers, and explicitly resume existing queues rather than blindly relaunching identities. Preserve unrelated markers and immutable failed attempts.
