# Current status — paused by user

As of 2026-10-04T20:59:37.247427+00:00. The research goal is **paused**. No new fits, Kaggle pushes, timing trials, retries or research iterations are authorized. `NO_NEW_RESEARCH_ITERATIONS.json` and `runner/PAUSE.json` are the new pause records. Preserve older unrelated pause/failure markers.

## Completed drain

E425's existing four-epoch soft-routing fit, fixed epoch-3 FP32 export and matched reporting all finished successfully. Soft routing scored **0.696672177** on the historically exposed 368 excluded validation sequences. Differences against learned hard routing /uniform routing /matched GRU were **+0.000225892 /+0.000171562 /+0.000579733**; all 95% paired component intervals include zero. All four experts received prediction gradients on successful updates. No new OOM events. See [E425_DRAIN_RESULT.md](E425_DRAIN_RESULT.md).

E464's waiting controller was intentionally stopped before any of its eight head fits started. This is a user pause, not a scientific failure. E427 and loaded-host timing v4 remain unsealed source-only work. W384 v4 failed before any GPU work; v5 is held and its last guard refinement is untested. No additional research is running locally.

## Existing remote job

The already-submitted E463 TPU real-data preflight remains **QUEUED**. Its exact handle is `vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1-runner`. A read-only fifteen-minute collector observes this handle and downloads terminal outputs; it cannot launch or retry experiments. Status/results go to [E463_DRAIN_RESULT.md](E463_DRAIN_RESULT.md) and `checks/e463_pause_drain_collection.json`. The current collector files may update after this snapshot; those future updates are local until committed. No TPU result is claimed while queued.

## Incumbent and artifacts

Public best remains **0.6655**, fixed seed-3 combo i16 ZIP, versus previous 0.6641. Its actual complete supplied-validation callback scored **0.700503766**. Most supplied validation was fitted; on the 368 excluded sequences the score is **0.699822159**, and those sequences are historically exposed. Complete-recipe four-fold CV is **0.693658483** versus matched control **0.691244104**. See [VALIDATION_SCORECARD.md](VALIDATION_SCORECARD.md).

Two fixed endpoint ZIPs are ready for manual upload: head-only and parent-only. Seed-4 and q8r/q8x ZIPs passed parity/accuracy checks but still need runtime evidence. See [SUBMISSION_SHORTLIST_20261004T2041Z.md](SUBMISSION_SHORTLIST_20261004T2041Z.md). No new public improvement or newly deployable architecture is claimed.

## Resources at pause

No active local GPU attempts or live local reservations. Local charged GPU-job time is **11.0373 hours**, leaving **12.9627 hours** in the 24-hour local block. Conservative total device-GPU usage is at most **26.5130 hours** against the 80-hour allowance, leaving at least **53.4870 hours** before any future allocation. This deliberately overcounts provider GPU wall seconds as two allocated devices and is not exact campaign attribution.

Typed Kaggle quota remaining: GPU **22.2622 hours**, TPU **19.9626 hours**. The existing queued TPU job may consume TPU quota if assigned. Provider quota and device-GPU hours are different budgets. Raw snapshot: `checks/COMPUTE_AT_PAUSE_20261004.json`.

## Resume only on explicit user request

Read [RETROSPECTIVE_2026-10-04.md](RETROSPECTIVE_2026-10-04.md) and [FURTHER_RESEARCH_PLAN_PAUSED.md](FURTHER_RESEARCH_PLAN_PAUSED.md). Source-only drafts are not admission evidence. On resumption, refresh provider/local state and budget, remove only this pause's markers, and explicitly resume existing queues rather than blindly relaunching identities. Preserve unrelated markers and immutable failed attempts.
