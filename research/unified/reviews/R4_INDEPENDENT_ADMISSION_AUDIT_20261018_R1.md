# R4 independent admission audit — 2026-10-18

## Review scope

An independent `gpt-6-luna` / max-reasoning critic reviewed the R4 recipe,
runner, trainer, predictor, scorer and profile receipts read-only. The critic did
not fit models, read raw prediction arrays, edit files, access Kaggle or use
external compute. At review time R4 had no scientific fit or score. This is an
operational and measurement audit, not evidence for or against the feature
hypothesis.

## Findings and actor corrections

| Finding | Consequence | Correction in the current source |
|---|---|---|
| The runner expected `admission_r2.json`, but only the stale pre-profile `admission.json` existed. | The matrix could not launch with a receipt that bound its current plan and source. | The runner now rejects a missing, wrong-experiment, or non-ready admission. The final R2 receipt is generated from the repaired plan/source, split and input hashes, and the successful profile record before launch. |
| Resume identity was hashed after restoring the trained weights. | A resumed treatment/control pair could fail the initial-weight match even when both resumes were valid. | `record_model_identity` captures the seeded initial-parameter hash before loading resumed weights, and `resume_contract` binds it. |
| The matrix archived predictions but did not call the official grader. | “Pair complete” would have meant only that arrays existed, with no primary outcome. | The runner calls `score_run.py` on the archived prediction path, verifies the metric ID, split and prediction hashes, FP32 finiteness and fixed epoch, then stores a per-endpoint score receipt. |
| Matrix children did not pin the zero-worker loader used by the profile. | The short resource profile did not establish the launched loader configuration. | The child environment now sets `WNN_LOADER_WORKERS=0` and `WNN_LOADER_RAM_GB=1.5`. |
| The registered 44,760 updates ignored `drop_last` differences among folds. | Pair budgets could differ from the stated plan. | The plan records batches and optimizer updates by fold: fold 0, 1,492 / 44,760; fold 1, 1,493 / 44,790; fold 2, 1,495 / 44,850; fold 3, 1,488 / 44,640. |
| The pair verifier did not compare per-epoch order identities and update diagnostics. | Equal seeds and membership alone would not show the intended matched training order and budget. | Pair verification now requires identical per-epoch `sample_order_identity`, optimizer attempts/completions and batch counts, plus the planned total updates. |
| Prediction verification checked sidecar membership but not full array shape, dtype or finite values. | A malformed or nonfinite output could be called complete. | The runner now checks the epoch-2 checkpoint hash, run/split identity, exact evaluation groups, `(N, 20000, 2)` shape, `float32` dtype, prediction hash and finiteness in bounded eight-group chunks. The prediction command explicitly selects epoch 2. |

The runner hashes the large training and grading inputs once at the start of an
admitted, GPU-locked service; it does not repeatedly stream those files between
phases. Pair artifacts retain prediction arrays and group IDs. Targets and masks
remain in the hash-bound `cache/valid_targets.npz`, avoiding duplicate target
payloads for every fit.

## Remaining limits and launch interpretation

The five-batch training profile measured 2.71 GB cgroup RAM, with zero profile
OOM-counter change, but did not exercise epoch-boundary training-state saves.
The prediction smoke covered all fold-0 evaluation groups at epoch 0; it was a
compatibility artifact, not an official scientific grade. The final run remains
inside the 5,000,000,000-byte, one-core, no-swap task cap, with the shared GPU
lock and phase resource telemetry. The 113.8-hour estimate is for the 60 fits
on the local RTX 3060, before setup and evaluation overhead.

Once admitted, the result can answer whether this fixed feature bundle helps a
strong from-scratch H512×1 GRU on exposed component folds. It cannot establish
pristine confirmation, public transfer, teacher/student value, or submission
readiness. The user explicitly requested all 30 pairs and asked that early weak
scores not be used to reject the direction.

## Audit provenance

- Independent critic: read-only, `gpt-6-luna`, max reasoning; no fit, data
  payload read, or changes.
- Root corrections: the source links in the
  [R4 campaign README](../../campaign_2026-10-17/geometry_from_scratch_r4/README.md),
  [matrix runner](../../../scripts/run_geometry_r4_matrix.py),
  [scorer](../../../scripts/score_run.py),
  [trainer identity](../../../src/wnn/training/setup_model_runtime.py), and
  [resume contract](../../../src/wnn/training/checkpoints.py).
- Static lint and format are engineering checks only. No formal test suite or
  scientific fit is claimed by this review.

## Post-audit execution addendum — 2026-10-18

The first controller attempt (`geometry-r4-matrix-20261018-r1`) ended before
runner startup because it invoked a non-executable Python script directly. The
corrected Python entrypoint then started under attempt R2, but its frozen
`train.parquet` digest had been mistyped while generating the admission JSON.
The runner compared the actual file and stopped before the first fit. The
controller recorded a 4.50 GB cgroup peak under its 5 GB hard cap; memory was
mostly inactive file cache, and `memory.events` reported zero `max`, `oom`, or
`oom_kill` events. No optimizer updates or scientific scores were produced.

All three data files were rehashed in 1 MiB streaming blocks, all four split
manifests reverified, and admission R3 now binds those values plus the current
source tree. R2 and its attempt receipt remain unchanged. This catches an
actor-created receipt transcription error; it was not a finding from the
independent critic and does not change the scientific hypothesis.
