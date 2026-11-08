# Stage 2 and Stage 3 rolling operator

These commands operate on the frozen run created by `multistage_search.py prepare`.
They use the approved training source snapshot from `run.json`, the train-only
auxiliary split, one admitted GPU per worker, and the campaign deadline/reserve.
The rolling supervisors launch profiles and fits as detached owned workers. They
wait for the external review receipt before admitting a successor fit.

## Stage 2: fresh optimizer continuation

`roll-stage2` proposes one two-epoch child per supplied fixed-final Stage 1 result.
TPE samples a lower learning-rate multiplier from `0.1`, `0.25`, and `0.5`; pass
`--lr-multiplier` to enqueue a specified admitted value for a planned anchor while
still recording it as an Optuna trial. All other model and fitting settings remain
identical to the parent. The parent
endpoint is passed through `--initializer`, which loads model weights only and
creates a fresh optimizer. The parent’s control WP is retained as the skip
reference; the skip reference consumes no new fit attempt.

```text
python scripts/multistage_search.py roll-stage2 \
  --run-dir <runtime>/multistage-optuna-20261031-r1 \
  --parent-result <runtime>/multistage-optuna-20261031-r1/stage1/trials/000001/trial_result.json \
  --parent-result <runtime>/multistage-optuna-20261031-r1/stage1/trials/000002/trial_result.json \
  --lr-multiplier 0.1 --worker-prefix stage2 --max-workers 6 --poll-seconds 5
```

Each completed profile creates
`stage2/trials/<number>/profile/successor_profile_review_template.json`. Review
the native profile, exact config, initializer checkpoint, rank telemetry and
attempt receipt. Copy the template to
`successor_profile_review_receipt.json`, set `status` to `approved`, and fill
`reviewer` (`root` or `independent`) and `reviewed_at_utc`. Keep every hash and
identity field unchanged. The supervisor consumes the receipt and launches the
fit. Each Stage 2 fit uses the same auxiliary fit/control memberships and keeps
the fixed parent WP as its plotted reference.

## Stage 3: fixed-final teacher pool and matched students

Export each qualified fixed-final label-only teacher from its exact Stage 1 or
Stage 2 result. This exports fitting members only, with the selected measured
sequence batch size recorded in its export receipt.

For an imported completed fit, pass its campaign result file explicitly; the
worker verifies it is complete, stage-matched and inside this run before exporting.

```text
python scripts/multistage_search.py launch \
  --run-dir <runtime>/multistage-optuna-20261031-r1 \
  --mode teacher-export --stage 1 --result-path <runtime>/multistage-optuna-20261031-r1/stage1/preexisting/<attempt-id>/trial_result.json \
  --worker-id teacher-preexisting-<family> --gpu-key <admitted-free-gpu>
```

```text
python scripts/multistage_search.py launch \
  --run-dir <runtime>/multistage-optuna-20261031-r1 \
  --mode teacher-export --stage 1 --trial-number <trial> \
  --worker-id teacher-s1-<trial> --gpu-key <admitted-gpu>
```

The worker returns `stage3/teachers/stage1-trial-<number>/teacher_export.json`.
Build a frozen ordered weighted cache from at least two exports. Weights must be
positive and sum to one; pool targets are summed in FP32 without clipping.

```text
python scripts/multistage_search.py stage3-teacher-pool \
  --run-dir <runtime>/multistage-optuna-20261031-r1 --pool-id gru-lstm-r1 \
  --teacher-export <runtime>/multistage-optuna-20261031-r1/stage3/teachers/stage1-trial-000001/teacher_export.json --weight 0.5 \
  --teacher-export <runtime>/multistage-optuna-20261031-r1/stage3/teachers/stage1-trial-000002/teacher_export.json --weight 0.5
```

`roll-stage3` proposes matched two-epoch pairs using the supplied reviewed
student config. Each pair shares model, seed, batch, optimizer and data; only
`kd_weight` differs. The control has `kd_weight=0` and no teacher cache. The KD
student uses the frozen pool and additive
`L_label + kd_weight * L_teacher` loss. TPE samples from the admitted KD weights.
Both exact profiles require external review receipts before either fit runs.
After both fits complete, the operator writes a pair report and PNG/SVG comparison
with the KD-minus-control inner WP.

```text
python scripts/multistage_search.py roll-stage3 \
  --run-dir <runtime>/multistage-optuna-20261031-r1 \
  --family gru --base-config <packet>/recipes/r6/gru256x3.json \
  --teacher-cache <runtime>/multistage-optuna-20261031-r1/stage3/pools/gru-lstm-r1 \
  --pairs 2 --worker-prefix stage3 --max-workers 6 --poll-seconds 5
```

Each fit retains its producer-native training and control journals, selected
checkpoint, actual score, resource receipt and PNG/SVG plots. Search objectives
remain the source-qualified all-after-warmup train proxy; no primary-valid data
enters proposal, normalization, fitting, teacher targets or selection.

## Importing a completed external Stage 1 fit

`adopt-completed-fit` validates a pre-registered attempt, the original PID/birth/PGID
owner receipt, exact approved profile and external review hash, final checkpoint,
terminal status, and four-epoch index before importing artifacts. The owner process
must no longer be live. When the launching wrapper did not preserve its child's OS
exit code, the terminal receipt must say `artifacts_complete_exit_unknown` and
`exit_code: null`; the adoption record keeps that limitation explicit. It never
turns an unknown exit observation into a successful exit code or reruns the fit.
The imported artifacts stay inside the original auxiliary fit cohort, and selected-
control prediction plus scoring are recreated with the campaign's pinned evaluation
source. The command takes the exact source artifacts and original process receipt:

```text
python scripts/multistage_search.py adopt-completed-fit \
  --run-dir <runtime>/multistage-optuna-20261031-r1 \
  --attempt-id preexisting-stage1-gru-b256-r5 \
  --config <runtime>/requests/gru-b256-prefetch1-r5.json \
  --approved-profile <runtime>/profiles/gru-b256-r5/approved_profile_receipt_r3.json \
  --owner-receipt <runtime>/receipts/owner-gru-b256-r5.json \
  --terminal-receipt <runtime>/receipts/terminal-gru-b256-r5.json \
  --gpu-key <admitted-free-gpu>
```
