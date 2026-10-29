# Teacher-quality track — blend and distillation under the train-only protocol (R1)

User instruction (2026-10-29): use PR46/47/49 teachers; keep the Muon GRU student recipe fixed;
bounded investigation of teacher blending and distillation; adopt only at ≥ +0.001 on the
deployable configuration. Plan and fit budget (≤ 10 fits) declared before launch:
[PLAN_R1.json](PLAN_R1.json).

## Protocol (ADR0044, verified locally)

- Fit: official train only; eval: all 1,873 official valid sequences, official metric.
- The PR49 split builder was rerun here from the PR49 worktree (commit `188c0788e`) in a scratch
  root with hardlinked sources ([split](chronological_primary.json), [membership](MEMBERSHIP_R1.json)):
  10,325 fit / 282 excluded (train components connected to valid) / 1,873 eval — identical counts
  to PR49. File SHA differs from PR49's `0c4de3a2…` because it embeds local receipt mtimes; byte
  membership comparison pending PR49's file.
- Our historical students and blends were fitted on train+valid, so they **cannot** be scored on
  full valid; the train-only student chain is rebuilt here.

## Artifacts requested

PR49 teacher checkpoints, full-valid predictions and train-side exports live only on the
teammate's L4 receiver. Requested on [PR49](https://github.com/VladKochetov007/WunderFunder/pull/49#issuecomment-6492071041)
(user-approved route). Posted (unverified here) full-valid WP: fbc1k_s5 0.691355,
fttall_tlead_s5 0.691415, fttall_tlead_s6 0.690949, fin256_s3 0.688260, PR46 plain GRU256x2
Muon 0.677856.

## Execution

- Strict-FP32 launcher for the PR49 trainer: [strict_launch.py](strict_launch.py); shared args:
  [recipe_args.sh](recipe_args.sh). Profile ([log](profile_parent.log)): strict flags confirmed,
  10,325 train sequences, full valid, 160 finite updates, 19.9 seq/s.
- **Parent P seed 3 done** ([runner](run_parent.sh), [log](tt29_parent_s3.fit.log), [history](runs/tt29_parent_s3/history.json)):
  full-valid WP per epoch 0.67972 / 0.68490 / 0.68743 / **0.68671 (fixed ep3)**; finite, one thermal
  freeze/thaw. A train-only Muon H256×2 already sits at PR49 fin256-teacher level (0.68826, AdamW)
  and above the PR46 plain-GRU benchmark (0.67786, different batch/features).
- **C0 (labels-only child, matched control) running** ([runner](run_child.sh)); runtime config verified:
  init `tt29_parent_s3@ep3`, 6 epochs, LR 1e-4, batch 16, `distill=""`, strict FP32, 10,325 train.
  D1/D2 wait for the fbc1k_s5 train-side cache.
