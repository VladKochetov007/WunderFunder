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
- **C0 (labels-only child, matched control) done**: full-valid WP per epoch 0.68369 / 0.68391 / 0.68169 / 0.67831 / 0.67627 / **0.67562 (fixed ep5)** — six labels-only epochs at LR 1e-4 on train-only data overfit hard (−0.0111 vs its parent). Plan amendment 1 (made before any distilled result): adoption baseline = best undistilled single, max(P, C0) = P 0.68671. Earlier status: ([runner](run_child.sh)); runtime config verified:
  init `tt29_parent_s3@ep3`, 6 epochs, LR 1e-4, batch 16, `distill=""`, strict FP32, 10,325 train.
  D1/D2 wait for the fbc1k_s5 train-side cache.

- **Fallback teacher reproduction running** ([runner](run_rep_teachers.sh), [exact PR49 args](rep_teacher_args.py)):
  `tt29rep_fin256_s3` → train export → `tt29rep_fin256_s4` → export → `tt29rep_fbc1k_s5`
  (distilled from both fin256, α 0.5, boot-h 8, TL3-Muon×3, EMA) → train + full-valid exports.
  Note: fbc1k_s5 is itself an H256×2 GRU, i.e. a deployable student-shaped model.

## Incident: export stopped on stale target cache (13:16Z restart)

The `tt29rep_fin256_s3` fit finished (full-valid 0.68780 at ep3), but the train export failed:
`target index does not cover the full official training source`. The scratch root hardlinked
the Sep-26 `cache/train_targets.npz` (10,606 groups; row group 528 was undecodable in the
pre-Oct-3 download, E-035). The current `train.parquet` decodes 528, the fit used it (10,325 =
10,607 − 282), and the PR49 validator requires it. `rebuild_train_targets.py` rewrote the
scratch cache with all 10,607 groups after breaking the hardlink (the shared main cache is
untouched); the 10,606 old groups are bit-identical. Chain restarted, skipping the finished fit.
My earlier monitor watched only `local.log`, so the failure sat ~70 min unnoticed; the new
monitor also scans every fit/export log and the unit's exit.

## PR49 checkpoint release (arrived 12:17Z, not used in R1)

Semyon posted all 16 PR49 checkpoints (unquantized, SHA-verified):
[release](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr49-checkpoints-20261029-r1)
(81 MB zip; stored git-ignored and extracted at `submissions/pr49_tl3c_checkpoints_20261029/` — models under `extracted/tl3c_<name>/checkpoints/model_ep3.pt`; manifest and SHA-256 copies committed in [pr49_release/](pr49_release/)). Best fixed endpoints:
fttall_tlead_s5 0.691415, fbc1k_s5 0.691355, fttall_tlead_s6 0.690949, fin256_s3 0.688260,
fmu128_s3 0.687845. The local fallback was ~1.5 h in when it arrived and is kept for R1 because
the PR49 export/distill validators bind teacher, parent and child to one split-manifest hash
(release `split_sha256` ≠ our local file hash, same counts). Using the release requires either
PR49's exact split file or a membership-equality receipt; that is the cheapest next step for a
multi-teacher (tlead s5/s6 + fbc1k) cache.
