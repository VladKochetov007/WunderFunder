# R1 family-target cohort input DAG

This packet prepares the frozen R1 raw-versus-clipped family-head pair. It does
not fit students, score predictions, upload a dataset, or push a Kaggle kernel.
The recipe remains the four commands in
`local/R1_FAMILY_TARGET_MODE_PAIRED.json`; raw and clip2 differ only in
`--distill-family-target-mode`.

## Inputs

| Input | Provider reference | Use and identity check |
| --- | --- | --- |
| Fold-0 assets | `vladkochetov07/wnn-c20261004-fold0-assets` | Split, target row orders, propensity caches, and the existing raw-wMSE s3/s4 train/valid predictions. Verify the pinned assets manifest and each file hash. |
| Raw features | `vladkochetov07/wunder-funder` | Train and valid parquet only. Use the portable registry’s byte sizes and exact footer identities. Feature-only teacher/student exports do not request labels or masks. |
| Own teacher checkpoints | `vladkochetov07/wnn-r1-source-teachers-e512-20261007` | The locally staged 15 MB content-addressed bundle includes the six independently trained source checkpoints, their provenance, the full-fine-tune initializer checkpoints, and propensity ancestry. It contains no prediction arrays or raw parquet. Manifest SHA-256: `98014f1463b1e377e94896c1906aa60c190a2d5b975fd1f89c2dde4bada51472`. |
| Same-fold baseline pair | Kaggle kernel source `vladkochetov07/wnn-raw-teacher-pair-e510-20261007` | E510 outputs the independently trained s3/s4 baseline checkpoints and full-row-order feature-only predictions. Its realized source-family snapshot is still pending; E512 must fail closed until it binds the exact fold and source hashes. |

The fold-0 fitting set is exactly 10,535 train groups plus 1,405 valid groups.
The 468 evaluation groups, group 528, and linked evaluation ancestors stay out of
all source fits. `train_targets.npz` has 10,606 rows and `valid_targets.npz` has
1,873 rows; these full row orders remain the order for teacher prediction
exports. The split manifest SHA-256 is
`09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`.

## Profile-only cache stage

The profile kernel consumes the inputs above, exporting the four missing strong
and full-finetuned teacher predictions on Kaggle from their pinned own
checkpoints. It reuses the two raw-wMSE prediction arrays from fold-0 assets and
the baseline predictions from E510 through `kernel_sources`; it never uploads
multi-gigabyte arrays from this laptop. The source coefficients and order stay
fixed: strong `.15 + .15`, full-finetuned `.10 + .10`, raw-wMSE `.125 + .125`,
same-fold baseline `.03125 + .03125`, for the unnormalized total `.8125`.

It writes only `main_source_{train,valid}.npy` and
`families_{train,valid}.npy`; the unused normalized and homogeneous main arrays
are omitted. The exact cache payload is 9,552,000,000 bytes for 11,940 fitting
groups. Four temporary prediction exports add 7,986,560,000 bytes, so the
profile worker must verify at least 19.1 GB free before writing and retain at
least 1.5 GB after the cache/export peak. If the live Kaggle volume cannot meet
that preflight, split teacher export and cache/profile into separate kernels
connected by `kernel_sources` rather than reducing checks or reserve.

After validating cache byte hashes, row order, fit membership, masks, and source
checkpoint ancestry, the worker removes only the four temporary strong/full
prediction arrays. Their hashes and sidecars remain bound by the cache metadata.
It then profiles the frozen clip2 s3 student command with one warm-up and two
measured batches, omitting `--save-training-state` because the trainer rejects
continuation-state saving during profile-only exits. No student fit or evaluation
starts. The terminal result is `profile_ready_awaiting_controller_review`.

The cache/profile kernel attaches E510 as a kernel source. Its worker discovers
the actual nested input mounts and validates the E510 realized source snapshot,
baseline checkpoint/provenance hashes, prediction sidecars, shape/dtype, and
finite values before building the cache. The Kaggle metadata schema accepts
`kernel_sources` as owner/kernel-slug references; because that field does not
pin a source version, the controller must bind the actual E510 version/output
receipt and the worker must reject mismatched attached bytes.

## Student production stage

Only after review of the actual E512 cache and measured 31-output profile will
the production kernel be materialized. It attaches the reviewed E512 output
with `kernel_sources`, plus the raw, fold-0, and own-checkpoint assets. It runs
the four frozen commands as a serial matched pair: clip2 seeds 3/4 and raw
seeds 3/4, each initialized from its own `c20261003_combo_s{seed}_f0@ep3`, two
epochs, fixed zero-based checkpoint epoch 1, with the remaining flags unchanged
from the frozen R1 JSON.

The student worker exports only the held-out 468 evaluation groups, in frozen
fold-0 order, using feature-only reads. Each result is a float32
`(468, 20000, 2)` prediction array with a sidecar recording the ordered groups,
student checkpoint hash, and profile/cache identities. The independent grader
owns the fixed official metric, paired comparison, and uncertainty report.

## Current stage

The source-cache code can now write only the R1-required main variant while
preserving the historical default. The staged teacher asset manifest is ready
locally. The E510 realized baseline snapshot and live Kaggle free-space/profile
receipt are the remaining packet inputs; no full cache/profile packet or
production packet has been pushed or run.
