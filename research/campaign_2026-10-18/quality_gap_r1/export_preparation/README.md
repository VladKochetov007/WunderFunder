# Frozen F0 control export preparation

**Actual preflight completed, 15:27 UTC:** [execution and exact raw receipt](preflight_execution_r2/records/README.md) confirm full source hashes and historical split loader passed in 367.746 seconds. Model/checkpoint arrays and predictions were not loaded. The next step is the bounded no-training export; it is not yet admitted or run. Earlier preparation and running-state statements below are dated history.

## Current status, 18 October 15:22 UTC

F1 fits, exports, grade, R2 result collection, final-path verification and independent acceptance are complete. The strong-control and propensity artifacts have been restored and hashed; no further restoration is needed. [Isolated R2 compatibility](compatibility_overlay_r2/README.md) passed [independent static review](compatibility_overlay_r2/INDEPENDENT_REVIEW_R2.json), preserving the rejected R1 and its failure evidence. The bounded full-source CPU preflight is running in the separate [operational packet](preflight_execution_r2/README.md) under attempt `preflight-r2-20261018t1520z`. Its initial tar call failed before extraction because `--gzip` was omitted; the failure is retained, and a separate exact-empty-root recovery succeeded. No preflight result/pass, strong-control inference or measured residual comparison has been recorded. The older preparation/ownership statements below are historical.


**Status: prepared only; no export, grading, or training is admitted here.**
This note defines one bounded, no-training re-export of the restored Main F0
`clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1@ep5` checkpoint. The task's one
Beleriand GPU remains owned by the active F1 worker; root owns any later
execution and admission.

## Split identity

[`F0_MEMBERSHIP_COMPARISON.json`](F0_MEMBERSHIP_COMPARISON.json) compares the
frozen Main split (`09263ad4…`, 1,388,846 bytes) with the canonical current F0
split (`c09f7240…`, 1,388,846 bytes). The ordered `fit.train` (10,535),
`fit.valid` (1,405), `evaluation.valid` (468), all `excluded.*` lists, shared
fold metadata, and component-to-sequence map are equal. The manifests are not
interchangeable for the frozen predictor: its loader returns the SHA of the
raw split file and `predict_run.py` requires that value to equal the checkpoint's
recorded `split_manifest_sha256`. Keep the old `09263ad4…` file at
`research/campaign_2026-09-29/artifacts/splits/fold0.json`; do not replace it
with the canonical `c09f7240…` file.

The only source-content identity that differs is `cache/chain_folds.npz`:
the frozen split records 550,072 bytes / SHA-256
`a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f`, while the
current canonical source is 462,719 bytes / SHA-256
`a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567`. The
historical cache was not in the restored model archive. Its known recovery path
is `/home/vlad/development/wunder/wnn_connectome_starterpack/cache/chain_folds.npz`;
availability on that original host has not been verified. Other source records
retain their byte/hash/index/footer identities; their recorded mtimes differ.

## Runtime source closure

The local frozen predictor snapshot is
`research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot/`.
Its `scripts/predict_run.py` matches pinned SHA-256
`30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0`; the
paired trainer snapshot is pinned as
`a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6`.
The historical interpreter was
`/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python`
(Python 3.13.9). B's matching interpreter path has not been confirmed.

The exact restored checkpoint is at
`/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/restored-controls-v1/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt`
(3,983,015 bytes, SHA-256
`e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda`); its restored
provenance JSON is 166,152 bytes, SHA-256
`bdf643043884b1344564b636a2972ad52b8202ee30cf22598b1afd90405dd961`. The
10-file restore does not include the propensity checkpoint required by
`predict_run.py`: stage `artifacts/runs/c20260929_prop_f0/model.pt` only after
finding the exact 165,623-byte file with SHA-256
`b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7`.

The loader also needs the frozen split's logical sources under the isolated
project root: `datasets/train.parquet`, `datasets/valid.parquet`,
`cache/chains.npz`, `cache/chain_folds.npz`, `cache/train_targets.npz`, and
`cache/valid_targets.npz`. The pinned registry is
`research/campaign_2026-10-04/portable_inputs.json` (SHA-256
`90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c`). With
`WNN_PORTABLE_DATA_MANIFEST` set, the existing loader checks relocated source
bytes against that registry. The six sources total about 37.1 GB, so this is a
single process-level full-byte verification cost. The registry does not bless
the current `chain_folds.npz`; the recovered old cache or the explicitly
reviewed compatibility path below is still required.

## Preferred exact-source route

After the old cache and propensity checkpoint are recovered and their identities
verified, stage an isolated root at
`/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r1/root`.
Copy the old split and portable registry to their logical paths; make read-only
links for the six source files and restored checkpoints; keep `cache/` itself a
new writable directory so the prediction lands only in this stage. Bind B's
Python executable after recording its version and package/source resolution.
The inference command is the historical frozen predictor with an explicit new
output name:

```bash
export WNN_PROJECT_ROOT=/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r1/root
export WNN_PORTABLE_DATA_MANIFEST="$WNN_PROJECT_ROOT/research/campaign_2026-10-04/portable_inputs.json"
export PYTHONPATH="/path/to/frozen/source_snapshot/src"
"$PYTHON" /path/to/frozen/source_snapshot/scripts/predict_run.py \
  clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1 valid \
  --epoch 5 \
  --output-name clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1 \
  --campaign-split-manifest "$WNN_PROJECT_ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --device cuda --inference-precision fp32 --campaign-full-valid-feature-only
```

`/path/to/...` must be bound to the frozen snapshot and a verified B interpreter
before execution; these placeholders are intentionally not guessed. The
expected output is 1,873 ordered validation sequences, with a 299,680,128-byte
float32 prediction. Before grading, require the sidecar checkpoint SHA and old
split SHA to match the pins above, then compare the prediction SHA with the
historical full-valid CP5 pin `f933297d2b8a3cdc5bb1b1657a60120dec8421ed95cd37785077f6b49cbd37d6`.
The existing Main F0 grade is 0.6811553603333363 on the 468 evaluation groups.
A different prediction hash is a reproducibility observation to diagnose;
exact byte parity is not assumed across different hardware or libraries.
Retain new source/runtime-qualified predictions and report numerical and metric
comparisons with their limits. Historical byte equality is not an additional
grading gate, and differences do not authorize editing frozen source or splits.

## If the old cache cannot be recovered

Do not rewrite the frozen predictor, trainer, checkpoint, original split, or
portable registry in place. A separate `compatibility_overlay_r1/` may be
prepared for review: copy the frozen `src/wnn/` package into that isolated
overlay and change only the copied `splits.py` source-identity gate for
`cache/chain_folds.npz`. The gate must require this receipt's exact PASS status,
old split SHA `09263ad4…`, canonical split SHA `c09f7240…`, and current cache
identity `462719` bytes / `a112f809…`; all other source checks remain bound to
the original portable registry. Keep the loader's component-map reconstruction
and evaluation-membership check active against `cache/chains.npz` and the
current fold cache. Put the copied package first on `PYTHONPATH`, followed by
the unchanged frozen `source_snapshot/src`; invoke the unchanged frozen
`predict_run.py` and keep the old split bytes so the checkpoint split-SHA check
still passes.

This is a new compatibility export claim, not proof that the historical cache
was recovered. Pin and independently review the overlay diff and hash, then do
a CPU-only loader preflight for the old split SHA, exact ordered memberships,
component map, and 468-group evaluation before root considers GPU inference.
No overlay has been created or admitted in this preparation.

This note and the membership receipt are metadata preparation only. A fresh
remote metadata read could not be completed because B's SSH banner timed out;
no remote retry, numerical-array read, large-file hash, prediction, or training
was done.
