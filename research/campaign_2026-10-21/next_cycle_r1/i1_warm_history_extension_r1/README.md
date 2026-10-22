# Conditional warm start for i1 history

This packet prepares a narrow checkpoint continuation for a possible later
matched F0/F1 follow-up. It extends only a standard 211-input GRU parent with
`research_features=none` to the fixed 367-input `i1_history` child, with
`drop_i1=false` and `none` or `all` feature control. The existing canonical
feature-prefix expansion preserves the parent parameters and normalization,
and starts all 156 appended input weights at zero.

The new default-off `--i1-warm-normalization fit-appended` option fits only the
appended mean and scale from the first two batches of the existing fitting
loader, using the same flattened `[::7]` sample and estimator as canonical
normalization. It restores and hashes the inherited 211-slot mean and scale,
and records identities for those buffers and the newly fitted suffix. The
default `inherit` path does not fit or alter normalization.

The source also adds focused synthetic CPU tests for parent-function parity,
sequence/chunk state handling, mask restoration, exact preservation of the
parent normalization prefix, and fail-closed configuration checks. They have
not been run in this preparation worktree; root owns bounded test admission and
review. No tensors from research checkpoints, dataset arrays, model fits, GPU
work, new recipes, or scientific selection were performed here. The mechanism
remains conditional on completion and interpretation of the active i1 and
dynamics screens.

## WMSE checkpoint-layout compatibility restoration

The frozen prior F0/F1 child commands include `--mean-variance-head 1` because
their 23-row parents were continued with a 25-row model. The current canonical
trainer branch had lost that optional path, so the old commands could not parse
and a direct 23-to-25 state load would fail. This isolated code change restores
only the dormant WMSE layout contract: the flag defaults off, two zero rows are
appended after existing outputs under `torch.random.fork_rng`, and the initializer
copies all parent tensors while retaining the child’s two zero rows. The
initializer records parent and child output-row counts. It rejects incompatible
architecture, precision, loss, averaging, control, QAT, rank, or dynamics-head
settings and any state-dict difference beyond those two rows.

The implementation source was recovered from the pinned prior trainer at
commit `68ff682977b822f38351aac0f621146f5f13b440`; its original
`src/wnn/training/mean_variance.py` SHA256 is
`e4a067bb9d467d121720854876cfffb083ab803a4ed440ec96025779553e15c5`. The
current canonical change reuses its output appender and shape-copy semantics,
with a fail-closed guard for the current dynamics head. The main loss module is
unchanged. The selected `--loss wmse` calculation and evaluation use only the
first two outputs, so these rows receive no gradient and no variance is learned.
Both arms carry the same 25-row topology solely to preserve checkpoint
compatibility. This interpretation is independently confirmed by a source audit
of the pinned loss, initializer, and evaluation paths.

Five data-free tests cover unchanged readout values, zero added rows, preserved
RNG state, 23-to-25 state copying, the default-off guard, and fail-closed shape
validation. They are prepared for root’s bounded
CPU test slot and have not been run here. The actual parent tensors, appended
normalization identities, and parent-output parity remain mandatory pre-fit
gates; this note does not claim they have been checked.

Current isolated source hashes for this restoration are `mean_variance.py`
`729cb7d8f9d48f751cc8bca269b4e60acfce5432460398ed63d97c23a49e93d5`,
`cli_targets.py` `5ca705b8a9cb88835e9a15d223813495ae72d2698872da60e2bf8ed7ed5d4fa8`,
and `setup_model.py`
`12b537612b5c9bbd2e17b279054d4f93ad367aea74a455e6c965275673e2c7b1`.
`losses.py` is unmodified in this code change (snapshot SHA256
`19c301948734373e706c3b59e784cdb65f33d899f011a33aacd59e2db34f3f2a`). The
source-specific command plan will bind the complete trainer source tree before
any profile or fit admission.


## Exact-warm profile plan

[`STRONG_PARENT_WARM_I1_PLAN_R1.json`](STRONG_PARENT_WARM_I1_PLAN_R1.json) contains four frozen two-epoch CP1 fit commands, exact warm-history profiles for F0/F1, the expected source SHA map, and the full-stage cost gate. Each profile retains its actual `--init`, target cache, propensity checkpoint, feature prior and `fit-appended` normalization; it omits only `--save-training-state`, which canonical validation rejects together with profile early exit. Each profile uses ten warmup and one hundred measured batches. Profiles are engineering measurements only and root owns their admission.

[`PROFILE_F0_WARM_R2.json`](PROFILE_F0_WARM_R2.json) is the separately source-pinned F0 profile handoff for immediate review. It preserves the real strong F0 parent and every target/propensity/feature/objective path. The profile is not a scientific fit and does not replace the actual-parent tensor, shared-normalization or parent-output parity gates before the four-fit panel.

The historical cost projection is 7.92h using a 900-second ceiling for each of two profiles, 15% margin on four 1,492-batch fits, and explicit export/grade reserves. It leaves little slack. Replace historical fit/export estimates with observed phases before admission; preserve the complete four-fit panel and all full-fold predictions. No fit, profile, tensor read or quality grade has been run from this packet.

## Conditional paired grade

[`grade_strong_parent_warm_i1.py`](grade_strong_parent_warm_i1.py) is a
code-only, create-only grader for the predeclared four-fit screen. Before it
opens the official target cache, it checks the frozen plan and source hashes,
direct parent and cache metadata, teacher and propensity ancestry, exact fit
membership and update clocks, paired initialization/normalization/prior/sampler
identities, and all four full-valid prediction sidecars, hashes, NPY headers,
dtype, C-order and row-group coverage. It then uses the existing
`global_wp_clip2_v1` scorer and i1 component-bootstrap helpers to report both
history-minus-control and fixed 90/10 control/history-minus-control on exposed
F0/F1. The output includes each target, the four fixed position bins and
2,000 paired component resamples; no coefficients are searched. It also records
the 25-row checkpoint topology, verifies the two dormant trailing output rows,
and reports the control/history appended input-projection diagnostic.

This is preparation only: the grader has not loaded any run checkpoints or
arrays, graded labels, or authorized execution. Root must first confirm the
four-fit outputs and current resource admission. After integration into the
canonical project root, the bounded CPU-only invocation is:

```sh
systemd-run --user --scope --slice=wnn-campaign.slice \
  --property=CPUQuota=100% \
  --property=MemoryMax=2000000000 \
  --property=MemorySwapMax=0 \
  /usr/bin/env \
    CUDA_VISIBLE_DEVICES= \
    WNN_PROJECT_ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack \
    PYTHONPATH=/home/vlad/development/wunder/wnn_connectome_starterpack/src \
    TMPDIR=/home/vlad/development/wunder/research_tmp/i1-warm-grade-20261022 \
    WNN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
    /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/grade_strong_parent_warm_i1.py \
    --output /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/GRADE_STRONG_PARENT_WARM_I1_R1.json
```

The command uses one CPU, a 2,000,000,000-byte memory limit, no swap and
physical-disk scratch; it masks GPUs. The report path is create-only. Keep the
fold-specific results separate: F0 and F1 change seed, parent, target cache,
teacher ancestry and propensity together, so they are not clean replications
and provide no training-seed uncertainty or public-score estimate.
