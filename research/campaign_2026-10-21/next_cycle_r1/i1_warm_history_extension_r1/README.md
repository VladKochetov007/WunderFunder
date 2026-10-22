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

[`PREFLIGHT_R2.json`](PREFLIGHT_R2.json) and its create-only helper prepare those actual-parent and paired-initialization gates. The helper has not been run in this worktree. Root reports that the F0 warm profile completed 1,100 finite updates with no skips; this is throughput evidence only. The F1 profile stopped before updates because its actual cache metadata is 842,417 bytes while the frozen canonical reader allows 512,000. Preserve that failure and the frozen R1 source/plan hashes: F1 remains blocked until the narrow metadata-bound repair is integrated and a separate source/plan amendment is frozen. No metadata validation is disabled.

The frozen R1 cost projection is 7.92h using a 900-second ceiling for each of two profiles, 15% margin on four 1,492-batch fits, and explicit export/grade reserves. It predates the measured F0 profile and is not an admission decision; recompute the complete-panel cost from observed phases before admission. Preserve all four fits and full-fold predictions if the panel is admitted. No quality fit, actual-parent parity preflight, or quality grade has been run from this packet.
