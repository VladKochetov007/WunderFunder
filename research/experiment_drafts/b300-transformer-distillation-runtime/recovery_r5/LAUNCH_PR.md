# R5 B300 launch handoff and post-selection checkpoint retention

Keeping 30 FP32 snapshots for all four 1.54B fits would occupy roughly 740 GB
before optimizer states or predictions. This follow-up to merged #42 runs one
LR pair through both complete epoch curves and the frozen shared-epoch selector,
exports MIDAS evidence, and provides an explicit hash-bound cleanup action before
the next pair. Selected weights, fixed-final weights, resume state and evidence
are retained. No experiment result or B300 readiness is claimed.

1. **Status, lineage and owner.** Engineering preparation for
   H/E-B300-TRANSFORMER-RECOVERY-R5, based on research-team merge `7e407431`.
   Teacher execution/retention only; receiver owns admission and execution.
2. **Mechanism.** Reduce simultaneous stored snapshots by finishing and archiving
   one LR pair before the next. This changes storage order, not model quality.
   Keeping per-fold best alone is insufficient for a shared mean-WP winner.
3. **Compute advantage.** Existing eight-B300 fits and full inner sweeps remain.
   Hash verification adds disk I/O. Weight-only peak estimate is ~407 GB, with
   ~49 GB selected/final weights after both pairs; this is not total disk capacity.
4. **Frozen comparison.** No recipe/split/metric/precision change: 30 epochs,
   context1024, Muon+AdamW, BF16/BF16x9 with FP32 masters/state. Every epoch is
   retained until both full curves and the common epoch decision are verified.
5. **Fitting panel.** Four already planned teacher fits, two per LR1e-4/3e-4 on
   F0/F2 with seeds201001/201002. Zero additional scientific fits, student fits,
   or reused controls. No local or remote fits started by this change.
6. **Validation/uncertainty.** Same official inner WP, equal-fold mean and earlier
   exact-tie rule. All curves retained; two folds do not establish significance.
   Storage estimates come from parameter counts, not receiver measurements.
7. **Leakage/coverage.** Existing component isolation and fitting-only transforms
   remain. Exposed cross-fold development limitation is unchanged. Retention
   refuses incomplete selection, changed evidence, unsafe paths and wrong hashes.
8. **Decision.** Receiver checks must demonstrate exact survivor/deletion sets,
   interruption recovery and fail-closed archival guards before real cleanup.
   A failure blocks retention/next-pair launch; it does not reject Transformers.
9. **Run.** See [LAUNCH_AND_STORAGE.md](https://github.com/VladKochetov007/WunderFunder/blob/codex/b300-r5-launch-storage/research/experiment_drafts/b300-transformer-distillation-runtime/recovery_r5/LAUNCH_AND_STORAGE.md) and
   `receiver.env.example`. Explicit prepare/check, fit-cell, finish-pair,
   cleanup-pair and verify-pair actions; existing pinned runtime, no dependency
   install or automatic queue. Storage admission remains conservative.
10. **Evidence.** Keep full config/source/split identities, all scores/predictions,
    selected + final weights, optimizer state, original indexes, MIDAS bundles,
    retention plans and durable intent/completion receipts. Export before cleanup;
    references to removed checkpoints are historical and not availability claims.
11. **Results/trace.** Source preparation only. Local tests, training, inference
    and GPU profiles were not run. B300 remains awaiting restart. Receiver tests
    are added but unexecuted locally. Unified ideas/operations index and append-only
    H/E ledgers link the handoff. Ruff, shell syntax, AST/JSON and source structure
    checks pass; these do not establish receiver correctness or performance.
    No measured quality, storage or throughput gain.

After a retention intent, ordinary fit/resume and new MIDAS export are blocked;
full-curve re-evaluation also needs the now-removed snapshots. The existing
single indexed inner-inference path can still validate the retained selected
weight. KD/submission admission remains a separate continuation.
