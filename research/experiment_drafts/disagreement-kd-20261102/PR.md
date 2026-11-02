# [Experiment] Three-teacher disagreement-weighted distillation

1. **Status and lineage.** Implementation prepared, no receiver acceptance,
   numerical checks, training, inference, profiles or local tests. Raw112 sequence
   runtime from `research-team` `7e4074314a62d0907c9608f9220cf72958708a2d`, with
   source-holdout loader from [PR47](https://github.com/VladKochetov007/WunderFunder/pull/47)
   head `f0d8da40fb545a630e5f042bfcc3b3360784177e`. This is a new student/connection
   study based on PR55 card 2, not TL3/event211 or an asserted 0.692 baseline.
2. **Mechanism.** Rows where three teachers disagree may be especially informative
   or unreliable. Compare fixed mean KD against higher and lower bounded KD weights
   from the identical teachers; both directions can fail. Measure high-disagreement
   slices separately without tuning on exposed official valid.
3. **Compute advantage.** B300 prepares three independently seeded teachers and
   complete cached predictions; one compact GRU runs at inference. Student loss
   weighting is training-only. Three cache reads and per-epoch integrity checks
   cost additional I/O; no measured throughput or public-score gain is claimed.
4. **Frozen comparison.** constant/up/down × seeds1001/1002; identical raw112,
   strict FP32 GRU256x2, AdamW .001, 5 epochs, batch32/rank, W1, chunk512,
   accumulation8, 5% warmup and cosine floor .1. Constant/up/down differ only in
   per-row KD weighting; the corresponding target-weighted mass is globally
   reduced per accumulation window. Fixed final epoch, no early stopping.
   Teacher mean is unclipped until the objective clips it. q=var/(var+.01),
   weights=1, 1+.5q or 1-.5q. No target/future in q.
5. **Fitting budget.** Six new students (2 per mode), plus three separately
   budgeted teacher prerequisite fits and three full-fit teacher prediction
   exports if exact same-family/split/source/normalizer caches cannot be reused.
   Total in that case: nine fits; no existing teacher is presumed available.
   Profiles are engineering, not scientific fits; no queue or auto launch.
6. **Validation.** Source-only fit, full valid report on unchanged pooled clipped
   weighted Pearson per target and mean. Paired seed deltas, subgroup behavior,
   exact score mask and all outcomes; valid is exposed development evidence.
   Two seeds on one split do not estimate independent fold or teacher variance.
7. **Leakage.** Complete source-qualified component graph, train-only normalizers,
   all-valid-source excluded from teacher/student fit and ancestry, no future
   features or targets in weights; same row group ID/sequence ordering/cache hash
   for all teachers. Source separation does not prove hidden temporal order.
8. **Decision.** Candidate mean gain ≥0.0005 WP against constant, with both
   paired deltas reported and CPU-serving costs measured separately, motivates
   a preregistered robustness follow-up. Mixed/small gains remain uncertain;
   negative results constrain only this weighting formula. A zero-crossing
   interval alone does not rule out the mechanism.
9. **Run.** [README](https://github.com/VladKochetov007/WunderFunder/blob/experiment/disagreement-kd-20261102/research/experiment_drafts/disagreement-kd-20261102/README.md) provides the receiver-only split, teacher,
   descriptor, profile, fit and scoring commands. All real data, cache and
   numerical acceptance checks, B300 profiles and ownership are pending.
10. **Evidence.** Preserve source/config/teacher ancestry and split hashes,
    per-rank gradients/losses/denominator and timing metrics, all failure receipts,
    GPU memory/time, cache I/O, fixed checkpoint and prediction hashes, separate
    evidence/weights archives and complete fixed comparisons.
11. **Result and trace.** Zero fits; code is not production accepted. Three
    same-family teacher caches and identity/profile approvals are prerequisites.
    No local or hidden-test/public-score claim, and no inference equivalence claim.
