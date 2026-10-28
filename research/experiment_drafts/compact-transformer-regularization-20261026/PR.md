# Compact Transformer regularization: four paired five-epoch B300 fits

1. **Status/lineage/owner.** E-COMPACT-REG-20261026; teacher branch after PR32/42,
   SemyonEpanov. Four fresh fits authorized; all four full fits started under native cron after measured profiles/admission. No reused trained controls.
2. **Mechanism.** Test whether a joint dropout0.1/WD0.1 package improves inner WP
   over dropout0/WD0.01 at fixed compact architecture. A smaller train/val gap alone
   is not success; undertraining is an alternative explanation.
3. **Compute advantage.** Four independent two-B300 groups provide matched F0/F2
   evidence concurrently. Batch is selected from measured cohort throughput with
   bounded own CUDA allocators. Shared utilization is not attributed to these fits.
4. **Frozen recipe.** Transformer768x16, 12heads, FF4, causal context1024, raw112,
   chunk512/accumulate8, Muon/AdamW, peakLR1e-4, five-epoch warmup5%/cosinefloor0.1,
   clip1, BF16/BF16x9 with FP32 masters/states. Selected batch128/rank, global256,1150updates per fit.
5. **Panel.** Two arms x F0/201001 and F2/201002 = four new fits, zero student fits.
   Both arms share one selected batch and complete epoch budget. No automatic expansion.
6. **Validation.** Whole-component inner holdouts derived from each outer fitting
   pool; every epoch retained and evaluated. Fixed64-sequence fit probe plus full
   inner WP/WMSE. Inner epoch selection is tuning, not independent confirmation.
7. **Leakage.** Canonical source-qualified dependency grouping, fitting-only
   normalization, causal inputs/state resets, no teacher ancestors or label inputs.
   Outer folds were previously exposed. Joint regularization factors are not isolated.
8. **Decision.** Compare paired inner-WP deltas at a common selected epoch and
   both target scores; mixed signs remain inconclusive. No transfer/public gain inferred.
9. **Execution.** Exact source/profile/config/split/wrapper binding, native-cron
   one-shot owner, per-job timeouts, immutable epoch snapshots, external live evaluator.
   Co-residency with ARC-AGI is explicitly authorized; no foreign process cleanup.
10. **Evidence.** Source commit0a70b4ab96c1810484544f382df4bcf0c08e8fd4,
    receiver55 CPU tests, raw engineering profiles, plans, time/memory estimates,
    rank/epoch/hardware logs, safe weights and validation artifacts with hashes.
11. **Results.** Full fits are running; first predictive quality is pending epoch evaluation. Failed dropout0-only
    startup profiles remain engineering failures, not negative scientific results.
    Runtime estimates and progress will be updated from actual receipts.
