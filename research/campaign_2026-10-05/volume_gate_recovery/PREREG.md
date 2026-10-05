# E427 Oct 5 volume-context recovery preregistration

## Question and decision rule

Does the fixed causal `volume28` context gate improve the deployment-relevant global weighted-prediction score when its context is aligned to each sample, relative to identical stale-context and zero-context controls? The preregistered directional contrasts are aligned minus stale and aligned minus no-op. This is a three-arm mechanistic continuation, not a broad hyperparameter search. Report every arm and every panel. Do not select a member, arm, metric, checkpoint, epoch, or panel after observing scores. No single score is treated as evidence of untouched generalization.

## Frozen scientific recipe

Use the existing E427 implementation, trainer, FP32 exporter, scorer, continuation recipe, and guards byte-for-byte as represented by the input paths in the fresh draft specs. The parent is `c20261004_combo_s3_final_r1@ep3`; the E427 fit-only context statistics artifact is `research/campaign_2026-10-04/checks/e427_context_stats_s3_v1.json`. Keep the exact aligned, stale, and no-op arms; seed 3; one epoch; batch 32; chunk 1000; hidden width 256; two GRU layers; no dropout; Muon on RNN parameters; EMA teacher 0.999; the declared since/distillation objectives; fixed epoch-0 checkpoint with fit-time evaluation withheld; and FP16 training. Do not add epochs, retry, or alter precision. The fixed FP32 export covers ordered VALID group IDs 0–1872 and writes `[1873, 20000, 2]` predictions.

Fresh Oct 5 runner attempt IDs are used for every stage under `research/campaign_2026-10-05/volume_gate_recovery/runner`. The original prepared model-run names are preserved because the existing scorer binds to them; all three fit-output directories and all three prediction paths were absent when this packet was checked. Recheck every declared artifact path is absent immediately before sealing/launch. Any collision blocks admission; do not overwrite or rename the outputs without a separately reviewed scorer-compatible packet.

The score is `global_wp_clip2_v1`; fixed comparisons are aligned-minus-stale and aligned-minus-noop, 2,000 component bootstrap replicates, seed 271828. Preserve all three panels: fold0 primary 468 groups (374 fit-overlap, 94 not-fit); the 368 stats-excluded historically exposed groups (zero fit-overlap); and the 94-group nonfit diagnostic subset. The 368 were excluded from these continuation fits and the parent final fit, but the teacher lineage and prior campaign exposure mean this is not an untouched holdout. The 94-group result is diagnostic and uncertain.

## Staging and stop behavior

The intended order is all three production-shape smoke checks, all three fixed one-epoch fits serially, all three fixed FP32 exports, then the fixed three-panel score. Preserve all stages in the stage inventory and reserve their full fixed timeouts: smoke 900 seconds/arm, fit 3,600 seconds/arm, export 1,800 seconds/arm, and CPU score 1,800 seconds. Total GPU timeout reserve is 18,900 seconds (5.25 hours), plus 0.5 CPU-hours. Before each new stage, require exact successful receipts for its dependencies, exact artifact hashes, verified teardown, explicit zero final OOM counters, and a fresh host/GPU admission receipt. Do not lower the original 24 GiB worker reservation or 4.2/4.8 GB allocator/stop limits.

At 2026-10-05 07:00 UTC, admit no new stage. Let any already-admitted runner-owned worker finish or reach its original sealed timeout and verify teardown; do not kill it. Keep unstarted stages listed as pending and retain their full original reservation in accounting. A stopped sequence is incomplete, not a negative scientific result.

## E464 dependency

Before any E427 stage, validate all 17 E464 fit/export/score attempt receipts in exact order, their terminal success and clean OOM evidence, their declared spec/code hashes, each exact artifact set and artifact bytes, the E464 completion record, and its paired score report hash. Check current fit-statistics, parent checkpoint/provenance, and all declared input identities against the existing receipts. Hash declared inputs before issuing `POSIX_FADV_DONTNEED` on read-only descriptors opened only for those declared paths; then collect a fresh cgroup/GPU snapshot and run the existing resource admission checks. Cache advice is advisory and never overrides measured memory or relaxes a guard.

## Admission status at packet construction

This packet is draft-only. The immutable original worker reservation is 25,769,803,776 bytes (24 GiB); the existing support reserve is 4,294,967,296 bytes (4 GiB). Together they require 30,064,771,072 bytes, exceeding the 30,000,000,000-byte hard parent limit by 64,771,072 bytes before accounting for any current usage. Thus no E427 stage is currently admissible under the unchanged limits. Root must obtain a truthful fresh resource admission after the declared-input cache advice, or choose a scientifically justified new placement/resource design with its own review. This packet does not reduce reservations, start a worker, seal a spec, or claim a fit result.
