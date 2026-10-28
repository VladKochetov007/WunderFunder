# [Experiment] Block GALAM-GRU72 and frozen BatchNorm diagnostic

[GPU commands and packet](https://github.com/VladKochetov007/WunderFunder/blob/codex/galam-block-bn-diagnostic/research/experiment_drafts/galam-block-bn-20261029/README.md).
[Incremental changes above dependency PR46](https://github.com/VladKochetov007/WunderFunder/compare/8482de5c71ce8bdd9974780438d518ea50bfeca8...codex/galam-block-bn-diagnostic).
Implementation commit: `980533a97f5736ec0ae4aaab15e20cf607f91f67`.

1. **Status, lineage and owner.** Implementation preparation; zero receiver runs.
   Stacked on PR46 at `8482de5c71ce8bdd9974780438d518ea50bfeca8`, targeting
   `research-team`. H/E-GALAM-BLOCK-20261029 and H/E-GALAM-BN-20261029-R1.
   The user requested code and GPU commands, preserving all active training jobs.
   The existing step backend remains the default. No deployment or fit is launched.
2. **Mechanism.** The existing GRU adaptation dispatches its entire network once
   per row. A sequence GRU call, chunk attention, causal convolutions and pointwise
   decoder remove repeated dispatch while preserving the causal graph. Separately,
   per-step BatchNorm EMA emphasizes the training tail; train-only recalibration
   tests whether its saved inference statistics hurt a frozen checkpoint.
   Kernel dispatch and BN mismatch are separate hypotheses, not established causes
   of the complete epoch time or quality plateau.
3. **Compute value.** Measure complete forward/backward/AdamW updates before
   spending on more epochs. The existing inference trace is motivation, not a
   training-speed prediction. BN uses frozen weights and three typical layerwise
   passes over 256 train sequences, then paired full-valid evaluation.
4. **Frozen comparison.** GRU72, raw112, history64, chunk64, B384, AdamW1e-4,
   beta0.9/0.999, WD0.01, clip1, FP32 and TF32/autocast/BF16x9 off. The two
   copied configurations differ only by the explicit block selector. Per-time BN
   reductions never mix timestamps; recurrent/attention/conv state and TBPTT
   boundaries are preserved. Sampled monitoring uses the original step path.
   BN diagnostic uses the step backend and an immutable indexed checkpoint;
   only running means/variances/counters change, with no optimizer or gradient.
5. **Bounded work.** Zero new scientific fits, zero teacher/student fits, zero
   reused fitted controls counted as replications. This is engineering plus a
   frozen-checkpoint diagnostic, not a 4–6-fit scientific screen. Synthetic
   speed comparison: four warmup and twelve measured updates per backend.
   Optional canonical real-data profiles: 25 updates per backend, including
   sampled monitoring. One checkpoint, one fixed calibration seed/subset, one
   original-versus-recalibrated comparison; no automatic sweep or retraining.
6. **Validation.** Receiver checks compare outputs, state, gradients, BN buffers,
   streaming continuation and causal prefixes. Profile receipts bind source,
   configuration and runtime. BN scores pooled `global_wp_clip2_v1` on complete
   official valid, warmup99 and `is_scored`, reporting both targets and deltas.
   Valid is exposed and checkpoints may have been selected there. No pristine
   holdout, seed robustness, significance or public-score forecast is claimed.
7. **Leakage.** BN membership comes only from the original checkpoint-qualified
   train split. No valid examples or labels enter recalibration. Normalizer,
   source, checkpoint/index and split identities are checked. Source separation
   and approximate feature-linked exclusions do not prove calendar chronology.
   Outputs are new artifacts and cannot overwrite or resume the live checkpoint.
8. **Decision.** Numerical acceptance precedes speed measurement. Threefold
   update speedup is a target, not a result; report wall time and memory even if
   slower. Full-epoch benefit requires subsequent measurement. BN delta at least
   0.001 WP motivates a separately frozen checkpoint/seed confirmation; mixed or
   negative results remain evidence about this calibration method and checkpoint.
9. **Run.** README supplies exact receiver commands for acceptance, synthetic
   update timing, canonical real-data profiles and BN recalibration/evaluation.
   Use a separate clean checkout, one explicitly selected free CUDA GPU and fresh
   output directories. Runtime acceptance and memory measurements are pending;
   there is no silent batch, precision or tolerance fallback.
10. **Evidence.** Return source/config/runtime/device identities, numerical errors,
    per-backend timing/memory, original profile receipts, calibration membership,
    original/candidate checkpoint hashes, before/after BN statistics and both
    target scores. Keep checkpoint and evidence archives separate. Failed or
    incomplete receiver runs remain failures, not quality results.
11. **Trace.** Packet README, recipe, static validation and independent source
    review are linked from unified ideas and append-only H/E ledgers. No local
    tests, training, inference, model imports or GPU profiling ran. No active
    worker, recipe, checkpoint, dashboard or launcher was changed.
