# [Experiment] Source-traced GALAM with chronological primary validation

1. **Status, lineage and owner.** Controller implementation packet; H-GALAM-20261027 /
   E-GALAM-20261027. Base19383f2c, branch experiment/galam-reconstruction-20261027.
   Visible GALAM graph and canonical sequence adapter implemented. CPU correctness
   checked; zero scientific fits. Exact reference identity remains unverified.

2. **Why it should work.** A compact recurrent memory and attention/ConvMixer
   memory can expose complementary temporal information. The author reports gains
   for the historical next-state task. Testable hypothesis: the112→2 adaptation
   improves the chronological endpoint or its blend contribution. Extra parameter
   count and changed training recipe are alternative explanations.

3. **Compute advantage.** No speed or scaling claim. Streaming state is bounded,
   but per-step LSTM/attention/ConvMixer and BN make this correctness-first code.
   Profile actual B300 shapes before assigning GPUs; compare against equal-compute
   compact Transformer/recurrent baselines. No GPU-hour estimate is fabricated.

4. **Frozen comparison.** Prospective closest-schematic adaptation versus a
   source-clean baseline on the same earlier train and full later valid. Same
   fitting seeds, update budget, normalization and official scoring. The supplied
   configs are examples, not a frozen historical recipe. No original weights,
   training schedule, SWA recipe or FiLM ensemble were found.

5. **Bounded fitting panel.** Suggested future initial screen: two matched fitting
   seeds1001/1002, one control and one adaptation each, four total new fits after
   explicit execution scope. One fixed chronological boundary, not two random
   folds. No teacher/student fits, cached-control reuse or GPU reservations are
   admitted in this change. Later SWA/blend arms need their own declared budget.

6. **Validation and uncertainty.** Full official valid and unchanged global
   clip2 WP; report each target, paired seed deltas and train/valid curves.
   Component resampling describes evaluation sensitivity, not independent fits.
   Previously examined valid remains exposed. Two seeds do not establish reliable
   statistical significance. Preserve all outcomes and parameter choices.

7. **Leakage and coverage.** Chronological manifest pins sources and producer
   indexes, excludes fit components linked to valid and forbids fit-valid rows.
   State resets by sequence; inputs are current/past only. BN is per-time across
   batch, never across future rows. Reject singleton and padded-DDP BN batches.
   Teacher/initializer/preprocessor ancestry must exclude primary valid. Shuffled
   seq_ix cannot define temporal order. Source-boundary chronology is the adopted
   policy; actual calendar timestamps are unavailable.

8. **Decision-changing result.** Promote only a complete independently evaluated
   endpoint that improves the clean baseline or a frozen blend within serving
   constraints. A negative result narrows this declared adaptation, not the whole
   GALAM family. Missing exact-reference evidence prevents a replication claim
   regardless of the measured score.

9. **How to run.** README gives runnable CPU inspector/check commands and the
   implemented receiving-host manifest producer. The canonical train_sequence
   adapter is wired, but no dataset pins, compatible actual BN batch plan,
   B300 profile or fit is admitted. Receiver paths are unresolved and labelled.
   No second trainer, inference service or queue manager is introduced.

10. **Evidence to return.** Source/config/data/split hashes, environment,
    preprocessing and initialization ancestry; per-epoch train and valid metrics,
    LR/gradients/update counts, state/reset checks, time/RAM/VRAM; all checkpoints
    and fixed predictions with exact scorer identities. Evidence and weights in
    separate archives. Current receipt contains only local CPU engineering checks.

11. **Risks and next action.** Dimensions, history, exact fusion operands, BN
    convention, GeGLU-text/ConvMixer-image discrepancy and FiLM remain unknown.
    Independent review corrected graph ordering, LayerNorm and DDP BN admission.
    Next: review the delivered source and receiving-host membership/profile under
    separate compute scope. Do not label this package exact, trained or scored.
