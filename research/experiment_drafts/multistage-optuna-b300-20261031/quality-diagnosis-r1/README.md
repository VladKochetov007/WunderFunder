# Quality diagnosis after the completed B300 wave

1. **Status, lineage and owner.** Root analysis after PR53's completed wave.
   This note proposes follow-ups; it starts no fit, profile, SSH session or
   resource allocation. Existing recipes, endpoints and failures remain frozen.
   [Machine-readable evidence and proposed cells](EVIDENCE_AND_PROPOSED_PANEL.json)
   binds the actual full-primary report and the historical PR52 report by hash.
   Read [the completed wave](../PROGRESS_20261031_FINAL.md) first.

2. **Why investigate this mechanism.** The new raw112 GRU384x2 scores0.682425941;
   the old single fin256_s3 scores0.688259667 and tiedlead_s5 scores0.691415279.
   These are different complete recipes, not an isolated architecture comparison.
   New GRU has3,280 actual optimizer updates in four epochs. Legacy batch32
   chunk2000/1000 recipes count12,880/25,760 updates in four epochs. Larger batch,
   shorter gradient history, inputs, normalization, objectives, averaging and
   ancestors all differ. No one difference is a confirmed cause of the gap.

   New GRU's final epoch improves WP by0.003309448; Transformer's by0.003920498;
   LSTM's by0.000238775. The first two have no observed final-epoch plateau.
   These curves do not establish how much another epoch or a different schedule
   would gain. The implementation detaches state after every microchunk:
   four accumulated256-row chunks do not create1024-row gradient propagation.

3. **Compute advantage.** Spend additional compute on optimizer updates and
   gradient history before another broad architecture grid. Historical fit
   durations do not determine changed-batch/long-context throughput. All new
   durations and effect sizes remain unknown until measured profiles/results.
   Parallelism can preserve independent small-batch fits instead of increasing
   each fit's effective batch simply to occupy memory. EMA, auxiliary heads,
   event/tiedlead inputs and correlation fine-tuning from TL3 remain subsequent
   separate comparisons; do not silently add them to this panel.

4. **Proposed frozen comparison.** Same Source4 GRU384x2, raw112, prefix32
   fitting-only normalization, WMSE, AdamW lr0.001/wd0.01/clip1, strict FP32,
   legacy warmup/cosine5%, four fixed epochs, train10,325/valid0 ancestry.

   | Arm | Sequence batch | TBPTT rows | Accumulation | Updates in4 epochs |
   |---|---:|---:|---:|---:|
   | A: existing large-batch recipe |256|256|4|3,280|
   | B: smaller batch/more updates |64|256|4|12,960|
   | C: longer gradient history |64|1024|1|12,960|

   A/B estimates the joint batch/update-frequency change at fixed data exposure;
   it cannot attribute an effect to update count alone. B/C shares batch/order,
   optimizer steps and1024-row update boundaries, while changing the detach
   horizon. Raw input normalization, targets and precision remain fixed.

5. **Bounded fitting panel.** Seeds1301/1302 for each arm: six endpoints, five
   proposed new fits. Reuse exact completed A/1301 only after matching its model,
   data, source, normalizer, epoch and artifact identities. A/1302 is new; both
   B and C seeds are new. No new teacher fits. New fitting budget admitted by
   this analysis is zero. Do not reproduce A/1301 merely to fill a GPU slot.

6. **Validation and uncertainty.** Primary evaluation remains the complete
   official later valid boundary, fixed is_scored mask and pooled two-target
   clipped WP. Save each paired delta and all trajectories, but report configured
   final epoch4 for this panel. Two matched fitting seeds do not establish
   significance. Existing valid is development-exposed. Additional inner2065
   controls are nonchronological and unscored train proxies, never a substitute
   for primary validation or an inferred timestamp split.

7. **Leakage and coverage.** No fitting, normalizer, teacher, initializer or
   learned mixture weight ancestry may contain primary valid. Preserve all
   sequences and the existing split SHA. seq_ix does not order time. If a later
   TL3 teacher is reused, its full-train membership cannot be called clean for
   selecting weights on an inner subset of that same train. It remains eligible
   for full-train distillation followed by official-valid reporting.

8. **Decision-changing evidence and cheaper first action.** Before new fitting,
   evaluate the six declared mixtures `(1-alpha)*TL3 + alpha*candidate`, with
   alpha0.05/0.10 and the three fresh full-primary GRU/LSTM/Transformer endpoints.
   Freeze this vector list before scoring, align actual raw prediction hashes,
   memberships, masks and target order, and blend before final clipping. Keep
   TL3 and historical90%TL3+10%GALAM as references. No Optuna weight learning on
   primary valid, and no claim these unmeasured mixtures already improve it.

   Historical10%GALAM improves TL3 by0.000119022; historical10%LSTM improves by
   0.000040966, while20%heterogeneous students loses0.000117898. A weak singleton
   can help, and a stronger student can add nothing to TL3. Target the complete
   blend contribution, not prediction disagreement or model count alone.

   For A/B, improvement in both matched cells supports deepening the joint
   optimization direction. For B/C, improvement supports long-gradient history.
   Mixed outcomes require per-target/regime and cost analysis. A zero-crossing
   conditional interval alone does not close a mechanism. Set practical utility
   against the actual extra fit/serving cost before admission; do not invent a
   numerical effect forecast here. Screen configurations further only when the
   comparison resolves an actionable uncertainty.

9. **How to run / missing work.** Exact receiver configs are enumerated in the
   JSON; they are proposals, not runnable admission receipts. Reuse the existing
   Source4 adapter after verifying its source capsule and current available node.
   Actual-device profiles, source/normalizer/control reuse checks, receiver paths,
   bounded launch argv, fitted-model export and final collection contracts remain
   preparation dependencies. No changed-batch wall-time or memory measurement
   exists. Full-primary and TL3 prediction arrays are not local in this worktree;
   the no-fit mixtures still need a qualified receiving analysis job.

10. **Evidence to return.** Every epoch's train diagnostic/valid loss and WP,
    per-target metrics, LR, successful optimizer updates, processed rows, preclip
    gradients, clipping/nonfinite rates, phase timings, RAM/VRAM and throughput.
    Preserve mask/cohort differences on graphs. Return configured-final weights,
    prediction/order receipts, exact source/config/normalizer hashes and all
    failed/censored cells. Native train diagnostic uses16 fitting sequences/all
    rows; it is not the same distribution or mask as official valid.

11. **Results and trace.** No follow-up result or launch exists. The evidence
    JSON stores exact old results, learning-curve deltas and step calculations;
    proposed gains/durations are null. This is a planning addendum to PR53,
    not a rewrite of that completed contract or permission to resume stopped fits.
    After this screen, prioritize transferring TL3's stronger event/tiedlead,
    EMA/auxiliary and WMSE-to-correlation recipe into the winning runtime, then
    clean TL3-teacher distillation. Test them separately with matched controls.
    The existing tiny EMA/ridge panel was component-sensitive; it is not a
    validated0.655 full-primary solution or grounds for a large new grid.
