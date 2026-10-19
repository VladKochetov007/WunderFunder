# [Experiment] large-dynamics-teacher-b300-r2: event dynamics, teacher scale and GRU transfer

1. **Status, lineage and owner.** Proposed teacher-scale recipe; root owns design,
   receiving B300 implementer owns source binding and execution. H-LARGE-DYNAMICS-TEACHER-20261019-R2;
   E-LARGE-DYNAMICS-TEACHER-SCALE-20261019-R2. Builds on the earlier AR/grouped-input
   proposals; not a claim that their trainers or fits completed. This draft
   supplies a recipe and analytical calculator, not a new executable DDP trainer.
2. **Why it should work.** Latest user source [TradeFM](https://arxiv.org/abs/2602.23784) motivates an additional, separately budgeted [event-objective contrast](TRADEFM_ADAPTATION.md): predict observable-change hazards and marks while retaining supervised per-row targets. This is the preferred first dynamics screen; it does not import proprietary data, real-unit feature assumptions or a market simulator.

 A larger causal teacher may represent conditional
   cross-feature dynamics that a compact recurrent teacher misses. Compare0.68B
   and 1.65B capacity recipes at the same context/data/row budget. Alternatives:
   repeated-data overfitting, bad optimization or capacity that cannot transfer.
   Grouped numeric encoding, auxiliary future-observation targets and full-history
   attention are separately budgeted descendants rather than a confounded grid.
3. **Compute advantage.** Eight B300s support replicated large teachers and
   longer contextual training. Profile8-way versus two 4-way jobs at the same
   global batch. DDP does not pool memory. First flagship 1.65B,0.68B reference,
  3.17B eligible extension. No local score, scaling law or B300 speed is promised.
4. **Frozen comparison.** The proposed two arms use 24×1536/FF4096 and 32×2048/FF5632
   dense causal trunks,64-dimensional heads, raw 112+canonical event 99 inputs,
   strict FP32 with autocast/TF32 off. Start4096-row windows/2048 new-row cores,
   nominal 65,536 new rows/update,12-epoch cooled endpoint. FP32 Muon match_rms_adamw
   hidden-matrix groups; AdamW input/readout/norm groups; peak base LR1e-4 versus
  7.5e-5, momentum 0.95,5 NS iterations, gradient clip 1.0, declared decay groups.
   The scale-specific LR makes this a capacity-RECIPE comparison. WSD uses the
   same 1500-update initial warmup, stable prefix, final 20% cosine cooldown to 0.1×.
   Exact component membership, normalized objective, shapes and source are still
   unbound; launch_ready=false. AR/auxiliary/long-memory are off in this first pair.
5. **Bounded fitting panel.** Each panel has four total new teacher fits. The new transition_recipe.json compares auxiliary likelihood off/on at the same0.68B scale in two cells. The original capacity recipe below is a SEPARATE four-fit contrast; running both costs eight teacher fits before students. Neither panel is automatically funded. Capacity panel: two arms each at
   F0/seed 1001 and F2/seed 1002, subject to ownership and frozen split binding.
   No mandatory CP30. No cached controls or student fits are counted here.
   Student transfer needs its own 4–6-fit inventory with teacher caches named.
   Expanded panels require a dated allocation; engineering profiles are separate
   charged work and not scientific replications.
6. **Validation and uncertainty.** Unchanged global_wp_clip2_v1 on the fixed
   official mask and source-qualified held-out components. Report both paired
   deltas and their descriptive mean, conditional component sensitivity and
   by-target/activity/sequence-age diagnostics. Two cells do not identify seed
   versus fold variability or establish significance. All folds remain exposed;
   no public-score forecast. Retain all failures and selection history.
7. **Leakage and coverage.** Teachers, initializers, normalizers and auxiliary
   transforms exclude evaluated components. No lagged true labels, full-sequence
   normalization or future-containing patch summaries. Later slow tokens become
   visible only after their block completes. Require numerical causality/reset,
   weighted-DDP and producer-consumer checks before fitting. A production all-data
   teacher is separate from held-out evidence.
8. **Decision-changing result.** A useful teacher gain with compatible regimes
   justifies matched GRU distillation; smaller or mixed effects may justify
   mechanisms, optimizer/context repairs or teacher-mixture tests. Negative results
   narrow the tested capacity recipe. A zero-crossing interval alone does not
   close scale. Final priority is a stronger GRU or complete served blend, not
   teacher parameter count. Student transfer remains untested.
9. **How to run.** Implemented commands are only the stdlib capacity calculator
   and plan expander listed in README. No torchrun command is advertised as
   runnable. The receiving implementer must add canonical-component model/DDP
   integration, source-bound folds/data, an audited FP32 Muon path, backend/dtype
   checks and actual-shape profiles; then record exact torchrun arguments and
  8-card reservation. Current source Muon casts NS work to BF16 and needs adaptation.
10. **Evidence to return.** Immutable source/config/env/split/ancestry, exact
    commands and actual devices; attempted/successful steps, loss components,
    LR, preclip norms/clipping, finite/update checks, processed/new-row throughput,
    memory, checkpoint/resume identities and official predictions/grades.
    Separate checksummed evidence/weights archives in private storage, no raw
    dataset upload. Include teacher export and student fitting costs in total ETA.
11. **Results and trace.** No new fit or empirical improvement. The unified
    [design](../../unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md),
    [sources](../../unified/sources/TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md)
    and [independent review](../../unified/reviews/LARGE_TEACHER_DESIGN_REVIEW_20261019_R2.md)
    preserve the reasoning and remaining checks. Older stopped B300 GRU evidence
    is not Transformer evidence. Publication/source/base links must name actual
    commits after this draft is created. No model launch or competition upload.

User-paper addendum: [TradeFM adaptation](TRADEFM_ADAPTATION.md), [transition recipe](transition_recipe.json), and [explicit four-fit plan](TRANSITION_PLAN.json). Paper generation quality is not evidence for our official WP or student quality. Strict-FP32 Muon remains a proposed adaptation, not the source paper recipe.
