# [Experiment] Causal temporal-difference distillation

1. **Status and lineage.** Implements one PR55 card in the `sequence_*` raw112
   runtime; no receiver check or fit has run. It is not the TL3/event211 trainer.
2. **Mechanism.** Add causal teacher/student prediction-difference loss; compare
   against level KD with the same teacher, initialization and labels. Gradient
   scale is an alternate explanation and requires a separately budgeted control.
3. **Compute advantage.** Two causal Transformer teachers run on B300; four GRU
   fits compare student behavior, with no extra student inference operations.
4. **Frozen comparison.** Configs fix one train-source-only split, seeds 1001/1002,
   strict FP32, one B300, 5 fixed epochs, batch1, chunk256, accumulation4,
   Muon+AdamW for Transformer teachers, AdamW for GRU students, warmup5%,
   fit-only normalization and `is_scored` mask. Treatment adds
   lag1/weight0.1; control has weight0. Teacher-clipped and student-unclipped
   unweighted pair mean is separately normalized over each distributed update.
5. **Budget.** Two new teacher fits plus 2 controls and 2 treatments = six
   scientific fits, additional cache/score exports and engineering profiles.
   No ready matching teacher artifact is assumed. No automatic scheduler/retries.
6. **Validation.** Official pooled weighted Pearson with clip2/mask, t0/t1 and
   paired deltas. +0.0005 WP is a follow-up threshold, not significance. Full
   valid is development-exposed; two seeds in one split cannot quantify fold or
   training variance. Preserve matched component bootstrap as conditional evidence.
7. **Leakage.** `train_source_only` rejects valid-source fitting; independently
   verify whole-component exclusions, data identities and teacher cache ancestry.
   Feature and teacher inputs are causal; first k rows each chunk, warmup, padding
   and gaps are excluded. Component grouping does not prove actual chronology.
8. **Decision.** Improvement on both seeds with acceptable resource use motivates
   follow-up, not automatic submission. Mixed/small/negative results constrain
   this fixed lag/loss and need gradient-scale and other explanations considered.
9. **Run.** [README](https://github.com/VladKochetov007/WunderFunder/blob/experiment/temporal-kd-20261102/research/experiment_drafts/temporal-kd-20261102/README.md) contains exact receiver commands and six configs.
   Actual verified split, two new teacher fits and their caches, receiver numerical
   checks and separately reviewed real-device profiles are required. No Mac run.
10. **Evidence.** Fit/source/normalizer/cache identities, all paired outputs,
    pair coverage, losses and gradient norms, resource profiles, per-target grades
    and distinct evidence/weights archives; record failures and exclusions.
11. **Results.** Unrun; no model, CPU serving, leaderboard or quality result. Prior
    TL3 0.692 is a different recipe/data pipeline and is not this baseline.
