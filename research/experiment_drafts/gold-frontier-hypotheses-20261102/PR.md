# [Research] Gold frontier: objective, state, teacher and blend hypotheses

1. **Status and lineage.** Five proposal-only cards in [HYPOTHESES.md](https://github.com/VladKochetov007/WunderFunder/blob/codex/gold-hypotheses-20261102/research/experiment_drafts/gold-frontier-hypotheses-20261102/HYPOTHESES.md),
   based on `research-team` at `7e4074314a62d0907c9608f9220cf72958708a2d`.
   One user-requested portfolio PR; none are implemented, profiled, trained or
   graded. No existing jobs are modified.
2. **Why these could work.** Test filtered residuals, target-gradient interference,
   recurrent-state mismatch, a parallelizable causal convolutional teacher, and
   selection overfitting in mixture weights. Each card states an alternative and
   a falsifying observation. PCGrad is a continuation of the team's R6 diagnostic,
   not a newly discovered idea. Related previous frequency, TBPTT and OOF work
   are linked in the packet.
3. **Where compute helps.** B300 capacity buys paired fits, causal clean teachers
   and cross-fit exports; it does not by itself prove a transfer or latency gain.
   D requires separately budgeted student transfer. E can reuse fully qualified
   OOF predictions; otherwise all producer fits/exports cost extra.
4. **Comparison.** Each A–D card states its fixed control, changed mechanism and
   required matched variables. Exact architecture, optimizer/schedule, membership,
   checkpoint rule and coefficients are intentionally not yet frozen; these are
   blockers for implementation/launch, not silently assumed defaults.
5. **Fit panel.** A–D each propose 2 control and 2 treatment fits on F0/1001 and
   F2/1002 = 16 primary fits total, conditional on source qualification. E requires
   zero new fits only with an already qualified OOF matrix. Missing clean parents,
   teacher fits/caches and D's minimum four student-transfer fits are additional.
   Each idea is separately admitted; no runs have been scheduled.
6. **Validation and uncertainty.** Use official pooled weighted Pearson, full
   causal sequences, matched scored rows, per-target and cell deltas. The working
   +0.0005 WP screening threshold is not significance. Folds and seeds are
   confounded, and fixed-prediction component intervals omit training and selection
   uncertainty. Existing valid and LB are exposed; no fresh holdout is claimed.
7. **Leakage and coverage.** Exclude outer components from all fitting ancestors,
   transforms, OOF predictions and nested weight selection. No true lagged target
   or future input enters a predictor. Audit component/source aliasing, resets,
   warmup and prefix invariance. Anonymous IDs do not establish chronology.
8. **Decision.** Follow a useful reproducible improvement into source-clean
   replication and exact deployed-blend measurement. Mixed/small results remain
   uncertain; negative evidence narrows the fixed recipe. A zero-crossing interval
   alone does not close the direction.
9. **How to run.** No runnable commands are claimed. Each card needs an implemented
   adapter, frozen recipe, source/data/cache binding, receiver numerical checks,
   B300 profile and owned run paths. Full-sequence CPU serving verification is
   separate and requires the exact delivered ZIP.
10. **Evidence to return.** Source/config/env and lineage hashes, all cells,
    losses/LR/preclip gradients and steps, per-target predictions, metric/mask,
    time/VRAM, exclusions, separate evidence and weights archives with SHA256;
    serving requires exact-ZIP parity and one-vCPU size/runtime/quality results.
11. **Results.** Zero new runs, tests, models, submission artifacts or score
    claims. [README](https://github.com/VladKochetov007/WunderFunder/blob/codex/gold-hypotheses-20261102/research/experiment_drafts/gold-frontier-hypotheses-20261102/README.md) links prior evidence and the comparison cards.
