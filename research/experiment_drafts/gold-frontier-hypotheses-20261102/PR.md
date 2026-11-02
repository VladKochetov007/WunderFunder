# [Research] Gold frontier: implement five comparisons for receiver evaluation

1. **Status.** Implemented A–D training paths and E bounded nested-OOF analysis on
   the existing training/metric components. All are **receiver acceptance
   pending**: no local tests, fits, model imports, inference or GPU profiles.
   [Packet](https://github.com/VladKochetov007/WunderFunder/blob/codex/gold-hypotheses-20261102/research/experiment_drafts/gold-frontier-hypotheses-20261102/README.md),
   [cards](https://github.com/VladKochetov007/WunderFunder/blob/codex/gold-hypotheses-20261102/research/experiment_drafts/gold-frontier-hypotheses-20261102/HYPOTHESES.md).
2. **Changes.** A adds 2/16-row filtered residuals to label loss; B adds
   target-PCGrad for a restricted one-GPU label-only GRU; C compares reset versus
   prefix-replayed state (the baseline already carries state); D adapts the
   existing causal TCN to sequence training versus GRU; E fits a predeclared
   shrinkage rule on inner OOF and scores a separate outer panel. Each card
   documents its mechanism, alternative explanation and related earlier work.
3. **Compute.** A–C use one GPU per fit; D configs require one B300 per fit.
   E is bounded CPU numerical analysis, not a GPU job. Actual B300 throughput,
   memory and quality are unmeasured; more compute is not a guaranteed win.
4. **Frozen comparisons.** Exact preliminary configurations and commands are
   in the packet. A/B are cold labels-only GRU, not the inherited TL3 recipe;
   C requires a source-qualified distillation cache. D is raw112 strict-FP32,
   6 epochs and fixed final checkpoint; GRU256x2 vs TCN256x6 are not claimed
   equal-compute. Separate cache/transfer and live server checks remain necessary.
5. **Fit cost.** A–D each have 2 control + 2 treatment fits = 16 new fits total
   if all are admitted; E needs no training only when the qualified OOF matrix
   exists. Missing clean parents/caches, D student transfer (at least four more
   fits), exports and receiver profiles are additional. No runs are scheduled.
6. **Validation.** Use matching complete held-out sequences and official pooled
   `global_wp_clip2_v1`; report both target deltas, every cell and missing data.
   F0/1001 and F2/1002 confound fold/seed; existing folds and public feedback are
   exposed, so no independent significance or leaderboard uplift is asserted.
   Fixed-prediction component bootstrap excludes fit/selection uncertainty.
7. **Leakage and coverage.** Fit-only transforms and fold-clean complete ancestry;
   causal inputs, sequence resets and no teacher-forced labels. E validates
   declared ancestry graph, hashes and component intersections, but its receipt
   semantics, full ancestry and NPZ-to-source row/mask mappings require independent
   operator review and attestation. The manifest digest does not itself prove the
   timestamp of freezing. It must not be used as automatic proof of no leakage.
8. **Decision.** A prespecified +0.0005 paired WP is an initial resource-allocation
   signal, not a significance claim or universal cutoff. Follow useful results
   with more clean cells, student transfer and exact serving checks; mixed outcomes
   remain uncertain. Never add validation deltas to public scores.
9. **Run.** The packet contains per-arm commands and configs. Bind real data, clean
   splits, cache identities and worker ownership; complete receiver-side numerical,
   causal and gradient checks and B300 profiles first. D requires a reviewed
   matching profile receipt; E additionally requires qualified OOF data and
   externally documented freeze/lineage review. No existing jobs are modified.
10. **Evidence.** Preserve source/config/split/ancestor hashes, full logs including
    LR/preclip gradients, all per-target grades, profiles and failures, plus
    separate evidence/weights archive hashes. E is capped at 5M selected rows and
    may require substantial unmeasured host RAM. A submission additionally needs
    exact-ZIP parity, offline Python 3.11, <=20 MB, 1-vCPU runtime <=60 min.
11. **Results.** Zero new measurements. Neither a first-place claim nor a ready
    submission follows from code preparation. Baseline evidence is linked in the
    cards. Current PR #54 all-data checkpoints must not be reused as clean parents
    on components they have already seen.
