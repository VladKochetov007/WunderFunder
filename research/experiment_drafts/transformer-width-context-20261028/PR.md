# [Experiment] Improve standalone Transformer quality: width versus context

1. **Status, lineage and owner.** Prepared, zero fits; teacher quality, zero student
   fits. H-TRANSFORMER-WIDTH-CONTEXT-20261028 / E-TRANSFORMER-WIDTH-CONTEXT-20261028-R1.
   Depends on draft #47 at `f0d8da40fb545a630e5f042bfcc3b3360784177e`; targets
   `research-team`. The branch includes that dependency until merged. New work is
   the pure-Transformer packet and minimal shared receiver study routing. Ivan's
   preparation packet; actual worker/cell ownership must be claimed on the receiver.
2. **Mechanism.** More hidden capacity may represent interactions the current
   compact teacher misses; more past KV memory may expose longer-lived predictive
   structure. Separate width/context arms distinguish these mechanisms. Alternative
   explanations include undertraining, LR mismatch and extra compute. LR/dropout/WD
   stay frozen because #44 already targets them; its stopped jobs are not resumed.
3. **Compute advantage.** Train the stronger standalone model first; use two seeds
   to avoid promoting one initialization. Reuse canonical code, not a new trainer.
   R4 already proposes1.54B context scaling; this compact source-separated screen
   tests which axis merits more compute. No universal scaling law, CPU suitability
   or public-score forecast. GRU is not a prerequisite for Transformer research.
4. **Frozen comparison.** Baseline768×16/12heads/window1024; wide1024×16/16heads/window1024;
   context768×16/12heads/window4096. All raw112, five epochs, Muon+AdamW1e-4,
   warmup5%, cosinefloor0.1, WD0.01, dropout0, clip1, W2batch128/rank, chunk512,
   accumulation8, efficient attention and compiled blocks. BF16/BF16x9 follows the
   explicit user precision amendment; masters/loss/gradients/state remain FP32.
   No KD/hints. FINAL5 only, `final_only` retention and exact safe-state resume.
   Window is per layer; KV detaches every512 rows and can span weight updates.
   Data/update budget matched, not compute matched. No claim fiveepochs is optimal.
5. **Bounded panel.** Three arms × seeds1001/1002 = six TOTAL fresh Transformer
   fits; no GRU fits and zero reused completed controls. #47 teachers are unrun.
   Baseline adoption into a later KD packet needs explicit ancestry qualification;
   that adapter is not implemented. No automatic combined width+context arm,
   LR sweep,1.5B scale-up or extra seeds. Twelve-update profiles are separate
   disposable engineering work, not scientific replications.
6. **Validation.** One train-only/full-valid component-purged source split, two
   seeds, pooled official clip2 WP with warmup99 and `is_scored`. All six FINAL5
   endpoints, t0/t1, candidate-minus-baseline deltas and conditional centered-component
   bootstrap intervals. Full valid is exposed development data; two seeds and
   conditional intervals do not establish significance or training variance. No
   inner-valid#43-to-full-valid score subtraction or epoch selection.
7. **Leakage/coverage.** Shared #47 producer covers every physical sequence,
   including train528, and excludes all train components touching valid. Missing
   or corrupt groups fail closed. Fit-only normalization, fresh initialization,
   causal raw inputs, sequence resets and exact source/config/data identities.
   Valid features enter overlap detection only; graph linkage is approximate and
   source separation does not prove calendar chronology. Receiver numerical checks
   and actual full-data coverage remain pending.
8. **Decision-changing result.** For either candidate, mean paired gain≥0.001 WP
   with both seeds nonnegative motivates a separately allocated robustness/horizon
   or scale follow-up. Smaller/mixed gains remain uncertain; inspect target tradeoffs
   and cost. Negative FINAL5 results constrain this recipe, not asymptotic model
   capacity. A zero-crossing conditional interval alone is not a direction kill rule.
9. **Run.** [README](README.md) has exact receiver commands. Clean reviewed commit,
   provisioned Linux B300 runtime, two free GPUs, fresh external output root:
   prepare → accept → profile → explicit review/admit → fit → evaluate. Commands
   name one cell and reject cache generation for this study. No local tests,
   training, inference or GPU profiles ran; no remote job started. Profiles may
   fail memory admission; there is no silent batch/precision fallback.
10. **Evidence.** Frozen source/config/data/split/study bindings; correctness and
    all six original/reviewed profiles; rank losses/LR/preclip gradients/updates,
    timing/memory, terminal states and weights; full prediction/grade manifests;
    all paired scores/intervals and resource reports. Return failures too, and
    hash evidence and weight archives separately. No data/model payload in Git.
11. **Results/trace.** [Review](REVIEW.md), [static receipt](VALIDATION.json), unified
    index and append-only H/E records accompany the runnable packet. Zero quality
    measurements for this study. Publication does not resume jobs, merge PRs or
    establish hardware readiness. Compression can later compare student families;
    it is not the current research focus.
