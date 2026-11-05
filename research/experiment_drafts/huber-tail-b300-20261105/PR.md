# [Experiment] Curvature-matched Huber versus weighted MSE

1. **Status.** Implements `huber_curvature` using the existing weighted Huber
   kernel and exposes a source-qualified 2+2 GPU recipe. Receiver checks and
   profiles pending; no local tests, inference, training or results.
2. **Why.** Large residuals may dominate weighted MSE gradients. The treatment
   caps their gradient growth; matching near-zero curvature controls one scale
   confound. Earlier Huber was weaker than WMSE in team reports, so this is a
   targeted continuation with negative prior evidence, not a novel claim.
3. **Compute.** Four independent single-GPU GRU fits, plus receiver profiles and
   prediction export; no extra inference modules. B300 speed is unmeasured.
4. **Comparison.** Cold label-only GRU256x2, raw112, six epochs, AdamW 0.001,
   FP32, same masks and label weights; WMSE vs `2*Huber(delta=0.5)`, fixed final
   epoch. Only local objective curvature is matched, not full optimization.
5. **Fits.** F0/1001 and F2/1002 × 2 arms = four new fits. No assumed reused
   parents, teachers or controls. Exact source and fold lineage must be verified.
6. **Validation.** Official pooled global weighted Pearson, identical evaluation
   rows, t0/t1 and all paired deltas. Two cells confound fold and seed; all existing
   folds are development-exposed. Bootstrap of fixed predictions is conditional.
7. **Leakage.** Full component-qualified fitting membership, fit-only statistics,
   no future labels as inputs or outer-score checkpoint selection. The split loader
   and official masks must be verified against actual supplied files on receiver.
8. **Decision.** +0.0005 WP is a preliminary follow-up signal, not significance.
   Mixed/small results are uncertain; negative results constrain this exact recipe.
   A component interval crossing zero alone does not close the direction.
9. **Run.** See packet README for commands, clean prediction export, scorer and
   resource profile. Missing dataset paths, fold manifests and B300 receipt are
   explicit prerequisites; no remote jobs have been started.
10. **Evidence.** Preserve source/config/data/fold hashes, training logs and
    preclip gradients, optimizer clocks, runtime/VRAM, all grade and prediction
    manifests and separate evidence/weights archive SHA256s. Source identity v2
    rejects exact resume from v1; no silent continuation of older jobs.
11. **Results.** Zero fits, scores, CPU serving tests, leaderboard claims or
    ready submissions. Any downstream KD or fixed-mixture contribution is a
    separately budgeted study.
