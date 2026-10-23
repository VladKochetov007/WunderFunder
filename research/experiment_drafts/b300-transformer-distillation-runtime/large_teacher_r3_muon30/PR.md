# [Experiment] Prepare 30-epoch Muon teachers for eight B300s

The sequence trainer previously hardcoded AdamW, and the repository's recurrent
Muon silently used BF16 Newton–Schulz on CUDA. This amendment adds semantic
Transformer Muon groups with explicit FP32 matrix work, checkpoint-compatible
hybrid state, and a correctly indexed warmup/cosine schedule. It prepares the
same four cold-start 630M/1.54B teacher fits for 30 full-data epochs.

[R3 launch packet](research/experiment_drafts/b300-transformer-distillation-runtime/large_teacher_r3_muon30/README.md)
and [complete parameters and rationale](research/experiment_drafts/b300-transformer-distillation-runtime/large_teacher_r3_muon30/PARAMETERS.md).
Historical R1/R2 packets remain unchanged. No local tests, model imports,
training, inference or GPU profiles ran; no jobs have been dispatched.

1. **Status and lineage.** E/H-B300-LARGE-TEACHER-R3-MUON30-20261023 replaces the
   four unlaunched R2 cells, not four additional fits. Implementation is prepared;
   `launch_ready=false` pending receiving-host acceptance. No inherited checkpoint.
2. **Mechanism.** Compare 1280×32 (~630M) with 1792×40 (~1.54B) under one fixed
   longer Muon recipe. If capacity limits useful causal dependencies, expect
   higher paired teacher WP at the larger size. Width and depth change together.
   This is not an isolated Muon-versus-AdamW, epoch, batch or scheduler comparison.
3. **Compute advantage.** Eight B300s through replicated DDP; strict FP32 includes
   NS intermediates and math attention. Accumulation2→8 amortizes four times fewer
   Muon updates per data pass; effective row batch changes explicitly. Optimizer
   phases have CUDA-event timing, read after an existing synchronization, and
   are reported separately from complete training-window time. No speedup claim.
4. **Frozen recipe.** Thirty full epochs, final index29; raw112 causal tokens,
   context1024, detached chunk512, batch2/rank, accumulation8. Peak shared LR3e-4
   in the AdamW-matched RMS convention `0.2*sqrt(max(matrix dimensions))`;
   Muon momentum.95, NS5, epsilon1e-7, independent Q/K/V blocks. Auxiliary AdamW
   beta(.9,.95), epsilon1e-8 for input/output/norm/bias. Matrix decay.01,
   norm/bias decay0, clip1, dropout0. Linear5% warmup (~1.5 epochs), cosine to
   .1×peak=3e-5 on the final actual update. All actual group settings and exact
   planned/warmup update counts enter run identity; scheduler/group state resumes.
5. **Allocation.** Two sizes × F0/101001 and F2/101002 = four new teacher fits,
   120 total teacher epochs, zero students, zero reused endpoints. Source/config
   changes invalidate exact R2 resumes and profiles. Relative to R2, data passes
   grow5x and update count1.25x, not a wall-time prediction. Engineering checks
   and three12-update profiles per cell discard weights and are separate costs.
6. **Validation and uncertainty.** Fixed-final canonical pooled WP, per-target
   scores and all large-minus-reference paired deltas; component bootstrap2000.
   Folds are exposed and seed/fold vary together; conditional component intervals
   do not establish training-seed variability, significance or public transfer.
7. **Leakage and coverage.** Full fitting-only normalization and source-qualified
   component exclusion; all labeled training rows>=99, official scored evaluation
   mask. No true future targets in inputs. Chunk-state semantics and independent
   sequence resets remain explicit. This is no claim of a newly pristine holdout.
8. **Decision.** Mean paired teacher gain≥.001 with both cells nonnegative is a
   useful signal for separately allocated student transfer, considered with GPU
   cost. Mixed/negative results narrow this recipe; a zero-crossing interval does
   not close the family. Hyperparameters are selected engineering priors, not a
   measured optimum; no hidden LR sweep or extra quality fits are allocated.
9. **Execution.** Remote checks include both AdamW/Muon interrupted recovery and
   two-rank Muon parameter/state parity. One/eight-rank exact production profiles
   plus separate eight-rank cold peak-LR stress (warmup0) precede explicit operator
   admission. The stress config cannot admit a production fit. All profiles,
   named LRs, finite gradients, memory headroom and source/runtime/data bindings
   are required; actual hardware/storage/split bindings remain receiver inputs.
10. **Artifacts and serving.** Four fixed scores, paired uncertainty, receipts,
    timings, checkpoints and two-output teacher caches, with separate weights and
    evidence archives. One/eight-rank prediction parity precedes distillation.
    Raw112 sequence students can consume the caches; native211/TL3 needs its own
    adapter. Large teachers are research assets; CPU student submission acceptance
    and a matched four-student labels/KD panel are separate future allocations.
11. **Evidence.** Static Ruff/AST/JSON/shell/structure review only, recorded in
    R3 VALIDATION.json. No numerical parity, peak-LR stability, memory fit, learned
    quality or empirical optimality established. Unified index and append-only
    ledgers preserve both the old recipe and this explicit successor.
