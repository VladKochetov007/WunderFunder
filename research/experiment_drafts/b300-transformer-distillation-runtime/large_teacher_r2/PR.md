# [Experiment] Prepare 630M versus 1.54B causal teachers for eight B300s

The previous sequence runtime rejected canonical training files without
`is_scored` and estimated normalization from only 32 fitting sequences. This
amendment fixes the training mask, adds source-bound full-fit normalization and
prepares a four-fit large-teacher panel with explicit receiver checks and commands.

[R2 packet and launch stages](research/experiment_drafts/b300-transformer-distillation-runtime/large_teacher_r2/README.md)
extend the existing causal Transformer/distillation runtime in draft PR #32.
No local tests, model imports, training, inference or GPU profiles were run.

1. **Status, lineage and owner.** E/H-B300-LARGE-TEACHER-R2-20261023, implemented
   preparation on current research-team plus PR32. All four endpoints initialize
   from scratch. Receiving operator/hardware/data paths are unbound. B300 acceptance
   and scientific results remain pending; `launch_ready=false`.
2. **Mechanism and expected observation.** More causal-attention capacity may
   learn dependencies a smaller teacher misses. Compare 1280×32 (~630M) against
   1792×40 (~1.54B), both with 64-wide heads and the same raw112 row representation.
   Expect a reproducible teacher-WP gain if capacity is limiting. This changes
   width and depth together; it is not an isolated width/depth mechanism test.
3. **Compute advantage.** One fit uses eight allocated B300s through replicated
   DDP, strict FP32, activation checkpointing and bounded math attention. Full
   optimizer state resides on every rank. Twelve-update profiles at one and
   eight GPUs establish actual resource use; no speedup, memory fit or training
   wall time is inferred from advertised hardware specifications.
4. **Frozen recipe.** Six complete epochs, fixed final epoch index 5; context1024,
   chunk512/detach, two chunks/update, two sequences/rank, eight ranks. AdamW
   LR1e-4, weight decay .01, clip1, warmup5% then cosine-to-zero, dropout0. Training
   uses all labeled rows after warmup; official evaluation remains scored rows.
   Full-fit-only moments are cached and hash-bound. State resets per sequence;
   KV carries across updates with explicit stale-state TBPTT semantics. No target
   teacher forcing, quantization, numerical-tokenizer or optimizer change.
5. **Allocation.** Two widths × F0/101001 and F2/101002 = four new teacher fits,
   zero new students, no control reuse. Engineering checks/profiles are separate
   discarded work. R1 three-epoch/512-context panels are preserved, not silently
   redefined. No automatic continuation or 30-fit expansion.
6. **Validation and uncertainty.** Fixed final weights, canonical pooled
   `global_wp_clip2_v1`, per-target grades and matched large-minus-small deltas.
   `compare.py` reuses canonical paired component bootstrap, 2,000 replicates,
   seeds1729/1731, using stable centered moments written while the official scorer
   verifies prediction arrays. It rejects inconsistent endpoint grades.
   The folds are exposed; fold and seed vary together. Conditional component
   intervals do not estimate seed variability or establish public-transfer significance.
7. **Leakage and coverage.** Canonical source-qualified component split manifests
   exclude evaluation membership from every fit and normalization pass. Fold,
   GPU model and world size are enforced before model construction. No pretrained
   ancestor. The canonical source contract supplies ordered 20,000-row sequences
   and row99 warmup semantics. New tests cover missing train scoring masks,
   fit-only moments, cache tampering, causality and interrupted recovery. Tests are
   prepared source, not passed GPU evidence.
8. **Decision-changing result.** Mean teacher delta ≥.001 with both cells
   nonnegative is a useful signal for a separately allocated transfer panel,
   conditional on GPU cost. Lower or inconsistent gains motivate LR/context/input
   diagnosis; negative results narrow this exact recipe. An interval crossing
   zero does not close the family. Teacher quality alone proves no student gain.
9. **Execution.** `run.sh` provides check → one-GPU profile → eight-GPU profile →
   reviewed admission → fit → evaluation/cache → separate bundles for each cell.
   The profile gate checks source/config/data/runtime identity, all-rank GPU
   memory reserve and conservative CPU-resume/scratch estimates. Safe resume
   restores epoch-boundary model/optimizer/scheduler/RNG. One/eight-rank export
   parity is a further receiving gate before distillation. No jobs are launched.
10. **Evidence and inference handoff.** Return all four fixed scores, component
    intervals, source/config/split/moments identities, logs, actual peaks/timings,
    checkpoints and manifests in separate weights/evidence archives. Teacher
    caches contain two FP32 outputs without hidden hints. They work with PR32's
    sequence student path; native211/TL3 requires an explicit adapter. Large
    teachers are not CPU competition submissions; student serving acceptance
    remains separate. Next matched labels/KD panel costs four new student fits.
11. **Results and trace.** No predictive results or large-model GPU acceptance.
    Local verification is limited to static Ruff, AST/JSON, shell syntax and
    diff checks, recorded in R2 VALIDATION.json. Unified ideas and append-only
    H/E ledgers link this amendment. Current source must be reprofiled; historical
    checkpoint source identities cannot claim an exact resume after these changes.
