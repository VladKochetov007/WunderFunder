# [Experiment] Implement staged B300 teacher search and ensemble distillation

The expanded B300 plan previously described experiments without runnable stage
configs or ensemble targets. R4 now materializes complete source-bound cells,
uses component-clean inner selection, reuses exact controls, confirms frozen
recipes on outer folds and trains matched GRU students against labels, one
teacher or three-teacher means. It reuses PR32's causal trainer, Muon with FP32
master/update state, recovery, canonical scoring and R3 admission rather than adding a training loop.

[Precision amendment](../PRECISION_AMENDMENT_20261024.md): new R4 cells explicitly
use the PR40 BF16/BF16x9 runtime. Historical strict-FP32 packets and the active
four W2 teacher fits retain their source pins; this is a new combined source.

[Receiving-host commands](README.md)
and [allocation and complete recipe](recipe.json).
G reference and candidate share the amended precision/backend/compiler settings.
The existing four-fit R3 packet remains unchanged. No local tests, model imports,
training, inference or GPU profiling were run. No jobs were dispatched.

1. **Status and lineage.** E/H-B300-HIGH-COMPUTE-R4-20261024 implements the
   conditional successor to R3 in the same PR targeting `research-team`.
   `launch_ready=false`: receiving-host correctness and resource acceptance are
   still required. Preparation is executable; predictive results do not exist.
2. **Mechanism and family.** Teacher branches test LR, warmup, forward context,
   chunk/backward horizon and capacity. G tests robustness of a frozen candidate;
   H tests the teacher–student connection. Better teacher fit or compute alone
   does not establish better CPU student predictions. Sequential coordinate
   search can miss interactions and is not a claim of optimal hyperparameters.
3. **Compute advantage.** Every new R4 fit uses an eight-B300 DDP group with BF16
   autocast and process-wide cuBLAS BF16x9 emulation. Master weights, gradients,
   loss and optimizer state remain FP32; Muon GEMMs are not IEEE-FP32 guaranteed.
   Teachers use efficient attention and compiled blocks, recomputation off.
   Additional disjoint groups can execute independent
   cells with separately allocated RAM and scratch. Models/optimizers are fully
   replicated. Larger context/capacity and three-seed teacher means consume
   additional compute; measured GPU-hours and endpoint quality decide whether
   that cost is useful. No performance estimate is presented as measurement.
4. **Frozen parameters.** Thirty epochs, final index 29, unprofiled batch 2/rank, clip 1,
   dropout 0, full fitting-only normalization and causal raw112 input. Teachers
   use R3 Muon/auxiliary AdamW, LR candidates 1e-4/3e-4/1e-3, warmup 1%/5%/10%,
   cosine floor .1, contexts 1024/2048/4096, chunk 512/accumulation 8 versus chunk 2048/accumulation 2,
   and 1792×40×28heads versus 2560×40×40heads. H fixes GRU512×2, AdamW 3e-4,
   beta (.9,.95), eps 1e-8, WD .01, warmup 5%, cosinefloor .1, chunk 512/accumulation 8,
   hint dimension 0, KD weight 0/1. All concrete configs include every loader field.
5. **Allocation.** B6 + C4 + D4 + E2 + F2 + G12 teachers and H6 students: 36 additional
   fits, or 40 including the original four A/R3 teachers. C–F reuse exact
   completed controls. If the selected recipe equals the G reference, six
   duplicate teacher fits are aliases: 34 total including A. Nominal cost is
   1,020 teacher fit-epochs and 180 student fit-epochs; size and membership differ.
   Failed attempts and all engineering profiles/exports are recorded separately.
   Stages require explicit advancement; no automatic queue or blind retry.
6. **Validation and statistics.** B–F use two frozen inner component holdouts
   from outer fitting pools. G crosses 3 new seeds with F0/F2 on full outer fitting
   membership; H uses 2 matched student cells. Canonical pooled clip2 weighted Pearson,
   per-target results and paired component bootstrap accompany fixed-final
   endpoint receipts. Exposed outer folds and correlated components/seeds do not
   become untouched data or guarantee significance through repetition.
7. **Leakage and lineage.** Inner reservation removes whole connected
   components, including their train rows, from fit. The loader rederives exact
   membership and parent/source hashes. Normalizers see only the corresponding
   fit set. Every reused endpoint binds source/config/split/weights/score and
   prediction manifest. H parents are fixed G candidate seeds 301001/301002/301003;
   single KD uses 301001 before viewing scores. No best-seed or outer-checkpoint
   selection, future-label inputs or reused outer-fit controls for inner search.
8. **Decision.** Retain all arms, including explicit failures; choose only
   variants with both required completed fold endpoints and give a reason.
   Mean WP gain ≥.001 without a fold regression is a useful signal considered
   with variation/cost, not a mechanical significance criterion. Mixed results
   motivate narrower diagnosis. Intervals crossing zero do not close the family.
   G/H comparisons do not automatically change recipes or launch follow-ups.
9. **Execution and acceptance.** Immutable materialization, explicit per-cell
   profile-one/eight/peak, admission, fit/resume/evaluate/cache/bundle commands.
   R4 receiver checks extend R3 CUDA recovery and objective/Muon DDP parity with
   inner-split, ensemble, staged-campaign and BF16 streaming/backend contracts.
   Legacy checks disable cuBLAS emulation in separate subprocesses. Exact 12-update profiles,
   finite gradients, peak-LR stress, 20% GPU reserve and conservative host/disk
   admission are prerequisites. No local correctness/performance pass is implied.
10. **Transfer and artifacts.** One/eight-rank full-fit prediction parity binds
    each completed G teacher before KD. Streaming FP32 mean merge requires
    three distinct same-recipe seeds, preserves original parent provenance and
    publishes atomically. It creates a composition identity, not a fake model
    checkpoint. H compares the same student with labels/singleKD/mean3KD.
    Preserve score/prediction/profile/failure receipts and separate evidence and
    weights bundles. One-core offline CPU / 20 MB submission acceptance remains a
    later measured gate; native211/TL3 compatibility and quantization are not
    asserted by raw112 training code.
11. **Evidence.** Static Ruff/AST/JSON/shell/module-structure checks and independent
    source review only, recorded in R4 VALIDATION.json. Tests are prepared for
    B300 and deliberately unexecuted locally. No empirical LR optimum, full
    memory fit, convergence, teacher gain, student gain or submission result.
    Unified index and append-only ledgers link this implementation to the
    earlier plan without rewriting historical allocations.
