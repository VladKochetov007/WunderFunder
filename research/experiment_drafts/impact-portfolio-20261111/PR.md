# [Experiment] IMPACT81: ten executable B300 mechanism screens

This PR implements all ten frozen hypotheses, with **40 fresh fits prepared and
zero executed**. Cards07–10 add16 configurations to the earlier24. The existing
canonical pipeline and native optimization recipes remain unchanged; the GRU
family has an explicit paired device amendment from L4 to B300.

1. **Scope and lineage.** Seven Transformer screens use pinned PR75
   `8626000a4b965763a70011a0291a891aa6535d7e`; three K16 GRU screens use PR79
   `67c3bd818ea243177eeb1dc789cd3d145eba0e81`. Preparation owner: Ivan / Codex.
   Receiver owner/allocation remain unassigned. The [latest implementation amendment](IMPLEMENTATION_07_10.md)
   supersedes the remaining proposal-only notes. Running work keeps its original checkout.
2. **Mechanisms.** Cards01–06 cover numerical embeddings, same-row group mixing,
   target-private readouts, intermediate supervision, copied-versus-shared depth,
   and fixed-lag ranking. Card07 adds future-label heads at row offsets1/16/64.
   Card08 regularizes a noisy hidden-state fork while retaining the clean GRU path.
   Card09 adds a bounded variance head; card10 adds signed clipping-event heads.
   Individual [cards](cards/) retain alternatives, expected observations and falsifiers.
3. **Compute advantage.** Compute buys matched full-data mechanism comparisons.
   Added losses/forks cost work; tied depth still executes24 logical layers.
   No kernel/throughput tuning, precision conversion, quantization, batch search,
   optimizer change or queue manager is part of this update.
4. **Frozen recipes.** Transformer: H768/L24/12heads, FF4, context1024,
   gradient horizon256, batch128, accumulation4, two B300s, four epochs, Muon/AdamW,
   BF16/BF16x9. GRU: K16/H256/L2, chunk1000, batch32, accumulation1, one B300,
   six epochs, AdamW, FP32. Both retain LR3e-4, WD0.01, clip1,5% warmup/cosine
   floor0.1 and all-epoch retention. GRU changes only paired seed and required device.
   [Contract](CONTRACT.md) binds full10,325-train/1,873-valid memberships and preprocessing.
5. **Fit accounting.** Every card has two controls and two treatments at
   seeds204101/204102:40 total,0 reused controls,0 teacher/pretraining fits,0 executed.
   Card05 uses copied-depth controls. Card08 controls use matched clean traversal.
   Older L4 GRU scores cannot replace the new B300 controls. Card10 may be infeasible
   under its fixed train-prevalence requirement; that outcome must be preserved.
6. **Validation and uncertainty.** Score every configured final endpoint, E4 or E6,
   with canonical `global_wp_clip2_v1` on all3,528,273 eligible validation rows.
   Report every paired seed/target delta and incomplete/negative outcome. This
   repeatedly exposed official split is development data; two seeds are not two
   folds, and no statistical significance or leaderboard transfer is established.
7. **Leakage and reductions.** No label enters a model input or carried state.
   Card07's loss may read future labels across chunk boundaries within the same
   fitting sequence, with one global pooled-target denominator per horizon.
   Card08 noise has an independent checkpointed RNG; only the clean state survives.
   Card09 leaves primary weights unchanged. Card10 labels come from raw pre-clipping
   targets; train-only counts require each tail in≥100rows and≥32sequences per target.
   Both card10 arms stop before profile/fit if this prerequisite fails. Global
   accumulation-window denominators and empty-cell behavior are explicit.
8. **Decision-changing result.** Preserve the descriptive allocation heuristic:
   mean paired WP gain≥0.0005, both seeds positive, and review target regressions
   worse than0.001. This is not an automatic promotion or significance threshold.
   Better auxiliary loss, concordance or class accuracy alone is insufficient.
9. **Implementation and launch.** New code lives only in this packet. Existing
   canonical source and configs are untouched. Cards01–03 keep their prior runner;
   the explicit auxiliary engine for04–10 reuses primary WMSE, sampling, update,
   checkpoint, monitor and scorer components. [Cards07–10 commands](B300_CARDS07_10.md)
   select the correct source/precision/world size and cover prepare, receiver checks,
   per-cell50-update profile, resource review, fit, strict export, full score and
   paired comparison. Every stage has an owner and timeout; no automatic launch/retry.
10. **Evidence and MIDAS.** Keep source/config/data/runtime identities, profile and
    reload decisions, separate primary/combined losses, mechanism diagnostics,
    LR/preclip norms, epoch checkpoints, fixed endpoint and prediction manifests.
    Heads are removed only after strict training-checkpoint loading. [MIDAS handoff](midas/README.md)
    records preparation and admits complete family-specific canonical plus packet
    source maps for real evidence export. It is not GPU execution or evidence import;
    checkpoint hashes without bundled weight bytes remain external claims to MIDAS.
11. **Verification and results.** No model measurement or gain is claimed.
    [Static review](STATIC_IMPLEMENTATION_REVIEW_07_10.json) records lint/AST,
    source/config identity and command syntax checks separately from pending B300
    receiver checks. Local tests, model imports, training, inference and profiling
    were not run. Historical static receipts remain unchanged.

[Recipe](recipe.json) · [portfolio and prior PR review](README.md) ·
[experiments ledger](../../experiments.jsonl) · [hypothesis ledger](../../hypotheses.jsonl).
