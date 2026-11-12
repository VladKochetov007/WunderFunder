# [Experiment] B300 wave: PR81 mechanism screens with a one-GPU Transformer recipe

1. **Status, lineage and owner.** This is a prepared wave amendment to the ten
   implemented PR81 mechanisms. Cards 01?07 bind PR75 commit
   `8626000a4b965763a70011a0291a891aa6535d7e`; cards 08?10 bind PR79 commit
   `67c3bd818ea243177eeb1dc789cd3d145eba0e81`. The packet's source adapters
   remain isolated. The new placement config, family bindings, launcher, MIDAS
   cell gate and receiver commands are local to this packet. Current state:
   roster prepared; exact sparse runtime admission, family checks, cell profiles
   and fits are pending. The campaign controller supplies the receiving owner.
2. **Why it should work.** The ten cards test distinct mechanisms: continuous
   train-quantile embeddings; same-row group tokens; target-private readouts;
   intermediate supervision; tied Transformer depth; pairwise ranking; future
   label heads; a noisy carried-state GRU branch; a bounded diagonal-variance
   NLL head; and raw-label clipping-tail classes. Each card predicts a
   mechanism-specific diagnostic plus a paired fixed-endpoint WP change. Its
   strongest alternative is added capacity, optimization interaction or seed
   sensitivity. Controls retain the same source, seed, data, schedule and
   family-native objective; diagnostics and the per-card falsifiers are recorded
   in [`cards/`](cards/).
3. **Compute advantage.** Eight B300s can profile or run eight independent
   one-GPU cells concurrently. The initial eight cells complete two matched
   two-seed screens (cards 01 and 02). Measured profile throughput and resource
   use determine how many later cells fit before the shared cutoff; no speedup
   or score gain is assumed. The full queue buys mechanism coverage, while the
   stage order preserves complete paired screens before later work.
4. **Frozen comparison.** Use official train/valid files and the frozen split
   SHA256 `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`:
   10,325 whole training sequences and 1,873 validation sequences / 3,528,273
   scored rows. Cards 01?07 use H768/L24, 12 heads, FF4, context 1024, chunk and
   horizon 256, batch 256, accumulation 4, world size 1, BF16/BF16x9 forward
   with FP32 master parameters, gradients, loss and optimizer state. They retain
   LR `3e-4`, WD `0.01`, clip 1, 5% warmup, cosine floor 0.1, Muon/AdamW roles,
   four full-data epochs and fixed E4 (820 updates/epoch; 3,280 total). The
   update receives the same sequence IDs as the prior two-rank batch-128 recipe;
   reduction order differs, so controls are fresh. Cards 08?10 use PR79 K16,
   H256/L2, chunk 1000, batch 32, accumulation 1, world size 1, strict FP32,
   AdamW, LR `3e-4`, WD `0.01`, clip 1, the same warmup/cosine shape and fixed
   E6. All preprocessing fits training members only; preserve every epoch's
   complete validation score and safe optimizer/RNG/data-position checkpoints.
5. **Bounded fitting panel.** The complete queue is 40 new fits: one control and
   treatment for every card at seeds 204101 and 204102; zero reused controls,
   teachers or pretraining fits. Stage 1 is eight fits: cards 01?02 at both
   seeds. Stages 2?5 are cards 03?04, 05?06, 07?08 and 09?10, each at both
   seeds and both arms. Every selected cell needs an exact full-size 50-update
   profile and separate operator approval. Card 10's two arms are admitted only
   if its training-only raw-tail gate passes: at least 100 finite eligible rows
   in at least 32 training sequences for each tail and target. No profile counts
   as a scientific fit.
6. **Validation and uncertainty.** The primary metric is
   `global_wp_clip2_v1`, on the full official valid file after every epoch and
   at fixed E4/E6. Report each matched seed delta, t0/t1 and pooled WP, and the
   card's frozen mechanism diagnostics. This official development validation
   has prior exposure; it is not an untouched holdout. Two seeds do not provide
   a reliable training-variance estimate or significance test. Component
   bootstrap describes evaluation sensitivity of fixed predictions only.
7. **Leakage and coverage.** Preserve whole official sequence membership and
   do not infer order from shuffled `seq_ix`. Normalizers, PLE knots and all
   other learned preprocessing use training members only. Causal model inputs
   use observations available at each row. Card 07 reads future labels only in
   its auxiliary loss on the same training sequence. Card 08 discards the noisy
   branch and keeps clean carried state. Card 09 variance is a loss-only output;
   its primary mean objective and sample weights stay intact. Card 10 derives
   event classes from raw training labels before official clipping, and its
   train-only gate cannot consult valid labels. The March 5 Predictorium
   statement supplies the adopted train/valid protocol; it is not independent
   evidence of current Connectome timestamps. Additional chronological splits
   need source-qualified timestamp/order evidence.
8. **Decision-changing result.** Preserve PR81's descriptive allocation
   heuristic: mean paired WP gain at least `0.0005`, both seed deltas positive,
   and manual review when a target regresses by more than `0.001`. This is not
   an automatic promotion threshold or significance claim. Improved auxiliary
   loss, accuracy or stability alone is insufficient without the fixed-endpoint
   primary metric. Keep mixed, negative, infeasible and incomplete results.
9. **How to run.** Use the receiving Linux host, provisioned PyTorch/CUDA
   environment, preserved data/cache and exact sparse Git worktrees supplied at
   the two pinned commits. Set actual paths and operator names in
   [`B300_COMMANDS.md`](B300_COMMANDS.md). Run family `prepare` and `check`,
   then profile and approve each exact cell before `fit`; use one B300 per cell.
   Follow the external `CLOCK_R1.json` fit and wave cutoffs. No command auto
   launches a later stage or retries an attempt. Export and score only the fixed
   final checkpoint. Card 10's feasibility receipt precedes its profiles.
10. **Evidence to return.** For each cell preserve source/runtime and packet
    hashes, config/native-config hashes, owner, data/split/membership and
    preprocessing receipts, exact device and environment, profile/reload and
    approval decisions, every epoch's train/valid metrics, primary/auxiliary
    losses, LR groups, preclip norms, clipping/nonfinite/skipped counts, update
    throughput, time/RAM/VRAM/disk peaks, checkpoints, prediction manifest and
    score identity. Keep model weights separately from the metadata evidence
    archive; report artifact SHA256 and storage location.
11. **Results and trace.** No mechanism fit or quality result is claimed in this
    packet. The wave manifest lists all 40 planned cells and pending gates; the
    campaign operator records receiver and profile outcomes, actual completed
    fits, exclusions, scores and independent review there or in linked receipts.
    Historical PR81 results and static reviews remain historical evidence.
    Unrun means no measured gain; public/private transfer is unknown until
    independently observed. Do not infer a leaderboard change by adding a local
    validation delta.

[Wave manifest](WAVE_20261112.json) ? [Contract](CONTRACT.md) ?
[Receiver commands](B300_COMMANDS.md) ? [MIDAS handoff](midas/README.md)
