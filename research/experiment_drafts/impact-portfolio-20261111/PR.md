# [Research] Latest PR evidence and ten next-impact screens

[Portfolio and cards](https://github.com/VladKochetov007/WunderFunder/blob/codex/next-impact-experiments-20261111/research/experiment_drafts/impact-portfolio-20261111/README.md)
· [PR60–80 audit](https://github.com/VladKochetov007/WunderFunder/blob/codex/next-impact-experiments-20261111/research/experiment_drafts/impact-portfolio-20261111/PR_REVIEW.md)
· [Shared contract](https://github.com/VladKochetov007/WunderFunder/blob/codex/next-impact-experiments-20261111/research/experiment_drafts/impact-portfolio-20261111/CONTRACT.md)

1. **Status, lineage and owner.** Proposal-only portfolio for Ivan, based on
   research-team `7e4074314a62d0907c9608f9220cf72958708a2d`. Ten H/E IDs in the
   append-only ledgers; seven Transformer and three K16 GRU screens. PR60–80
   reviewed individually, older overlaps retained. No new trainer implementation.
2. **Why it should work.** Numerical/grouped representations, private readouts,
   intermediate supervision, shared weights, task auxiliaries and state stability
   address different possible limitations. Each card gives controls, predicted
   observations, falsifiers and alternative explanations. Prior NLL, grouped
   encoding and multi-horizon work are explicit continuations, not new discoveries.
3. **Compute advantage.** Additional compute can distinguish useful mechanisms
   from current negative/mixed directions. No larger-model-by-default assumption,
   speedup promise or leaderboard forecast. Hardware optimization stays with the
   receiving team; large teachers require separately budgeted deployment/transfer.
4. **Frozen comparison.** Read-only PR75 plain and PR79 K16 configs are source-
   pinned. T preserves Muon, BF16/BF16x9, fixed4E; G preserves AdamW/FP32, fixed6E.
   B300 GRU use requires an explicit paired device amendment and fresh controls.
   Full official train/valid, exact state/masks, train-only normalization, fixed
   final endpoints. Each card changes only its declared scientific package.
5. **Fitting panel.** Each card: two fresh seeds204101/204102 × control/treatment,
   four fits total, no reused control or teacher fit. Forty fits is the full
   inventory, **zero selected/dispatched**. Card09 transfers an idea, not weights;
   both arms start from scratch. Profiles/export are extra. Continue PR8 rather
   than opening a duplicate grouped-encoding experiment.
6. **Validation and uncertainty.** Canonical pooled global_wp_clip2_v1 on all
   1,873 valid sequences/3,528,273 scored rows. Both seed and target deltas;
   repeated development exposure and all tried comparisons disclosed. Conditional
   component intervals are not fit-variance estimates or significance claims.
7. **Leakage and coverage.** Whole sequences, source/ancestor identities,
   train-only transforms, reset and prefix invariance. Future-label heads use
   labels only in loss, never model inputs. No valid/test updates. Shuffled IDs
   do not prove chronology, and official file separation does not itself prove
   component independence.
8. **Decision-changing result.** Working+0.0005 mean-WP screen with both paired
   deltas positive and per-target tradeoff review; not a promotion threshold or
   guaranteed effect. Preserve small/mixed/negative outcomes and informative
   follow-ups. No automatic40-fit sweep or valid-driven coefficient tuning.
9. **How to run.** Not runnable in this PR. Each selected mechanism needs its
   adapter, exact implementation source/config, receiver ownership/timeout/budget,
   real data binding and remote checks/profile in a separate implementation PR.
   Missing adapters are listed per card. No local tests/inference/training.
10. **Evidence to return.** Source/config/data/split/runtime/ancestry hashes,
    producer-stamped progress, losses/LR/preclip gradients/skips, fixed endpoint
    metrics/predictions with row coverage, separate evidence/weights archives.
    Fresh mirror timestamps do not establish fresh optimizer progress.
11. **Results and trace.** Zero new measurements. PR_REVIEW.md preserves results
    and incompleteness of60–80, including stale75/80 telemetry. recipe.json is a
    proposal inventory, not a trainer config. Independent critique and primary
    sources are documented; PR76's existing CPU ZIP is unchanged and unexecuted.
