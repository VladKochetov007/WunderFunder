# [Experiment] IMPACT81: six implemented Transformer screens for B300

This update implements cards04–06 alongside the previously prepared cards01–03.
All code/configs are preparation only: **24 fresh fits planned, zero executed**.
The existing training pipeline and native optimization recipe are unchanged.

1. **Scope and lineage.** Cards01–06 are implemented pending receiver acceptance;
   cards07–10 remain proposals. Preparation owner: Ivan / Codex; receiver owner
   and GPU allocation remain unassigned. [Cards04–06 amendment](IMPLEMENTATION_04_06.md)
   supersedes their earlier separate-PR proposals. The runtime is pinned to PR75
   `8626000a4b965763a70011a0291a891aa6535d7e`, plain Transformer. Existing running
   work must retain its original checkout and receipts.
2. **Mechanisms.** Card01 adds continuous scalar embeddings;02 mixes current-row
   feature groups;03 adds target-private rank8 readouts. Card04 adds label heads
   after blocks6/12/18 with primary+0.1×mean auxiliary WMSE. Card05 compares24
   independent copied-initialization blocks to12 unique blocks applied twice,
   preserving24 independent KV histories. Card06 adds0.05×within-chunk lag32
   pairwise logistic loss with clipped-label separation≥0.5 and fixed temperature1.
   Hypotheses, alternatives and falsifiers remain in the individual [cards](cards/).
3. **Compute advantage.** Extra compute buys matched full-data mechanism screens,
   not a promised speedup. Auxiliary losses add work; tied weights do not halve
   the execution depth. No kernel tuning, new precision, optimizer change,
   quantization, queue manager or throughput optimization is included.
4. **Frozen recipe.** H768/L24,12 heads, FF4, context1024, gradient horizon256,
   batch128, accumulation4, two B300 ranks, four epochs, fixed finalE4. Keep Muon/
   AdamW roles, LR3e-4, WD0.01, clip1,5% warmup/cosine floor0.1, BF16/BF16x9 and
   compiled temporal blocks. Use all10,325 train sequences and1,873 valid sequences.
   [Contract](CONTRACT.md) binds the exact split, source and all-fit normalization.
   Card01's previously declared64-position/train-sequence quantile estimator stays
   unchanged. Card05 uses its explicit copied-depth control, never an old plain score.
5. **Fit accounting.** Each implemented card has two controls and two treatments at
   seeds204101/204102. Cards04–06 add12 fresh fits; all six total24. Zero reused
   controls, zero teacher/pretraining fits and zero launched fits. The ten-card
   inventory would total40 only if the remaining proposals were later selected.
6. **Validation and uncertainty.** Score every fixed endpoint with the canonical
   `global_wp_clip2_v1` on3,528,273 scored valid rows. Report every seed/target
   delta and incomplete or negative result. This official split is repeatedly
   exposed development data; two seeds are not two folds and do not establish
   significance or public transfer.
7. **Leakage and reduction.** No labels enter model inputs. Card04 uses the exact
   primary mask/weights and global full-window denominator for each head. Card06
   pairs only rows within a sequence/chunk, uses the training mask (not missing
   train `is_scored`), and reduces per-target pair mass over all ranks and
   microchunks before backward. Empty targets contribute zero to the fixed
   two-target mean. Card05 shares parameters, not temporal state. No valid/test updates.
8. **Decision-changing result.** Retain the declared descriptive allocation
   heuristic: mean paired WP gain≥0.0005, both seed deltas positive, and review
   per-target regressions worse than0.001. This is not an automatic promotion or
   significance rule. Ranking concordance or intermediate-head loss improvement
   alone cannot establish official-metric benefit.
9. **Implementation and launch.** All additions live in this packet. Cards01–03
   retain their existing canonical runner. Cards04–06 use an explicit local
   `TrainingEngine` for the additional loss/depth behavior, reusing canonical
   batching, primary WMSE, DDP, optimizer updates, monitor, checkpoint and resume
   components. No global monkeypatch or existing-source rewrite is used.
   [B300 commands](B300_COMMANDS.md) cover preprocessing, source-bound checks,
   per-cell50-update profile, operator resource review, fit, strict endpoint export,
   canonical score and paired comparison. Every stage has an owner and timeout;
   no automatic multi-cell launch or retry is provided.
10. **Evidence and MIDAS.** Preserve source/config/data/runtime identities,
    acceptance/profile/reload receipts, primary and auxiliary losses, pair counts,
    per-target concordance, LR/preclip gradients, fixed endpoint, epoch snapshots,
    manifests and full score coverage. Optional depth/head diagnostics require
    separate receiver analysis. [MIDAS handoff](midas/README.md) records the
    preparation route and imports only real producer-bound evidence; weights are
    not copied, checkpoint hashes remain external claims to MIDAS. No MIDAS
    research-agent run, GPU execution or evidence import is claimed.
11. **Results and verification.** No measurements exist for these new cards.
    Static lint/AST, metadata/hash and shell-syntax checks are recorded separately
    in [the cards04–06 static review](STATIC_IMPLEMENTATION_REVIEW_04_06.json).
    Receiver probes, real full-size profiles, validation scores and paired deltas
    remain pending. Local tests, model imports, training, inference and profiling
    were not run. The prior source-bound receipt remains historical rather than
    being rewritten for new code.

[Recipe](recipe.json) · [portfolio and prior PR review](README.md) ·
[experiments ledger](../../experiments.jsonl) · [hypothesis ledger](../../hypotheses.jsonl).
