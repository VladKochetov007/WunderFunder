# [Experiment] Causal attention and FFN routing on B300

1. **Status, lineage and owner.** Implements two packet-local Transformer studies,
   SR-A and SR-F, with ten unique fits prepared and zero executed. Preparation:
   Ivan / Codex; a receiver must claim each cell before running it. The branch starts
   at research-team `7e4074314a62d0907c9608f9220cf72958708a2d` and uses canonical
   PR75 runtime `8626000a4b965763a70011a0291a891aa6535d7e` without modifying it.
   [Current team PR audit](sources/PR_AUDIT.md) distinguishes final producer reports,
   stale progress and unverified deadlines. [Recipe](recipe.json) is authoritative.
2. **Why it should work.** SR-A tests a signed combination of two causal attention
   maps against a matched additive combination and a plain anchor. SR-F tests
   current-row routing between two existing FFN partitions against a static gate
   and the same plain anchor. They address context contamination and a fixed FFN
   response, respectively. [Mechanisms](MECHANISMS.md) specify equations, alternative
   explanations, literature and falsifiers. These are hypotheses, not measured gains.
3. **Compute advantage.** B300 compute buys full-data controlled architecture
   comparisons and fresh validation artifacts. Both attention arms execute two
   attention calls; both FFN arms evaluate both partitions. There is no sparse
   dispatch, kernel tuning, queue manager, batch search or throughput optimization.
   A teacher gain would justify a separate, measured student or serving study.
4. **Frozen comparison.** H768/L24, 12 heads, FF width 3072, context 1024, chunk 256,
   batch 128 per rank, accumulation 4, two B300 GPUs, four epochs, fixed E4 endpoint,
   all-epoch retention. Native Muon/AdamW, LR 3e-4, WD 0.01, clip 1, 5% warmup and
   cosine floor 0.1. Preserve the scoped BF16/BF16x9 recipe with FP32 master weights,
   gradients, losses and optimizer/state tensors. Normalization and primary WMSE
   are unchanged. Native config differences from the pinned control are seed only.
5. **Bounded fitting panel.** Five variants × seeds 205201/205202 = ten unique fits.
   Each study uses six endpoints: two plain, two mechanism controls, two treatments.
   The two fresh plain endpoints are shared between studies and counted once.
   Historical reused fits 0, teacher/pretraining dependencies 0, student fits 0.
   Each fit attempt is bounded at eight hours; each 50-update profile at one hour.
   The fit upper bound is 160 GPU-hours, profiles 20 GPU-hours, not an ETA or current
   launch authorization. There are no automatic retries or launches.
6. **Validation and uncertainty.** Use all 1,873 official validation sequences and
   3,528,273 scored rows with `global_wp_clip2_v1`. Report every paired seed and
   target delta, both plain-anchor and mechanism-control contrasts, and negatives.
   This repeatedly exposed official split is development data. Two seeds are not
   two folds, shared anchors are not independent observations, and no significance
   or public-transfer claim follows from this screen.
7. **Leakage and ancestry.** All cells train from scratch on the same 10,325 fitting
   sequences. No external teacher, checkpoint, validation-fitted preprocessing or
   target-fed state is used. Attention remains causal and bounded; routing uses
   only the current causal hidden row. The exact split SHA and complete canonical
   plus packet source maps are required at preparation, fit and export.
8. **Decision-changing result.** Mean paired WP gain ≥0.0005 with both seeds positive
   prompts review, including target regressions worse than 0.001. This is an allocation
   heuristic, not automatic promotion. Differential must be compared with additive;
   conditional with static. A win only against a degraded mechanism control is
   insufficient. Mixed results remain inconclusive and require an explicit next
   allocation; a zero-crossing interval alone does not close the direction.
9. **Implementation and execution.** Adapters preserve canonical parameter names,
   optimizer roles, state/cache shape and the public model interface. Thin runners
   reuse the existing fit/profile/checkpoint/scorer components without copying a
   training loop. [Commands](B300_COMMANDS.md) cover data preparation, B300-only
   probes, per-cell profiles, allocation review, fit, strict E4 export, full score
   and the ten-endpoint comparison. Model/data launches require owner and timeout.
10. **Evidence and MIDAS.** An actual metadata-only MIDAS research route is recorded.
    [MIDAS instructions](midas/README.md) export source-bound producer logs, losses,
    epoch index and full scores into the existing training-evidence schema. It
    does not launch jobs or import hypothetical results. Checkpoint bytes are hashed
    but not bundled; MIDAS treats those checkpoint entries as external claims.
    Keep failed and incomplete receipts and immutable source revisions.
11. **Checks and limitations.** [Static review](STATIC_REVIEW.json) records lint,
    AST, config/source identity and command syntax checks. Local tests, model imports,
    training, inference and profiling were not run. Numerical/causal/gradient/reload
    acceptance and actual memory/quality measurements remain pending on B300.
    Current jobs, shared pipeline files and optimization recipes are unchanged.

[Packet index](README.md) · [experiment ledger](../../experiments.jsonl) ·
[hypothesis ledger](../../hypotheses.jsonl).
