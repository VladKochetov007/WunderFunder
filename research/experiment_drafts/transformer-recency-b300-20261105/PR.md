# [Experiment] Test causal Transformer attention recency on B300

1. **Status:** implementation and configs; no tests, numerical parity, profiles,
   fitting, export or scores. Clean receipt-bound data and receiver acceptance
   are prerequisites. This is not a ready submission.
2. **Mechanism:** a fixed additive causal distance penalty on existing RoPE
   attention could prioritize recent signals. Compare bias 0 with bias 1 while
   preserving context 1024 and every other setting. Long-lag signal suppression
   is the main competing explanation. This is inspired by, not identical to,
   [ALiBi](https://arxiv.org/abs/2108.12409).
3. **Compute:** eight B300 per independent teacher candidate, no inference-model
   modification for eventual GRU students. Real throughput, memory and transfer
   must be measured, with student fits and teacher caches explicitly additional.
4. **Comparison:** T768×24, 30 epochs, Muon, LR .0003, warmup 5%, batch 1 per
   rank, 1024 context, BF16/BF16x9 scoped Transformer amendment, fixed final
   checkpoint. Configs differ only in bias value; no empirical optimality claim.
5. **Budget:** 2 control + 2 candidate teachers = 4 new fits, on F0/101001 and
   F2/101002. No reused fits. Student transfer requires at least 4 new students
   and four teacher-cache exports, plus profiles and receipt checks.
6. **Validation:** official pooled masked clipped global WP, per-target deltas,
   paired sequence age/coverage and all failed cells. Two fold-seed-confounded
   pairs cannot prove significance. Existing validation and public scores are
   development-exposed; fixed-prediction bootstrap omits fitting uncertainty.
7. **Leakage:** preserve causal window/position masks, fitting-only normalizers,
   complete sequence resets and source-qualified component exclusion across every
   ancestor. Prefix and chunk parity checks are mandatory on receiving host. IDs
   alone do not prove chronology, and all-corpus PR54 is ineligible as a parent.
8. **Decision:** +0.0005 paired WP is an exploratory allocation threshold;
   mixed, zero-crossing intervals or small gains remain uncertain, negative
   effects narrow this particular value and context only. Teacher benefit must
   be followed by measured student and complete served-blend contribution.
9. **Run:** see README for exact configs, receiver acceptance, eight-rank
   profiles, explicit reviewed profile receipts, fit, export and score commands.
   Math attention only; no assumptions about FlashAttention or other backends.
10. **Evidence:** source/config/ancestry/environment hashes, all losses, LR,
    gradients, steps, timings, GPU memory, failures, raw prediction manifests,
    official scores, and separate evidence/weights archives. Model-to-CPU serving
    and exact submission checks remain separate.
11. **Results:** none. No local tests or training ran and no leaderboard effect
    has been measured. The current live jobs remain untouched.
