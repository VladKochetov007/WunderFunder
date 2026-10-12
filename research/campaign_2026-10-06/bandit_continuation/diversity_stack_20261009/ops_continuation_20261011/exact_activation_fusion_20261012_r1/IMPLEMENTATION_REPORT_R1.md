# Exact activation and absmax fusion, implementation R1

This isolated branch implements the single H-CPU-MAX-01 inference treatment assigned for a benchmark-only runtime check. It does not change model weights, training, serving selection, or the main worktree.

The Q8 GRU hidden update now tracks the maximum absolute value of each newly produced state as it writes the state vector. The Q8 projection GELU path tracks the same statistic while evaluating the already-required activation. Both pass the resulting whole-vector maximum to the existing Q8 code encoder, skipping the encoder's separate maximum scan. Activation/GRU arithmetic, the encoder byte operation, dynamic scaling rules, and recurrent state reads/writes are kept intact. Scalar and available AVX2/AVX512 implementations use the same treatment; Q16/factor and LSTM paths continue through the existing no-tracking/full-scan behavior.

The independent review basis is `research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/zen4_cpu_research_20261012_r1/reviews/ALG_REVIEW_B_20261012_R1.md`. The proposal's current scientific status was `proposed_not_admitted`; this implementation is not evidence of a runtime gain.

The implementation is committed at `4fbfcbb9d53b29f8930546e3cea550a35e241b8e` on `codex/exact-activation-max-fusion-20261012`, based on `1f78b7b54a443359a5a9dde08dfff9535da0d622`. It changes only:

- `native/src/approx.rs`
- `native/src/rnn_fast.rs`
- `native/src/vnni.rs`

The exact source patch is retained as `IMPLEMENTATION_DIFF_R1.patch` (SHA-256 `ffc6cb19c9440c59431b9219f292ee0b6b0b594b707f214f42e1a3e9f2f835ef`).

Focused invariant checks passed under the campaign's one-core, 1.2 GB, 120-second transient limit: 5 passed, 0 failed. They compare GELU output against a separate absmax scan; compare scalar and AVX2/AVX512 GRU output and Q8 codes/scales; and compare the precomputed-max encoder against the full-scan encoder for ragged vectors and exceptional values. The initial compile exposed three local integration issues (an inline/target-feature annotation conflict, a Q16 return-type mismatch, and unnecessary unsafe wrappers); those were fixed before the passing test run.

The frozen x86-64-v3 release extension build succeeded in 34.94 seconds with Rust/Cargo 1.94.0, one Cargo job, no PGO, and the crate's release LTO settings. The candidate ZIP was built from the exact compact canary and changes only `v3/wnn_native.abi3.so`. Exact source/build/artifact hashes and the benchmark protocol are pinned in `CANDIDATE_BUILD_RECEIPT_R1.json` and `PREBENCHMARK_FREEZE_R1.json`.

The frozen comparison uses only ten complete feature sequences of 20,000 rows and two reset passes per panel (400,000 callbacks), with order baseline/candidate then candidate/baseline. It computes no score. Even a positive local result is specific to the measured Intel Core i9-11900H and cannot establish AMD EPYC 9224 portal runtime. Candidate artifacts remain benchmark-only pending independent root review.
