# Q3 independent preimplementation numerical and delivery contract

Prepared from root's prospective amendment and the actor's proposed portability policy, before reading the new Q3 implementation or running empirical inputs. Root amendment SHA256 6a5e042c0d9040645dc62a3d18c98fdb5e8ed81efc3b1f95fce66d698c0a33c2 authorizes preparation only. The earlier full-four runtime failure remains closed.

## Fixed candidate and scientific scope

The only candidate is distilled256 seed3, plain128 seed3, and chunk500128 seed3, in their original relative accumulation order, with fixed one-third weights. The existing rowwise weight quantizer, dynamic activation quantizer, and block128 matrix algorithm are unchanged. New integer backends implement the same candidate; they do not permit changing weights, clipping, calibration, members, or floating fallback.

This is a post-failure adaptive composition decision using exposed validation history. Calling the dropped plain seed4 redundant is a mechanistic rationale, not independent evidence of dominance. No alternate drop, coefficient, or seed is available if either fixed gate fails.

Runtime comes first: a fresh complete ZIP, three 100k-row trials and one million-row trial, one pinned CPU, SMT sibling busy below 5%, and every qualified projection at or below 50 minutes for 39.4M rows. The million-row trace's resets/repetition structure must be disclosed. Preserve contaminated trials rather than silently selecting the fastest core/run. No quality queries after a failed runtime gate.

Both fixed quality gates are required: quantized Q3 minus FP16 same-composition Q3 is at least -.0001 pooled and on each fold; quantized Q3 minus current fixed-three all-h128 is at least +.0003 pooled and nonnegative on each fold. These are utility cutoffs on exposed folds, not confidence bounds. The two readouts do not authorize separate quantizer, backend, or composition searches.

## Integer kernel correctness

Freeze signed ranges, scale handling for zero/tiny/large/nonfinite vectors, clipping order, rounding mode and ties, and dequantization operation order. Quantized weight bytes, activation bytes, scales, integer dot products and float outputs need separate oracles. Test values at and around rounding half-ties and clip endpoints, alternating extremes, cancellation, all-zero inputs, dimensions/tails used by all three members, and block boundaries.

VNNI, AVX2 signed-int16 products accumulated in int32, and scalar int32 must agree exactly within one compiled floating engine. Prove no int16 product or int32 accumulation overflow for actual signed ranges and matrix dimensions; an AVX2 packed-byte multiply-add that saturates intermediate signed16 is not equivalent. If VNNI uses unsigned-by-signed bytes and a correction term, verify that correction against a literal signed scalar oracle. Include negative products and maximal mixed signs. Integer associativity is safe only after bounding overflow.

The original member accumulation order and explicit per-ISA FMA versus separate multiply/add policy are preserved. Execution/scratch reordering must not change sequence reset behavior, row advancement on need_prediction=false, output clipping location, or any member's mutable state. Synthetic complete models need nontrivial weights and long streams; exact primitive kernels alone do not establish recurrent-model parity.

## ISA portability and qualification limits

The actor proposes keeping v3, avx2, and v2 floating-engine builds, each with integer dispatch. Require exact full-model backend agreement within each compiled floating engine. Cross-ISA floating trunk arithmetic is already different: FMA rounding can cross a later dynamic quantization tie and propagate through the GRU. Therefore cross-ISA full-model equality must not be claimed from exact integer kernels.

Before testing, freeze the proposed inherited cross-ISA screens per target: p99 absolute error <= .05, relative RMSE <= .03, and maximum absolute error <= .5, with the exact relative-RMSE denominator and aggregation declared. Report per-target maxima and long-prefix/reset results. These screens are compatibility checks, not a proof of the -.0001 WP cutoff on another ISA. Do not reinterpret a failed screen by adding clipping or changing the quantizer.

Measured VNNI runtime and quality qualify that path only. Non-VNNI paths may retain the same scientific candidate while remaining unqualified for the competition's time budget and for cross-ISA quality. Root must explicitly accept that readiness limitation or require further prospectively allocated evidence. No extra backend WP queries are currently authorized. CPU dispatch must test every instruction used by a binary, including the exact VNNI instruction family, and fail clearly if no supported binary exists; import success on the development CPU is insufficient.

## Required independent evidence before empirical allocation

1. Frozen source/build/toolchain/config manifests and unchanged quantizer source comparison; original engines remain immutable.
2. Independent scalar quantizer/dot-product oracle with adversarial boundary/tie/range tests, all kernel shapes and remainders, and all supported integer backends.
3. Complete synthetic stream comparisons within each floating build, all three ISA builds' compatibility screens, explicit reset/prefix/flag tests, and runtime dispatch capability tests that do not execute unsupported instructions.
4. Fresh CPython 3.11 scorer-userland import/callback smoke from an empty working directory, no undeclared dependencies, complete ZIP hash, and correct fixed three-member manifest.
5. Reviewed frozen real-fixture timing plan before any empirical fixture read. Root alone grants timing allocation and later quality allocation after runtime passes.

No empirical input, checkpoint, prediction, target, fit, or score was read for this note.
