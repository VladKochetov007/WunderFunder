# Causal teacher hints beyond two-output distillation

**Current record: a prospective, unrun four-fit amendment.** The shared executable runtime is implemented at [PR #32, source d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6](https://github.com/VladKochetov007/WunderFunder/commit/d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6), but this PR29 panel has not been profiled or fitted. Launch readiness remains false; no new result or B300 acceptance is claimed.

- [Current amended PR body](PR.md)
- [Runtime amendment](RUNTIME_AMENDMENT.md) and [machine-readable amendment](runtime_amendment.json)
- [Frozen original CP30 PR body](PR_CP30_FROZEN.md): byte-identical copy of the prior PR text.
- [Historical CP30 recipe](recipe.json): preserved byte-for-byte; it is not a runtime CLI config.
- Shared runtime dependency: [PR #32](https://github.com/VladKochetov007/WunderFunder/pull/32), pinned to [d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6](https://github.com/VladKochetov007/WunderFunder/commit/d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6).
- [Canonical rationale](../../unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md), especially Section 24.
- [CPU inference contract](../../unified/inference/TRANSFORMER_DISTILLATION_CPU_PATH.md), a separate release gate.

## Current amended panel

Across F0/student seed 1001 and F2/student seed 1002, fit two aligned-hint students and two component-permuted-hint students. Reuse the same-fold T768×24 teacher/cache and GRU256 output-KD endpoint for each cell from PR28. This costs **four new student fits**, with no new teacher fits; it does not launch the historical 90-fit CP30 panel.

The two student arms use additive clipped-label WMSE + output-KD WMSE + 0.1 hidden-hint MSE, with KD weight 1.0. The aligned arm matches teacher/student rows. The negative control rotates fitting components while keeping row positions and output-KD targets aligned. The current runtime uses raw112 inputs, a 512-row-per-layer teacher KV window, strict FP32 and three fixed epochs. These are explicit amendments to the historical anchored-window, alpha-mixture design.

Historical CP30 scope and rationale remain available in [PR_CP30_FROZEN.md](PR_CP30_FROZEN.md) and [recipe.json](recipe.json); neither has been rewritten. See [RUNTIME_AMENDMENT.md](RUNTIME_AMENDMENT.md) for pinned config hashes, reuse checks, commands and pending acceptance gates.

## Historical CP30 specification

The original CP30 design was a proposal for 90 student fits across 30 cells and three arms, using an L207 teacher, anchored 4096-row recomputation, eight-epoch students, and an alpha=0.9 label/KD mixture. It was not executed. Its exact recipe remains in [recipe.json](recipe.json), and its original PR text is preserved byte-for-byte in [PR_CP30_FROZEN.md](PR_CP30_FROZEN.md). The current four-fit amendment above supersedes that design for the prospective screen; the historical specifications are not runtime configs.
