> Extracted from the final October 12 handoff. Its original SHA-256 was `e033f8d52f4d4444e5f0c5fa006c2b4f56b6e82f7187e2ed9fb1b74b20e6f704`. See [archive index](../README.md) and [split manifest](../SPLIT_MANIFEST.json). Local links were rebased for this page.

## Serving engineering, compression, and package validity

All final Q8 ZIPs use an independently built and reviewed precision-aware
native runtime and callback wrapper. The serving path resets recurrent state,
returns None during warm-up, emits the two required finite outputs for scored
rows, and replays deterministically. Exact archive tests are compatibility
checks; they do not replace full official scoring of the exact package.

The reviewed native single-member full-fold Q8 control completed F0/F1 quality
evaluation as listed above. Earlier packet generations found real blockers:
Python 3.11 dependencies missing in the system interpreter, absolute paths
bound to old worktrees, incomplete footer identity enforcement, and CLI source
transport that did not include sidecars. These were preserved and repaired in
identified revisions; static preparation passes were not treated as runtime
passes. The exact R6 admission and native source receipts are indexed under
[native single-member clean-fold Q8 R6](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r6/ROOT_ADMISSION_R6.json).

October-12 native CPU work tested callback-preserving changes. Activation
fusion and scaled projection precomputation passed output/reset parity but
paired timing directions disagreed; PGO was slower in paired checks. Shared
frozen backbones with multiple heads substantially reduced local callback cost
with bit-identical outputs; their quality remained fold-conditional, so the
default is off and this did not independently qualify a final package.
Raw-prefix repetition screening found zero exact adjacent repeats in 199,990
pairs across ten sequences, which defers only that exact cache opportunity.
The current instruction is to exploit actual Q8 headroom before Q4; no E
runtime guarantee follows from Intel callback timing. Complete methods,
artifacts, independent reviews, and unstarted native ideas are in the
[Zen4/native serving research index](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/zen4_cpu_research_20261012_r1/INDEX.md).
