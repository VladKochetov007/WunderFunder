# L4 teacher capacity: stage-2 source-index mapping correction R2

This R2 amends [STAGE2_HANDOFF_R1.md](STAGE2_HANDOFF_R1.md). R1's target
formula and teacher/student recipes are unchanged. The correction is required
because `cache/train_targets.npz` contains 10,607 source-index groups, while
the official chronological training split and qualified PR60 target contain
only the 10,325 fitting groups.

## Correct row mapping

`validate_chronological_train_index_order` validates the complete 10,607-group
source index. The student target remains in the exact ordered
`split.train_groups` sequence of 10,325 groups. For each native GRU:

1. `scripts.build_distill_cache._inspect_teacher` validates the prediction bank
against all 10,607 source-index groups and the prediction sidecar/checkpoint.
2. R2 builds `fit_source_positions` by mapping each official fit group to its
   position in the complete prediction bank.
3. It gathers those mapped rows in small chunks and streams them into the
   10,325-row target accumulator. It never uses the first 10,325 rows as a
   proxy for fit membership.
4. The state and final `target_contract` record the full-index row count/hash,
   ordered fit row groups, explicit source-position list, and mapping SHA256.

The 282 source-index groups not present in `split.train_groups` are excluded
from distillation. Transformer exports already contain only the ordered 10,325
fit groups, so their streaming path is unchanged. The PR60 cache's
`main_source_train` and `families_train` order must match the 10,325 official
fit groups, not the 10,607 source index.

R2 uses state/cache schema `wnn.l4-teacher-stage2-stream.v2` /
`l4_teacher_capacity_stage2_stream_v2`; do not resume an R1 partial with R2.
The root preserved the R1 initialization failure receipt. It failed before the
accumulator was created, and no prediction array was written or removed.

## Updated disk bound

| Stage-2 object | Bytes |
| --- | ---: |
| 10,325-row target/accumulator | 1,652,000,128 |
| Native 10,607-row prediction bank | 1,697,120,128 |
| Transformer per-sequence bank, conservative bound | 1,654,643,328 |
| Accumulator plus largest incoming bank | 3,349,120,256 |
| Both final train targets | 3,304,000,256 |
| Both empty valid headers | 256 |

The target/cache and 3 GB free-floor policy from the contract still apply.
R2's initialization check now budgets against the full native bank. Hard-linked
PR60 auxiliary files consume no new physical space; each new teacher bank is
removed only after its validation/hash receipt is written.

## Checks and remaining work

Ruff format/check and Python compilation passed. The new CPU test
`tests/test_l4_teacher_capacity_stage2_row_mapping.py` passed all three cases:
non-prefix row selection, duplicate source rejection, and missing fit-group
rejection. This verifies the mapping logic only. Root owns the actual node
initialization and artifact checks; independent R2 review is required before
using the new source snapshot for target construction.
