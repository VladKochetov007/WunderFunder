# Stage-two target cache independent review R1

## Verdict

**Approve CPU cache initialization only** against the exact qualified PR60 cache.
The frozen source has no material blocker for `init`. Teacher `add` and
`finalize` remain conditional on actual node artifacts passing the checks below;
the four student fits remain gated on a dedicated stage-two target lineage and
output-budget receipt.

## Reviewed source

The source files match the builder's frozen handoff hashes:

| File | SHA256 |
| --- | --- |
| `stage2_stream_common.py` | `1067be5942012374f28c755fc7374464821b4b518a44680da75b9164f48b149a` |
| `stage2_stream_finalize.py` | `b2108be47a63e4e3806142fac30f5d3a9d63448304b130277d77b5c8ae984ba9` |
| `stage2_stream_native.py` | `10a3029dbf477340bd54c0eadf441372494a5a1dc8e3d6ab5ab3f75d43e489b` |
| `stage2_stream_teachers.py` | `316b7fc10734f326cd99b8b3fd0b02ceed70d5c09b2976b308b15e92cb03790` |
| `stream_stage2_targets.py` | `0c7b98fc18b9160b912871885937d9ee77c0596b6550e02609e3a2d4f26a5c43` |
| `STAGE2_HANDOFF_R1.md` | `f2da733ca4c81d54d5d429bff3fca5b231cfa6f061c80fb4c4aa8d60b0258c82` |

## Checks

| Invariant | Static result |
| --- | --- |
| Official chronological split SHA, exact canonical 10,325 train index order, zero fitting-valid groups | Pass |
| Exact PR60 base cache eligibility | Pass: metadata, main train, and auxiliary train hashes are pinned to the sibling byte-audit receipt |
| GRU teacher source, fixed epoch 3, train export and recursive fitting-only ancestry | Pass by reuse of `_inspect_teacher` and `_audit_ancestor_closure`; actual node artifacts not yet reviewed |
| Transformer R4 source, full normalized config, approved profile identity/device and fixed epoch 3 | Pass by static source inspection; actual node artifacts not yet reviewed |
| Transformer fit-only source and normalization ancestry | Pass: exact event211 schema, official fit/evaluation members, train source fingerprints, all-fit normalizer identity and moments; initializer and teacher ancestry must be null |
| Teacher prediction membership, order, shape and dtype | Pass: exact train order, no valid teacher rows, finite FP32 `[20000,2]` sequence exports; GRU exports are checked by the canonical cache inspector |
| Target equation | Pass: fixed family order, per-teacher clipping to `[-2,2]`, ordered FP32 sum/mean, then `.2 * .8125 * mean + .8 * T_old` |
| Native W18 cache compatibility | Pass: finalizer writes expected array records, zero-row valid targets, retained auxiliary arrays, and calls `validate_distillation_cache` |
| No incorrect old-family relabeling | Pass: legacy family slots and source coefficients remain scoped to the retained PR60 base/auxiliary arrays; new teachers are separately recorded under `target_contract.teachers` |

The generic W18 cache validator checks structure and retained-array identity;
it does not validate the new `target_contract.teachers` ancestry. The dedicated
receipt must therefore bind each actual teacher checkpoint, full config and
profile, split/membership, source and normalizer identities, strict precision
and FP32 checkpoint tensors, recursive parents, prediction hashes, the exact
PR60 base hashes, formula, and finalized target hashes. It must treat copied
`sources` and `source_coefficients` as the old target's lineage only.

The old auxiliary arrays are hard-linked from the exact base cache and are not
written by the builder. The native W18 dataset reads cached targets through
read-only mappings. No teacher-bank fan-out is retained; each bank is
checksummed before the new bank files are removed.

## Machineable admission criteria

1. `init` may proceed only if the observed input cache hashes equal the pinned
   PR60 metadata/main/family hashes, the split/index binds to 10,325 train and
   zero fit-valid groups, and current free space after the reserved accumulator
   plus one incoming bank remains at least 3 GB.
2. Each `add` requires the next exact run in its frozen family order, fixed
   epoch-3 checkpoint, train-only membership and export, finite FP32 targets,
   and byte-bound checkpoint/config/profile/source/normalizer evidence. For
   Transformers, use the actual R4 panel and verify the approved profile receipt
   and FP32 tensor/runtime evidence from the checkpoint.
3. `finalize` requires every expected teacher exactly once, no pending run,
   unchanged base metadata/main hashes, exact old auxiliary array hashes, finite
   FP32 output arrays, exact fit group ordering, and successful native W18 cache
   validation.
4. Student admission requires the separate stage-two graded receipt to verify
   the new target equation and ancestry from the actual teacher artifacts,
   unchanged PR60 FBC seed-5 epoch-3 initializer/preprocessing, fixed student
   recipe, and actual total output/free-space budget. The ordinary W18 cache
   validator alone is insufficient for this step.

## Scope and limitations

Reviewed the frozen source and handoff read-only. No tests, profiles, inference,
SSH/GPU actions, cache initialization, teacher aggregation, or scientific fits
were run by this reviewer. Actual-node checks remain pending until each teacher
artifact is available. Validation is previously exposed development evidence.
