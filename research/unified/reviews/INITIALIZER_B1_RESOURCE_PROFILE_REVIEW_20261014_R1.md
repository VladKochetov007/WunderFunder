# Initializer B1 resource profile review R1

This is a read-only review of the frozen initializer-only CUDA resource-profile packet at student worktree commit `c85bbc5251372d1dd7701db047a75e3d3147f86a`. It is a source-admission review, not a profile result: no CUDA launch, fit, grade, provider action, or quality assessment was performed.

## Frozen sources

- [Profile README](../../campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R1_README.md)
- [Profile contract](../../campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R1_CONTRACT.json), SHA-256 `588507017c4a63e12070f2018a7512d2fdde4ea8d2a8b76d51b81659fd71f703`
- [Profile harness](../../../../worktrees/student-initializer-profile-20261014/scripts/profile_initializer_frontier.py), SHA-256 `e8b43b167655e89005fe3ee6f11833e09efefaa1c7de92100eac24cd6d4c2cb0`
- [Current CPU preflight R3](../../campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R3_PREFLIGHT.json), SHA-256 `8f729b44e80b64ccc8510cc2f07440cee91270891991d44345610154a7497863`

The contract pins the H256×2 seed-3 fixed-epoch-0 student checkpoint, its fit-validation and provenance, the clean homogeneous F0 cache, propensity source, teacher ancestry, split, normalization and sampler identities. Static checks confirmed the declared source hashes. The CPU preflight records no GPU launch and no quality claim.

## Recipe and data-boundary checks

The profile obtains the first actual B8 epoch-0 sample from the pinned B8 helper, which verifies the sampler order, source-qualified split identities, and overlapping `row_group=37` mappings before the harness takes `batch[:1]`. The profiled member is `(valid, 37, 44,346,154)`; the same numeric group ID also exists in train, so the explicit source identity is necessary. Runtime repeats the strict producer cache validation and full target-file verification.

The target and weight functions match the pinned trainer semantics for this sequence: clipped label/teacher target with `alpha=.9`, the production F0 propensity and true-mask row weights, clipped-label magnitude multiplier, and rows 0–98 excluded from loss. The helper's summed numerator and total full-sequence weight denominator are an explicit recipe change from the ordinary per-chunk denominators, as is one optimizer update per complete sequence rather than ten detached chunk updates. These changes are disclosed, so the resource smoke cannot be read as a fit of the ordinary recipe.

The implementation keeps the backbone frozen while leaving its forward graph available to initializer gradients, checkpoints the recurrent graph chunk by chunk, and checks finite loss and gradients, a nonzero initializer gradient, a real parameter update, unchanged backbone tensors, full/chunk parity, sequence reset, checkpoint reload, and finite contiguous FP32 export arrays. Export field names and array shapes align with the existing native parser, but this profile does not exercise native parsing or inference on an actual updated payload.

## Blocking defects in the frozen packet

1. `execute_profile` accepts `memory.max` only when it is exactly 5,000,000,000 bytes or unavailable. Cgroup page rounding can represent that declared cap as 4,999,999,488 bytes, causing the admitted worker to reject its valid reservation before profiling. The guard should validate a finite cap no greater than the reservation (and rely on the controller receipt for the reservation identity), rather than compare byte-for-byte with an unrounded decimal value.
2. `make_base_model` moves the backbone to CUDA, but `profile_mode` then constructs `InitializedGRU` without moving the wrapper. The newly created constant or predictor parameters therefore remain on CPU while input and hidden tensors are on CUDA. The learned arms fail on their first forward. The checkpoint-reloaded wrapper has the same missing device move.
3. The initial learned-arm equivalence check compares prediction SHA-256 values exactly against a zero-state full-sequence reference. The prefix mode necessarily segments the GRU at row 99 before adding its zero-initialized residual; CUDA GRU arithmetic across segmented versus unsplit calls is expected to be tested numerically, not assumed bitwise identical. Use an explicit tolerance and record the maximum absolute difference.

These are independent blockers to the exact frozen launch packet. They do not invalidate the cache, teacher, normalization, sampler, or target ancestry checks.

## Resource and verdict

The declared resource basis is a measured B8 host peak of 3,151,912,960 bytes plus 100,000,000 bytes of initializer-specific allowance, leaving 1,748,087,040 bytes against a 5,000,000,000-byte host reservation. The packet correctly labels CUDA activation/workspace demand as unknown, caps PyTorch allocation at 3.5 GB and observed VRAM at 5 GB, and leaves actual admission with root. This estimate supports considering one bounded profile after the three source defects are repaired; it is not a measured initializer peak or a full-fit safety proof.

**Verdict: HOLD on c85.** Do not launch this frozen packet. Review a sealed successor against these three defects, then make a fresh controller-owned GPU admission. No initializer quality or architecture-selection claim follows from this review.
