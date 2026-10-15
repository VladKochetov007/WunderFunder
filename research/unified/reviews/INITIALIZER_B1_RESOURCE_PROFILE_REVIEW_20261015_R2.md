# Initializer B1 resource profile review R2

This is a read-only source-admission review of the sealed initializer profile at student worktree commit `92d5ce5402f44abadfd009bb0242ae75cbed0bdb`. It gives a bounded GO for the exact one-sequence CUDA resource smoke after the controller-owned admission. No CUDA profile, fit, grade, provider action, or quality assessment was performed by this reviewer.

## Frozen packet and checks

- [R2 profile README](../../../../worktrees/student-initializer-profile-r2-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R2_README.md)
- [R2 contract](../../../../worktrees/student-initializer-profile-r2-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R2_CONTRACT.json), SHA-256 `c07644f386fd0c6d6a42935d2856c68e17fc52498718ef21920b6783e8c62665`
- [R2 harness](../../../../worktrees/student-initializer-profile-r2-20261014/scripts/profile_initializer_frontier_r2.py), SHA-256 `f94c3274e2cc16eb1793d0ded7f4a8db39eda20eb7a2085608782e944eb1bdee`
- [Final standard-library preflight](../../../../worktrees/student-initializer-profile-r2-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R2_FINAL_PREFLIGHT.json), SHA-256 `3bc2ad2aba91e754495499c5f2338ab94bd6f3a5ea4be30239138f4d5ea9821d`
- [Focused CPU regression receipt](../../../../worktrees/student-initializer-profile-r2-20261014/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R2_CPU_TESTS.json), SHA-256 `021b22adbaa2fc7e38109518397341de3a9220fb012babe63a2351ff18db0857`

I rehashed the contract and the three source files named in it; all match the packet. The pinned trainer revision and source-file manifest match the control's recorded provenance. The actor's receipt reports a bounded metadata-only preflight and 10/10 meta-device/cgroup regression tests; I did not rerun PyTorch tests under this review's source-only budget.

## Recipe and implementation boundary

The packet preserves the fitted H256×2 seed-3 fixed-epoch-0 backbone and its fit receipt, provenance, split, mean/std buffers, clean homogeneous F0 cache, propensity source, teacher roster, and the B8 producer/order helper. At execution it revalidates the actual producer cache and source-qualified sampler order, rechecks the teacher/propensity ancestry, and selects the first sampler member `(valid, row_group=37, seq_ix=44,346,154)` from the B8 batch. The overlapping numeric group ID is resolved by source, and the split identity is pinned.

The target and weight path uses the pinned trainer's input/target/weight functions: clipped-label/teacher target with `alpha=.9`, F0 propensity and valid true-mask treatment, clipped-label magnitude weighting, and no supervision on rows 0–98. The helper's global complete-sequence weight denominator and one optimizer step over a graph carried across ten checkpointed chunks are explicit differences from ordinary per-chunk normalization and detached TBPTT updates. They remain accurately labeled as a one-update resource smoke, not a reproduction of the standard fit.

R2 repairs all three R1 launch blockers. The cgroup check accepts a 5,000,000,000-byte reservation rounded down by less than one page, every learned and reloaded wrapper is moved to the selected device and all parameter/buffer devices are checked, and split/unsplit zero initialization is compared with recorded `rtol=1e-5, atol=1e-5`; checkpoint reload remains exact on the same call. The profile freezes the backbone while preserving gradients through its forward pass, and checks finite loss/gradients/state, a nonzero initializer update, unchanged backbone tensors, full-versus-chunk parity, explicit fresh-state use, finite FP32 native arrays, and reload/export consistency.

## Resource scope and caveat

The host estimate is 3,251,912,960 bytes against the 5,000,000,000-byte reservation, based on the measured B8 terminal peak plus a stated 100 MB allowance. The CUDA workspace remains algorithm-dependent and is not proven by that estimate. The packet caps the PyTorch allocator at 3.5 GB and observed VRAM at 5 GB; its previous B8 result supplies context, not an initializer peak or full-fit safety proof. The exact launch command uses `bandit_resources.py`, which verifies the worker's cgroup membership, `MemoryMax=5,000,000,000`, and zero swap before running the profile.

One reporting caveat remains: `current_cgroup_bytes()` returns `None` when the cgroup limit cannot be read, and `validate_declared_host_limit(None)` accepts it; the CPU regression receipt explicitly tests that behavior. This does not defeat the exact admitted launch because the controller independently verifies the worker cgroup and hard limit. At terminal collection, require that controller receipt to show the exact reservation and reconcile it with `host_memory_max_bytes_at_start`; a null profile-side field alone is not proof of the cap.

**Verdict: bounded GO for the exact resource-only smoke under the controller's 5 GB host / 3.5 GB allocator / 5 GB observed-VRAM limits.** A successful profile would support implementation and single-sequence resource evidence only. It does not establish initializer quality, admit a multi-sequence fit, or select an initializer.
