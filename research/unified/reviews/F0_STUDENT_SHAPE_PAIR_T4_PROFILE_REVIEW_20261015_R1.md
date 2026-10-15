---
kind: review
status: complete
created_utc: 2026-10-15T15:41:49Z
updated_utc: 2026-10-15T15:41:49Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - ../../campaign_2026-10-15/student_shape_pair_f0_r1/README.md
  - ../../campaign_2026-10-15/student_shape_pair_f0_r1/GENERATED_KERNEL_SEAL_R1.json
  - ../../campaign_2026-10-15/student_shape_pair_f0_r1/SOURCE_BINDING_R1.json
  - ../../campaign_2026-10-15/student_shape_pair_f0_r1/INPUT_BINDING_R1.json
---

# F0 student shape-pair T4 profile review R1

## Question or claim

Does the frozen source, mount, and generated-kernel packet support the two bounded T4 resource profiles without admitting a fit or making a quality claim?

## Mechanism and method

I reviewed packet commit `4a663209e811ff368e6428cbb91e91b6f85034e1` in the student-shape-pair worktree. The runtime source tree is pinned to `6934fb7d37ab5e95d353c47b3cb12856bc52e723`; source files under `scripts/` and `src/` are unchanged between that commit and the packet commit. The contract SHA-256 is `815fe98fdecfafec84e4f5f4800aabb9b06bcf8dcfdff5efec057ddb1723363d`.

I independently checked the source-binding SHA-256 `2e47a541b01cf2f30bda2f5631ab9867a93b1e39cb6be8ad41e6b9f3b72edf15`, input-binding SHA-256 `6d69c3e5c24c08d2c0684a10e81e320729b9f8612414f082b5ad2b51ae3f244c`, and generated-kernel seal SHA-256 `59af86255a22c8493b4e64cacc52d9ae7c569eb73bda20d59cd44463296302e0`. The code aggregate covers 68 files; I rehashed those files and confirmed each job references the same aggregate. Both job JSON hashes and all 22 external input path/hash/size mappings match the input binding and generated bootstrap. The input closure includes the raw F0 dataset, split/propensity/base assets, three CP2 teacher checkpoints and provenance files, clean T3 cache, and CP5 normalization donor.

## Evidence and interpretation

The jobs map explicitly to trainer runs `student_pair_profile_f0s3_h256_l2_t4_r1` and `student_pair_profile_f0s3_h512_l1_t4_r1`. Each uses the `f0_seed3_t4_profile` stage, five B16 batches at C2000 in FP16, 50 expected chunk-level optimizer attempts, no evaluation, and no declared artifacts. The shape pair shares seed, targets, sampler, normalization, and optimizer settings. Each worker is limited to 600 seconds; the generated kernel launches one runner with both specs under a 1,800-second outer deadline. The runner derives concurrency from the measured T4 inventory and host capacity. The metadata selects one `NvidiaTeslaT4`, and the terminal receipt should confirm `concurrency_admitted` is 1.

The local archive is 272,052 bytes with SHA-256 `8c6d3de42979ed0964a925e20a891db29e7621841b5d571aa27eb286bf1d5b48`; the bundle manifest hash is `63c66e8ab24bf126ecb0acfa28885fce29bebee07c327ce66dd7d04e0840becb`. I confirmed the generated kernel hash `86a5ff9dee6f7adfd809697349d45af2d507efcc1bb41bf2a3f5794fa7e42440` and metadata hash `4153ead9afb88b16eb458617b16e60b1634deb6820f69a02d2c0418220c122f7`. Metadata pins the intended private T4 kernel, enables GPU, and disables TPU and internet. No provider action or profile run occurred.

## Limits and validity

This is source and packaging evidence only. It does not show that the provider datasets are currently READY/private, that their versions are still exact, or that the T4 worker passes cache/order/startup checks. Root must perform the fresh owner, privacy, readiness, and typed-quota checks before dispatch. No full fit, checkpoint, score, prediction, or quality comparison is admitted by this profile packet.

Two wording defects do not change the execution bound: the generated kernel retains an error string that says `6,000-second` although its active deadline is 1,800 seconds, and the README says donor weights are not loaded although the checkpoint is deserialized to extract buffers. The intended model claim is that donor learned parameters are not copied into or used by the student.

## Decision and next step

**GO for exactly the two serial T4 resource profiles**, conditional on root completing its fresh provider/quota gate. The profile supports cost and resource observation only. Do not interpret it as evidence for three-epoch runtime or student quality. Correct the stale error wording when convenient; it does not override the enforced 1,800-second deadline.

## Source records

- [Frozen student shape-pair packet](../../campaign_2026-10-15/student_shape_pair_f0_r1/README.md)
- [Generated kernel seal](../../campaign_2026-10-15/student_shape_pair_f0_r1/GENERATED_KERNEL_SEAL_R1.json)
- [Source binding](../../campaign_2026-10-15/student_shape_pair_f0_r1/SOURCE_BINDING_R1.json)
- [Input binding](../../campaign_2026-10-15/student_shape_pair_f0_r1/INPUT_BINDING_R1.json)
