# Independent review request — F3/seed1001 Beleriand preparation

Review this unadmitted packet statically. No staging, execution, resource admission, or source edits are authorized by this request. The root task owns the later freshness checks and any separate admission.

Please verify:

1. `F3_SPLIT_METADATA_V1.json` binds the exact B `splits/fold3.json` path, raw SHA/size, four-fold identifier, ordered membership digests and counts (10,539 fit-train; 1,368 fit-valid; 505 evaluation groups / 145 evaluation components). The source entries are manifest metadata, not whole-payload rehashes.
2. The seven steps mirror F1: wide/deep stage1, wide/deep six-epoch continuation, wide/deep fixed epoch5 full-valid export, then paired official grade. Every attempt, runtime receipt, snapshot, run, prediction and grade path is F3-specific and unique; no F1 paths are reused.
3. Only fold-specific manifest/valid-fold and unique F3 names/paths changed. The two architectures, seed 1001, fixed six-epoch schedule, exact canonical trainer/source/runtime, FP32/TF32-off, guard, resource shape and grader remain the frozen RANK01 recipe.
4. Static count arithmetic is correct: 10,539+1,368=11,907 selected groups; floor(11,907/8)=1,488 batches/epoch; 10 chunk updates/batch; 14,880 updates/epoch; 29,760 in stage1 and 89,280 total. Confirm the plan requires runtime stage1-contract verification before continuation.
5. The plan remains `UNADMITTED`, says no execution occurred, and makes fresh ownership, split/data/source/runtime, resource, disk and collision checks root admission gates. It does not infer that F3 is currently unclaimed from the authority snapshot, and does not conflate F2 issue16 with F3.

Return a review receipt bound to the exact plan and split-metadata SHA-256 values.

## Immutable review targets

- Plan: `F3_BELERIAND_PLAN.json`, SHA-256 `b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8`.
- Split observation: `F3_SPLIT_METADATA_V1.json`, SHA-256 `5facd5e512dd8782009ea6630bc42a560ed3f974b513d419b7fa9f0257fc99a9`.
- Source split manifest SHA-256: `21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f`.
