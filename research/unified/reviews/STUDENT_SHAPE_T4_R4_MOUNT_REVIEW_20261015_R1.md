---
kind: review
status: complete
created_utc: 2026-10-15
updated_utc: 2026-10-15
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: []
evidence_paths: []
---

# Root independent review of R4 target-cache mount repair

GO for the same bounded profile rerun, conditional on fresh quota and private/READY/version checks. Root reviewed the cloud actor's isolated packet; root did not build it.

Direct comparison of both R3/R4 job JSONs finds exactly two changed top-level fields: `inputs` gains the existing train/valid target-cache descriptors, and `environment.WNN_DEFERRED_IDENTITY_PATHS` gains those paths. All other environment fields and complete training argv remain identical. Each job now declares24 inputs. Existing private fold0 dataset v1 supplies both content-addressed files; no target recomputation or payload upload is needed.

Independently opened both small tar archives and hashed every member with a2MB/member bound. Only the two embedded job JSONs differ; all non-job members are byte-identical. R4 archive272,182 bytes, SHA ae9f7bc635e75b1468c136fe929d1e9f02f6d1e87e0872073ffcc0b7bdd6f9e0. This changes input declarations, not training source.

Read the actual split loader and portable verifier in the frozen worktree. The loader requires both target-cache paths. It checks the portable registry's original split identity, exact size and whole-file SHA, then separately checks `row_group`/`seq_ix` index SHA. Deferring the runner's duplicate hashes does not bypass the trainer's content/index check. This review did not read model/data payloads or use a GPU.

Actor's offline preflight also binds24 outer mounts, both embedded specs, the unchanged68-file code closure and1800s outer deadline. Kernel R4 SHA e4610b3989ada343b47c220f09412f29910ec656feaf910c44fbbb1bd065baa2. Fresh provider checks remain the operator's responsibility. R3 failed before CUDA/update evidence; this is an execution repair, not shape-quality evidence.

Canonical packet: `research/campaign_2026-10-15/student_shape_pair_f0_r4/` in `/home/vlad/development/wunder/worktrees/student-shape-pair-kaggle-r1-20261015`, pending integration. Reviewed artifacts: `OFFLINE_SPEC_PREFLIGHT_R4.json`, `INPUT_BINDING_R4.json`, `SOURCE_BINDING_R4.json`, both job JSONs and `KERNEL_MOUNT_SUCCESSOR_R4.json`; source paths `src/wnn/splits.py` and `src/wnn/portability.py`. Preserve the R3 terminal failure and earlier transport history during integration.
