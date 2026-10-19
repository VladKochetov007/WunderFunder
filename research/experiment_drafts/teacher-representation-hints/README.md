# Causal teacher hints beyond two-output distillation

**Proposal only.** No local tests, training, inference or profiles were run. `launch_ready=false`; the missing integration and asset bindings are listed explicitly.

- [PR.md](PR.md): mechanism, controls, CP30, decision and receiving-host handoff.
- [recipe.json](recipe.json): named arms, all30fold/seed cells per arm, precision, proposed hyperparameters and null launch bindings. This is a specification, not trainer CLI input.
- [Canonical rationale](../../unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md): Section24, plus shared cache/lineage/statistics Sections22–27.
- [CPU inference path](../../unified/inference/TRANSFORMER_DISTILLATION_CPU_PATH.md): export/state/runtime/ZIP gates, separate from scientific FP32 comparisons.

IDs: `H-TEACHER-REPRESENTATION-HINTS-20261019` / `E-TEACHER-REPRESENTATION-HINTS-20261019`. Draft base: `research-team`. Source snapshot inspected: `3cebc9f0a0e375c71651115ca43bad129af857e8`; actual implementation/run source must be bound at admission. Root owns this design; receiving researcher claims individual cells before execution.

The canonical note deferred hidden alignment. PR #3 changes teacher family; this fixes teacher/student/output KD and changes only fitting-time representation supervision. It is not relation-KD or assistant KD.

90 student fits,30 L teachers/caches unless exact grid sources are reusable. A64-channel FP32 hint is32times the two-output cache payload; measure I/O cost.

Published draft: [PR #29](https://github.com/VladKochetov007/WunderFunder/pull/29) targeting `research-team`; initial packet head `0d8c7b537a2d8657260b76443dbf3ec628c73cc4`. Result fields remain empty until actual receipts exist. No plan printer, queue or placeholder trainer is supplied.
