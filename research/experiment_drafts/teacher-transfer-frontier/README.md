# Teacher scale × GRU capacity transfer frontier

**Proposal only.** No local tests, training, inference or profiles were run. `launch_ready=false`; the missing integration and asset bindings are listed explicitly.

- [PR.md](PR.md): mechanism, controls, CP30, decision and receiving-host handoff.
- [recipe.json](recipe.json): named arms, all30fold/seed cells per arm, precision, proposed hyperparameters and null launch bindings. This is a specification, not trainer CLI input.
- [Canonical rationale](../../unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md): Section22, plus shared cache/lineage/statistics Sections22–27.
- [CPU inference path](../../unified/inference/TRANSFORMER_DISTILLATION_CPU_PATH.md): export/state/runtime/ZIP gates, separate from scientific FP32 comparisons.

IDs: `H-TEACHER-TRANSFER-FRONTIER-20261019` / `E-TEACHER-TRANSFER-FRONTIER-20261019`. Draft base: `research-team`. Source snapshot inspected: `3cebc9f0a0e375c71651115ca43bad129af857e8`; actual implementation/run source must be bound at admission. Root owns this design; receiving researcher claims individual cells before execution.

This extends PR #3 with a predeclared teacher-scale by student-capacity interaction; the clean GRU and labels retain practical baselines. It is not a repeat of generic teacher-family selection.

240 student fits; up to90 teacher fits (30 each clean GRU/L/XL) if no exact eligible sources exist. Teacher source per cell is shared between student sizes. No8GPU-hour allocation is implied.

Publication and result fields remain empty until actual receipts exist. No plan printer, queue or placeholder trainer is supplied.
