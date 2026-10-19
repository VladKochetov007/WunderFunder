# Streaming Transformer students versus GRU under one teacher

**Proposal only.** No local tests, training, inference or profiles were run. `launch_ready=false`; the missing integration and asset bindings are listed explicitly.

- [PR.md](PR.md): mechanism, controls, CP30, decision and receiving-host handoff.
- [recipe.json](recipe.json): named arms, all30fold/seed cells per arm, precision, proposed hyperparameters and null launch bindings. This is a specification, not trainer CLI input.
- [Canonical rationale](../../unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md): Section25, plus shared cache/lineage/statistics Sections22–27.
- [CPU inference path](../../unified/inference/TRANSFORMER_DISTILLATION_CPU_PATH.md): export/state/runtime/ZIP gates, separate from scientific FP32 comparisons.

IDs: `H-STREAMING-STUDENT-DISTILLATION-20261019` / `E-STREAMING-STUDENT-DISTILLATION-20261019`. Draft base: `research-team`. Source snapshot inspected: `3cebc9f0a0e375c71651115ca43bad129af857e8`; actual implementation/run source must be bound at admission. Root owns this design; receiving researcher claims individual cells before execution.

PR #24 compares standalone D256 architecture models. This compares compact D128 streaming students with and without one identical teacher, plus deployment semantics. Shapes are not claimed parameter/FLOP matched.

180 student fits;30 L teachers/caches unless identical prior sources are reusable. Teacher inference is offline preparation; deployment executes only the student. No quantization in fits.

Publication and result fields remain empty until actual receipts exist. No plan printer, queue or placeholder trainer is supplied.
