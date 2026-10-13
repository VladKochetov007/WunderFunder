---
kind: experiment
status: proposed
hypothesis_ids: [H-STRATEGY-20261012-01, H-238]
experiment_ids: [E-STRATEGY-INNOVATION-F0-S3-20261013-R1]
evidence_paths: [RECIPE_R1.json, PREFLIGHT_R1.json, MOMENTS_CONTROL_COMMANDS_R1.json, INNOVATIONS_COMMANDS_R1.json]
---

# Matched causal-innovation F0 seed-3 pair

## Comparison

Both arms append 64 causal moment channels and 16 innovation channels to the same own F0 seed-3 initializer. The control zeros the final 16 channels; the treatment keeps them. Parameter count, 2,000-row optimizer/detach clock, batch16, teacher cache, target amplitude0.8125, alpha0.9, propensity, auxiliaries and six-epoch OneCycle schedule match. Epoch5 is fixed before outcomes. This tests the added innovations conditional on the moment representation. It does not yet test moments versus raw inputs.

The [recipe and source hashes](RECIPE_R1.json), [control commands](MOMENTS_CONTROL_COMMANDS_R1.json), and [treatment commands](INNOVATIONS_COMMANDS_R1.json) bind a new matched pair from Main-derived source. The older frozen R3 control is context only: the new trainer differs from that historical source. Both new models use the same fitting-only prior: a deterministic sample of 32 source-qualified fitting groups, sampled every64 rows. New input weights are zero and added normalization coordinates are mean0/std1; existing parent preprocessing is inherited. The effective capacity of the zeroed control remains smaller despite equal parameter count.

## Actual-producer evidence

[Preflight](PREFLIGHT_R1.json) loaded the real own parent `c20261003_combo_s3_f0@ep3`, SHA `d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e`, and extended its 211 inputs to291. Its predictions were exactly unchanged on the declared short synthetic input fixture. This verifies initialization, not model quality or full-sequence feature numerics. Earlier [implementation and parity evidence](../INNOVATION_IMPLEMENTATION_R1.md) covers a real 20,000-row causal feature path, and its [independent review](../reviews/innovation_implementation_r1/README.md) covers CPU/streaming/ONNX behavior.

The new feature-aware prediction script now supports campaign fit, evaluation-only and full-valid feature-only selection. Six selector tests pass. Actual F0 source checks yield10,535 fit-train,1,405 fit-valid,468 evaluation and1,873 full-valid groups; wrong-fold checkpoint bindings are rejected. Training and export read assets from `WNN_PROJECT_ROOT`, while source hashes refer to the immutable `SOURCE_ROOT`. This operational separation reuses Main's physical assets without making source provenance depend on later Main edits. Both scripts retain their prior default root when no override is set.

The [draft CLI failure](CLI_DRAFT_FAILURE_R0.json) preserves four unsupported identity-valued options inherited from R3. Main's cached-distillation branch fixes those settings in its equations; the draft options were removed after inspecting its clipped-original-label weights, MSE numerator and chunk denominator. The [actual-parser check](check_cli.py) and [receipt](CLI_PREFLIGHT_R1.json) parse all six staged fit commands, two profile commands and two exports, stopping before any data/device work. This checks the actual consumer API rather than assuming similar source versions share flags. No failed draft reached fitting.

## Delivery and limits

Root owns the pair from preparation through admitted profiles, fitting, predictions and paired grading. Source worktree: `/home/vlad/development/wunder/wnn-agent-worktrees/strategy-innovation-fit-20261013`. Keep it intact until all workers finish. The recipe binds every production Python source there; any source correction requires a new reviewed binding before admission. No existing running source or controller was changed.

Two profile-only commands use the actual batch16/20,000-row/291-input shape, five warmup and five measured complete batches, the full six-epoch scheduler geometry and fresh disposable run names. They save no resumable training state and carry a20-minute timeout with30 seconds for termination. The 6GB worker reservation includes conservative margin over the historical4.95GB baseline and new feature/prior allocations; it is a proposed bound, not evidence of fit. Actual device profiles must pass before either fit. Use the shared durable controller and the single GPU lock after the current amplitude chain; no direct concurrent launch.

All P1/P2/P3 resumes require actual predecessor hashes and full optimizer/RNG/scheduler checks. Export uses FP32 and default filenames matching the existing grade consumer. Each full-valid NPY payload is299,680,000 bytes before its header. F0 is repeatedly exposed, this pair has one fitting seed and one shared teacher cohort, and no score or future-public forecast exists. F3 awaits its own clean teacher cache and matched controls. Promotion still requires cross-fold/seed endpoint evaluation and measured serving cost.
