---
kind: experiment
status: complete
created_utc: 2026-10-14T00:11:07Z
updated_utc: 2026-10-14T00:11:07Z
hypothesis_ids: [H-STATE-INIT-20261014]
experiment_ids: [E-STATE-INIT-IMPLEMENTATION-20261014-R1]
evidence_paths: [INITIALIZER_TESTS_R1.xml]
---

# Causal initial-state wrapper: implementation evidence

## Question or claim

Can the four planned initializer arms share the existing GRU backbone and obey the same causal state/reset contract? The research wrapper is implemented and ten focused checks pass. No initializer has been fitted or graded yet.

## Mechanism and method

[`src/wnn/state_initialization.py`](../../../src/wnn/state_initialization.py) wraps the unchanged linear-encoder, dropout-zero `RNNModel` with zero state, a learned constant, a first-row MLP, or a prefix residual. The MLP has 64 GELU units and a tanh output bounded by a declared positive amplitude (default 1). Each learned arm starts at the zero-state baseline through a zero constant/output layer. This initialization is an explicit implementation choice, not a trained result.

The prefix arm accumulates Welford mean, population standard deviation and the last normalized feature vector over rows 0–98. It processes all those warmup rows normally, adds the residual immediately before row 99, then discards the statistics. Only the 211 causal feature inputs enter the wrapper. Targets, IDs and masks are absent from its API. Passing `None` resets hidden state, position and statistics together. Caller-owned state allows independent sequences without shared mutable buffers.

`InitializationState.detach()` makes the TBPTT boundary explicit. Initializer-only adaptation freezes backbone parameters, while retaining the graph through the frozen recurrence; using `no_grad()` for that recurrence would incorrectly prevent initializer learning. The current baseline trainer/exporter is unchanged. Trainer integration, matched fit commands, native support and checkpoint/export schemas remain separate delivery work.

## Evidence and interpretation

[`tests/test_state_initialization.py`](../../../tests/test_state_initialization.py) tests zero-baseline equivalence for all four arms; full/chunk parity with nonzero initializers, including boundaries 98/99/100; complete reset replay; the exact prefix context and row-99 injection; unchanged predictions before modified future inputs; gradients through a frozen backbone; and finite 20,000-row execution. [JUnit receipt](INITIALIZER_TESTS_R1.xml): **10 passed in 2.21 seconds**, one CPU thread, 2GB host limit, no GPU fit. Ruff format/check pass on both new files.

Chunk/full checks use `rtol=1e-5, atol=1e-6` for PyTorch GEMM shape differences. Same-call reset replay is exact. These tests use small random backbones and synthetic inputs to probe implementation properties; they are not full native parity, actual-data stability or prediction-quality evidence.

## Limits and validity

No teacher, label or evaluation group was used for fitting here. Existing folds remain exposed for future experiments. The finite long-sequence test uses H8×2, not the final selected width. The first actual-data arm must profile its real batch/chunk and share frozen normalization, target cohort, sample order and optimization budget with zero-state control. Native export is required before serving promotion.

## Decision and next step

Keep this arm independent of target-host CPU uploads. Integrate joint and initializer-only fitting after the initial ladder consumer is delivered; record the actual gradient horizon. Grade the official metric and the plan's position bins, then measure hidden-state and prediction perturbation decay separately. Independent implementation review is pending.

## Source records

[Full plan](PLAN.md), [theory and precedent](THEORY_AND_DISCRIMINATING_TESTS_R1.md), [contract](CONTRACT.json), and H-STATE-INIT-20261014 in the append-only hypothesis ledger.
