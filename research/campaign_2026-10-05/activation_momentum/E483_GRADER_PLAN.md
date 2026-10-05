# E483 independent grader plan (registered-design only)

This plan was prepared from `E483_DESIGN.md` before inspecting E483 actor code. It is not an admission or score. Grader implementation waits for the actor's final source hash and the blind design critic's completion.

## Frozen checks before score computation

1. Bind the exact E483 design digest, sealed fit specification, attempt record, source/runtime hashes and cohort/export identities. Reject any mismatch, missing proof, non-finished/nonzero attempt, signal, failed teardown, or worker/parent OOM counter increase. Validate the exact 18 declared artifacts by path, size and SHA-256; refuse overwrite of grader outputs.
2. Require the complete 18-cell Cartesian product: seeds 271831/271832/271833 × fixed baseline/sine × recurrent Muon beta1 0.8/0.9/0.95. Verify 474 updates, B4×256, constant LR, unchanged AdamW betas, and all cells finished. Validate resolved optimizer parameter groups/names: only the four GRU gate matrices have Muon and the registered beta1; other groups retain AdamW `(0.9, 0.999)`. Check recorded all-update clipping counters are present and in range.
3. Independently verify common initialization within each seed across the six cells, the declared fit-group order, 24 fit versus 16 exposed score components, fit-only external normalization, identity normalization buffers, H64/two-layer topology, 42 inputs and 61,058 parameters. Reject any test-row training role.
4. Independently reconstruct the two-layer reset-after GRU candidate scan for `u=tanh(v)` and `u+0.1 sin(pi*u)`, including native gate/bias placement, full-sequence state handling and the frozen H64 concat GELU head. Strictly load each checkpoint and compare full sequence predictions on the exported evaluation inputs. The implementation and numeric tolerances will be registered before reading score arrays.
5. Validate raw target values and prediction tensors for finiteness before clipping; independently verify canonical need/score masks and row/component identities on all 16 previously exposed components. Recompute official per-component weighted Pearson and aggregate global/t0/t1 weighted-Pearson scores from raw outputs without importing actor scoring code.
6. Export the paired raw selected predictions, targets, masks, component IDs, row indices and arm map for separate numerical reproduction.

## Registered summaries

Report all 18 arm scores and all 27 registered global contrasts, maintaining the three fit seeds as separate strata:

- Sine minus baseline at each of three momentum values: 9 contrasts.
- Momentum 0.8 minus 0.9 and 0.95 minus 0.9 for each shape: 12 contrasts.
- The two shape-by-momentum interactions relative to 0.9: 6 contrasts.

For each seed, resample the same 16 numeric-sorted components jointly across its six arms for 2,000 draws. Preserve paired draws when computing every contrast and each target metric; never pool fit seeds or select a winner using the exposed components. Report intervals as conditional component-bootstrap intervals, not seed uncertainty.

## Required source-only tests after actor handoff

- Fail-closed malformed/missing proof records, altered artifact bytes, duplicate/missing arms, role leakage, non-finite raw arrays, wrong masks, and changed fit/source hashes.
- Synthetic official-metric examples with known arm values; all 27 effects and intervals checked against direct calculations, shared-draw interactions checked, arm-order invariance, and candidate-equals-reference zero contrasts.
- Independent reconstructed model checkpoint-load/output parity fixtures without opening test labels; CPU bounded import/help and synthetic scoring checks under 1 CPU/2 GiB.

No test labels, GPU, training or score execution will be used while implementing these checks. The independent scorer source will be written only after the actor source is handed off and the blind design critique is complete.
