# Callback readout-skip value triage (R2)

**Finding: do not implement or package this optimization for the current scorer.**
The contest callback's `need_prediction` flag is false only for warm-up rows
0–98 and true for every row 99–19,999. It is not the scoring mask. Therefore,
the proposed readout skip can touch only 99 of 20,000 callbacks (0.495%). The
official input contract separately says every row must update model state and
that `is_scored` rows are a subset of required predictions; the callback does
not provide `is_scored` to this predictor.

The original four-member head inventory estimates 0.658M readout coefficients
among 4.020M total matrix coefficients per row. Even assuming runtime scales
linearly with coefficient work, the maximum sequence-level saving is

`(99 / 20,000) × (0.658 / 4.020) = 0.0008102`, or about **0.081%** of matrix
work. Applied to the 3,016-second callback reference, that is at most about
2.45 seconds before code-path, cache and non-matrix overhead; it is not a
measured speedup. A branch or extra API surface could erase that margin.

The source confirms the original proposal's mechanism but not enough benefit:

- The official [sequence contract](../../../docs/data_overview.md#sequences)
  states that `need_prediction` is false only on steps 0–98, true thereafter,
  and `is_scored` is a separate field unavailable in the callback (`data_overview.md`
  SHA-256 `c6d1e2c5ae25717362a993fab52e0af05c1f7f85e2d58dfe572dbfb2d9fdd1cc`).
- In [`native/src/python.rs`](../../../native/src/python.rs),
  `PyEnsemblePredictor::predict` calls `advance()` before reading
  `need_prediction` (SHA-256
  `63cec725cf1e9c40ee5228151daad94a0e577813b3833a8d097fc49338635919`).
- In [`native/src/rnn_fast.rs`](../../../native/src/rnn_fast.rs), dynamic Q8
  advances each recurrent layer and quantizes its hidden state before the
  readout `w1`/`w1z`, activation and `w2` operations (SHA-256
  `e65dd681690bb290f36b4287cac5ea3e130080fd47b821f1e1f9d98676aab145`).

The source-level opportunity remains real, but this caller has too few
unrequested rows to justify implementation effort. This is a campaign-specific
priority decision, not a finding that readout skipping is generally ineffective.
No native source was changed; no build, benchmark, fit, package or upload was
performed. The original [R1 proposal](README.md#proposal) remains intact as the
historical hypothesis.
