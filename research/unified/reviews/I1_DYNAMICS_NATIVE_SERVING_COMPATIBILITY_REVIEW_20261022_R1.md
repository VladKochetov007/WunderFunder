---
kind: review
status: source_confirmed_serving_gap
created_utc: 2026-10-21T22:06:31Z
updated_utc: 2026-10-21T22:06:31Z
evidence_paths:
  - research/experiment_drafts/dynamics-aux-20261021/README.md
  - research/unified/reviews/STRONG_PARENT_F0_F1_WARM_INIT_CATALOG_20261022_R1.md
---

# i1 and dynamics native-serving compatibility

This source-only review traces the canonical checkpoint exporter and callback
to check whether stripping the optional dynamics head makes the current
367-input screen models directly packageable. It does not load model tensors,
prediction or data arrays, run tests, or measure runtime. The review found a
serving compatibility gap; it does not change the frozen cold-screen recipes
or imply a model-quality result.

[`package_submission.py`](../../../scripts/package_submission.py) converts the
checkpoint state to FP32 and calls
[`strip_dynamics_auxiliary_state`](../../../src/wnn/training/dynamics_auxiliary.py)
before validating its input layout. That helper removes the 52-output head and
two target-statistics buffers, but retains `mean`, `std`, `inp.weight`, and the
checkpoint's declared `n_in`. The subsequent
[`native_input_metadata`](../../../src/wnn/native_representation.py) accepts
211 or 323 inputs in `none` mode and 467 in the named `base`/`ema` modes. It
does not describe a 367-column `i1_history` layout. Consequently both active
history C and the all-zero-history B control fail the canonical export gate;
removing the auxiliary head does not repair their input width.

The default package callback,
[`solution_multi.PredictionModel`](../../../submission/solution_multi.py), is
Python code but delegates feature construction and recurrent inference to
`wnn_native.Predictor`. It passes each raw `data_point.state` row to that
predictor and resets member state at a sequence change. Rust's
[`Predictor`](../../../native/src/python.rs) accepts only raw width 112 at its
row boundary, then constructs the 99 event columns and an optional supported
suffix. Its current layouts cover 211/323 `none` inputs or 467 `base`/`ema`
inputs; it has no 156-column i1-history builder. The separate
[`StreamingI1History`](../../../src/wnn/i1_history_streaming.py) helper is not
wired into this package callback. The lower-level Rust `ModelStepper` accepts
precomputed model inputs of its weight-defined width, but does not turn the
official 112-wide raw callback row into the complete 367-wide input. It is not
an existing end-to-end serving path for these checkpoints.

There is a source-supported projection path for the B-control representation.
[`RNNModel._normalized_inputs`](../../../src/wnn/nn.py) applies the 367-wide
normalization and then sets the masked history coordinates to exact zero before
the input projection. For a checkpoint whose provenance confirms
`research_features=i1_history`, `research_feature_control=all`, and
`drop_i1=false`, the model's mathematical function can therefore be represented
with the first 211 means/scales and first 211 `inp.weight` columns, preserving
`inp.bias` and every other model tensor. The current exporter does not perform
this projection, rewrite the exported width metadata, or record that
transformation. This is a feasibility observation, not verified FP32/native
parity. In practice, different matrix widths can change floating-point
reduction order, so exact output identity must be measured rather than inferred
from the algebra alone.

Before using a projected B-control model in a complete endpoint, the narrow
proof is to implement an explicitly gated export conversion for only that
semantic mask; preserve original checkpoint and source hashes; assert the
projected tensor shapes and unchanged bias/backbone; and compare the original
masked 367-input model against the packaged callback over complete sequences
with sequence-reset replay. The comparison must use the actual package
`PredictionModel.predict` path, exact supported raw inputs, full row coverage,
and an explicit numerical tolerance. Then measure the full package's runtime
and size under the official serving harness. The existing cold-screen
prediction grader does not establish any of those packaging or cost results.

Active C cannot use that projection because its 156 history inputs affect the
model. Supporting it requires a package input layout and causal feature
streamer with the fitting-only history prior, reset/carry semantics, and parity
against the frozen Python feature/model path, followed by a complete callback
cost measurement. No such native or package integration is present in the
reviewed source.

The dynamics head is absent from ordinary evaluation outputs and can be
removed from an exported state, so the auxiliary objective adds no required
serving head. The stronger claim that these screen checkpoints already have a
cheap, supported native endpoint is not established: the common 367-input
representation is still rejected. For the all-zero-history control, a 211
projection is a plausible low-cost route, but it remains unimplemented and
unmeasured. This caveat does not change the accepted cold-fit plan, whose
prediction and official-score stages run through the existing PyTorch path.

The related [warm-parent catalog](STRONG_PARENT_F0_F1_WARM_INIT_CATALOG_20261022_R1.md)
covers the distinct inheritance and normalization gates. This review makes no
recommendation to select C, project a checkpoint, or admit follow-up fits.
