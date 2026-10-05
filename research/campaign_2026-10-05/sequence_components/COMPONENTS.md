# Isolated sequence components

This directory contains small PyTorch modules for the sequence-model benchmark.
It does not read challenge data, launch training, or change the root model runner.

`DiagonalSSM(input_size, state_size, learn_decay=True)` embeds each input with a
linear map and `tanh`, then updates a diagonal state with a convex stable
recurrence. The state coordinates use a log-spaced spectrum from 0.85 to 0.999
at initialization. With `learn_decay=False`, that spectrum is fixed; with the
default, each decay is learned while staying in `[0.5, 0.9999]`. It returns
`(states, final_state)`, where `states` has shape `(batch, time, state_size)`
and `final_state` has shape `(batch, state_size)`. Pass the final state to the
next chunk; pass `None` to reset.

`KernelAttention(input_size, key_size=8, value_size=8)` uses positive softmax
query/key features and a recurrent leaky key/value memory. The value projection
uses `tanh`, and each output is a normalized positive weighted average, bounded
by the value range. It returns `(outputs, (key_value_memory, key_mass))`. The
state shapes are `(batch, key_size, value_size)` and `(batch, key_size)`.
Pass both tensors to continue a stream or `None` to reset. This is recurrent
positive-kernel attention with fast-weight memory; it does not construct a
Transformer block or a quadratic attention matrix.

`AdditiveSpline(input_size, output_size, knots=7)` applies independent learned
piecewise-linear univariate functions to each input coordinate and adds their
contributions. It is KAN-inspired, but it uses linear interpolation rather
than the original cubic B-spline parameterization. Its edge values start at
zero, so `baseline + spline(features)` exactly equals the baseline initially.
The default interpolation domain is `[-3, 3]`; values outside it use the
nearest endpoint value.

`PointwiseMLPControl(input_size, output_size, parameter_budget)` supplies a
per-position two-layer MLP control. `mlp_hidden_for_budget` picks a hidden
width from a requested trainable-parameter budget; report the resulting actual
parameter count when matching an experiment, since widths are integer-valued.
This control has no state and cannot itself model temporal dependence.

All recurrent modules accept `(..., time, feature)` tensors, permit chunked
streaming through their returned state, and use `state=None` as an explicit
reset. Their checks compare full-sequence and chunked execution, perturb only
future inputs to verify prefix causality, verify reset behavior, check the
zero residual anchor and spline gradients, and stream/backpropagate across
20,000 rows in bounded chunks. Run the CPU checks with:

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited \
  .venv/bin/python research/campaign_2026-10-05/sequence_components/tests.py
```
