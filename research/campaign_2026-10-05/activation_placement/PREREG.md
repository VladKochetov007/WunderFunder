# E474: localized activation placement

**Status:** source draft; no fit launched. Do not inspect exposed test outcomes
until all twelve predeclared fits finish. The test components have appeared in
earlier campaign screens, so this is an exploratory mechanism study rather
than clean generalization evidence.

## Question and fixed matrix

Does adding a small sine or matched linear term help specifically at the
incumbent model's input-embedding or hidden-head activation site? This screen
uses six models for each fixed seed, `271828` and `271829`:

| Arm | Embedding activation | Head hidden activation |
|---|---|---|
| base | GELU | GELU |
| sine_embed | GELU + 0.1 sin(x) | GELU |
| slope_embed | GELU + 0.1 x | GELU |
| sine_head | GELU | GELU + 0.1 sin(x) |
| slope_head | GELU | GELU + 0.1 x |
| sine_both | GELU + 0.1 sin(x) | GELU + 0.1 sin(x) |

All activations are parameter-free. PyTorch exact GELU is used at every site;
the candidate formulas are `F.gelu(x, approximate="none") + 0.1*sin(x)` and
`F.gelu(x, approximate="none") + 0.1*x`.

## Model and initialization

Use the actual `wnn.nn.RNNModel(42, hidden=64, layers=2, dropout=0.0,
n_out=2, cell="gru", enc="linear")`: normalized input -> linear embedding
and activation -> ordinary two-layer `nn.GRU` -> concatenate `[GRU output,
activated embedding]` -> linear head -> selected hidden activation -> output
linear. Embedding changes also change the embedding skip entering the head.
The count is **61,058 trainable parameters** in the live baseline
(input projection 2,752; GRU 49,920; concat head 8,386). Parameter-free
activation choices must preserve the exact state keys, shapes, initialization,
and count.

For each seed, initialize one GELU/GELU baseline, then clone its full state
dictionary (including mean/std buffers) into all six arms. This avoids any
dependence on parameterized construction order. Require exact baseline output
and recurrent-state parity against `RNNModel` before training.

## Data, objective, and training recipe

Use the immutable E470 `target_mechanics/cohort.json` cohort and E470 feature
construction/fit-only normalization, but load the 24 fit row groups only. Each
model input has exactly 42 causal feature channels, 20,000 rows, and two
targets. Do not open the 16 exposed test groups in the fit or probe code. Keep
the E470 weighted-MSE objective and `need_prediction` training mask unchanged;
`is_scored` is not a training mask.

For each seed, shuffle the 24 fit groups once and share that order across all
six variants. Use B=4 sequence groups; each batch resets GRU state to zero and
then carries state across the 79 contiguous 256-row chunks, detaching after
each optimizer step. This yields exactly 6 batches x 79 chunks = 474 updates
per fit. Use AdamW, lr 0.001, weight decay 0.01, betas (0.9, 0.999), constant
learning rate, global gradient norm clip 1.0, one pass, and no dropout. All
arms have the same batch/chunk/order and initialization per seed.

Use FP32 inputs/parameters, autocast disabled, and explicit IEEE FP32 policy
for CUDA matmul and cuDNN RNN in both the candidate and baseline comparison.
This intentionally differs from historical E470 runs that may have used
cuDNN TF32. Do not mix backend policy by arm or characterize the scores as
bit-exact historical E470 replicas.

At zero-based optimizer updates `0`, `46`, `236`, and `473` in every fit,
record fixed fit-only telemetry from the actual training minibatch: the
element-weighted RMS of pre-global-clipping gradients and the RMS parameter
delta after the optimizer update. Aggregate separately for the unchanged
`inp`, `gru`, and `head` named-parameter families. Update 0 is the first
shuffled B4 batch at chunk offset 0; later indices refer to the same flattened
batch/chunk traversal. The sample indices and family aggregation are frozen
for all arms before fitting. Telemetry is descriptive only: it cannot trigger
any fit, optimizer, LR, clipping, or activation changes. No test row is
consulted for telemetry.

## Fixed contrasts and interpretation

Retain all seven contrasts separately by seed:

- sine_embed - base; slope_embed - base; sine_embed - slope_embed
- sine_head - base; slope_head - base; sine_head - slope_head
- sine_both - sine_embed - sine_head + base

The last is the additive site interaction. The sine-versus-slope comparisons
distinguish added curvature from a matched derivative/output-scale perturbation
at zero, but they do not establish a universal activation mechanism. A sine
advantage over baseline without a sine advantage over the matched slope does
not support a periodic-curvature explanation. No model may be selected by
these exposed results; root-owned scoring retains the full matrix and uses
paired component resampling where reported.

## Validation and budget

Before admission, run the CPU invariants and a fit-only CPU one-update preflight
for all 12 arms. Then root/critic reviews source hashes and runs the actual
B4x256 CUDA shape probe before any fit. Source provides an optional twelve-fit
trainer only; it is not authorization to launch. Full matrix cost is 5,688
optimizer updates across 12 fits and 5.76 million fit rows processed with
repeated chunks. No GPU job is started by this implementation agent.
