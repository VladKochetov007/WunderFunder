# E477: readout activation mixture by head dropout

**Status:** source-only draft; no GPU probe or fit launched. The 16 E470 test
components have been exposed in earlier campaign work; this is exploratory
mechanism evidence only. The root-owned grader computes pooled WP and paired
component-bootstrap contrasts after all twelve fits finish.

## Question and fixed 12-fit matrix

Does a heterogeneous or same-channel mixed readout activation interact with
head-only dropout on the incumbent H64 architecture? The six conditions per
seed are the 3 activation choices crossed with head dropout `p in {0, 0.1}`;
the two fixed seeds are `271828` and `271829`.

| Readout activation | Definition |
|---|---|
| GELU | exact PyTorch `GELU(x)` |
| channel split | `concat(GELU(x[:32]), LeakyReLU_0.01(x[32:]))` |
| same-channel mix | `0.5*GELU(x) + 0.5*LeakyReLU_0.01(x)` |

Dropout is applied immediately after the selected activation at the 64-channel
head hidden layer. The 42->64 GELU input projection, ordinary two-layer H64
GRU and gates, `[recurrent, embedding]` concat order, linear head layers,
objective, and all non-head operations are unchanged. There are no new
trainable activation parameters. This compares readout-function diversity
and stochastic regularization at fixed width; it is not a larger ensemble.

## Matching and initialization

Use the final reviewed E474 24-fit-only loader and trainer helper. For each
seed, clone the exact GELU/GELU E474 baseline state dictionary, including its
neutral mean/std buffers, into each of the six arms. Assert exact initial
parameter/buffer bytes, key names and 61,058 trainable parameters. The
`GELU,p=0` condition must have exact CPU output/state/gradient parity with
E474 before training. Each seed uses one NumPy-shuffled permutation of the
24 fit groups, replayed for every arm. During the optimizer updates reset
PyTorch's RNG to the seed for each arm; within the p=0.1 conditions this gives
the same dropout RNG stream over matched batch/chunk updates.

## Cohort and fixed training recipe

Load only the E470 cohort's 24 fit row groups; do not open any test group or
test label. Use the existing 42 causal input features after the E470 fit-only
normalization. Keep the E470 weighted-MSE objective and `need_prediction`
training mask. For every fit, use B=4 sequence groups, reset GRU state at each
new four-sequence batch, carry state through all 79 contiguous 256-row chunks,
and detach after each chunk update: exactly 474 updates. Evaluation is in
`eval()` mode, so head dropout is disabled; reset state per sequence and
preserve chunk carry.

Use AdamW with lr 0.001, weight decay 0.01, betas (0.9, 0.999), constant
learning rate, global gradient norm clip 1.0, one pass, FP32 inputs and
parameters, autocast disabled, and the same IEEE FP32 backend policy as E474.
Dropout `p=0` is the unchanged baseline intervention. Only the head
activation/dropout condition differs among arms.

Record E474's fixed fit-only telemetry at zero-based updates 0, 46, 236, and
473: element-weighted preclip gradient RMS and post-update parameter-delta RMS
for `inp`, `gru`, and `head`. It is descriptive only and cannot trigger any
adaptation.

## Fixed contrasts

Report every contrast separately for each seed, using the same component
bootstrap draws for all compared arms:

1. split,p=0 minus GELU,p=0
2. mix,p=0 minus GELU,p=0
3. split,p=0 minus mix,p=0
4. split,p=0.1 minus GELU,p=0.1
5. mix,p=0.1 minus GELU,p=0.1
6. split,p=0.1 minus mix,p=0.1
7. **Mixture x dropout interaction:**
   `(mix,p=0.1 - mix,p=0) - (GELU,p=0.1 - GELU,p=0)`

Each score and contrast uses the root-owned `global_wp_clip2_v1` implementation
on pooled selected rows, with paired bootstrap resampling at the component
level. Do not pool the two optimization seeds into a seed-generalization
interval or select a winner from these previously exposed components. Report
the fixed `p=0.1` split-by-dropout interaction descriptively as secondary only
if the root grader derives it without changing the predeclared primary set.

## Validation and compute budget

CPU tests must prove formula behavior, all-initial-state identity, parameter
count, exact E474 baseline parity, dropout-off evaluation, causal future
isolation, state carry/reset, and finite outputs/gradients. Run one actual
fit-only CPU B4x256 optimizer update per arm. Then obtain source/critic review
and an actual-shape CUDA probe before any fit admission. Twelve complete fits
cost 5,688 optimizer updates and process 5.76 million fit rows across repeated
chunks. This implementation agent launches no CUDA job or full fit.
