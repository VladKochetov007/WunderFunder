Finite local initial-state influence persists in all8 selected windows: every
plus/minus direction remains above its calibrated primary-output floor at every
predeclared lag, including the289-update endpoint. The instrument passes its
identity, causality, finite-output and toy controls. This is not predictive-value
evidence, a stability theorem, or a burn-in/training approval.

Selection and exposure: prepare.py froze8 distinct components by metadata hashes
from the SAME stored matched_usage24-component panel. Within each selected
component it chose one previously frozen query by hash. Selection.json records
all IDs, exact intervals and per-case seeds. No checkpoint, array value, target
or new mask was read before this selection. The parent panel's query pool was
originally score-mask selected and its validation targets were historically
exposed; this is not an unexposed input sample or a training-data sample.
This run never accessed target/mask array members and never computed a score.

Actual callable: src/wnn/nn.py RNNModel, checkpoint
c20261001_d256_s3_f0@ep3, SHA256
f92504fbbbcf956a44e70aa3f7a08825d903b9c6a6cacce7bd9f89b6875fcad8.
The actual export is loaded on CPU with its exact211-input, two-layer256-unit
linear-encoder GRU and23-output state dictionary. eval_ema=true describes the
exported evaluation weights; no raw/lagged training-state replay is asserted.
Model/normalization tensors are hashed before and after and remain identical.
Source hashes match the frozen parent control. The true incoming hidden state
is read from original_states at the saved query-288 boundary. These states were
computed by the frozen parent CPU model after processing every preceding input,
including warmup. Their lineage is the stored artifact, checked by SHA256; I did
not independently regenerate the full prefix.

Only the original packed input rows[query-288,query] are replayed. Every arm
receives exactly the same289x211 tensor: no donor, raw splice, feature change,
parameter change, optimizer or training step. The direction seed is614407 plus
the original panel index. Each layer has an independently RMS-normalized random
direction scaled to1% of its incoming hidden RMS. Both signs are tested. Zero
hidden/direction norms yield exact zero deltas and are retained; the zero-norm
toy explicitly checks this path. No actual sampled direction has zero norm.
Float32 effective deltas are recorded separately from requested deltas.

Lag0 means after the first input row, not the incoming state; lag288/end means
after289 recurrent updates. Independent segments stop at rows0,8,32,96,256,288
to obtain BOTH layers' true hidden states. Calibration BEFORE perturbation uses
unchanged segmented repeats, full-window output, direct-prefix hidden states,
and stored original CPU reference outputs. Output floors are ten times observed
unchanged discrepancies, with a scale-dependent FP32-epsilon minimum. Hidden
floors use repeat/direct-prefix differences per layer. These are numerical
resolution conventions, not prediction-error tolerances or analytic error bounds.

| Local lag | Updates | Primary output max-abs response range | Above-floor signs | Hidden-difference L2 range |
|---|---:|---:|---:|---:|
|0|1|.000541–.002448|16/16|.10770–.12641|
|8|9|.000298–.002406|16/16|.07778–.10884|
|32|33|.000130–.001707|16/16|.04687–.09598|
|96|97|.000274–.001304|16/16|.03653–.08971|
|256|257|.000201–.001260|16/16|.02751–.06211|
|288/end|289|.000138–.000920|16/16|.02668–.05937|

These min/max envelopes mix8 inputs and2 signs; they are not fitted decay rates
or confidence intervals. Each case/sign/lag, including every zero/floor flag,
is retained in results.json and the case NPZ files. No exponential or contraction
theorem was fitted. At the endpoint the hidden-difference L2 is17.2–40.9% of the
actual initial-difference L2. Primary output response need not decrease at every
sampled row as hidden directions move through the nonlinear readout.

Numerical scale and activation envelope:

- Incoming per-layer hidden RMS .515600–.767501; requested per-layer delta RMS
  therefore .005156–.007675. Perturbed initial states push14–37 coordinates per
  signed case slightly outside the GRU reachable coordinate interval[-1,1].
  They are intentionally not clipped, preserving the requested direction and
  scale. The local state perturbation is not certified reachable by another
  real prefix; extrapolation to admissible reset/captured-state errors is limited.
- Across windows, normalized input maximum magnitude5.99–29.05 and encoded GELU
  maximum5.86–28.63. Sampled hidden entries above magnitude.99 account for8.79–
  18.72%. These measured scales describe the tested activations; saturation alone
  is not a proof of why a direction persists. Full per-window min/max/RMS,
  normalizer std and hidden/output scales are preserved.
- Calibrated primary-output floor range1.20e-7–8.34e-6; per-layer hidden floor
  range1.19e-7–1.85e-5. Unchanged CPU reference discrepancy maximum8.34465e-7
  passes the inherited2e-5 absolute+relative check. Repeat output/hidden error0.
- Same-length full-window future canaries alter inputs ONLY after local row32;
  all23 prefix outputs through row32 have discrepancy0. All actual outputs,
  hidden states, features, activations and saved arrays are finite. Future
  suffixes were modified only inside this instrument, not read from any future
  washout actor result.

Executable toy controls include hidden dynamics beta=.5 with observed geometric
forgetting, beta=1 with persistent visible response, and a persistent hidden
direction invisible to the readout. The last control shows why an output null
does not establish convergence of all hidden coordinates. These simple linear
fixtures run in float64 separately from the actual FP32 GRU, with all lags
preserved, including tiny values below practical FP32 resolution. Zero-norm
direction handling executes the actual torch perturbation function.

Relation to the paper: [Schiller et al., Assumption1, Eq12](https://arxiv.org/html/2602.10911v1#S4.SS2)
bounds output differences by C*lambda^t times initial-state separation, for
arbitrary initial states and input subsequences extracted from the training data,
with common constants. It concerns incremental OUTPUT stability; hidden-state
contraction is not required. We test one trained evaluation parameterization,
8 validation input windows and local signed directions for289 rows. Their
persistent finite responses neither establish nor contradict the existence of
such constants. We have not checked the theorem's training-input domain, all
initial states, all parameters or optimization hypotheses. No paper gain result
is transferred to this task.

Cheap falsifier for the chunk-overlap idea: a claim that32/96/256 rows make every
tested1%-RMS state error numerically invisible is falsified here, in16/16 signed
cases at each horizon. In a future isolated overlap implementation, compare the
reconstructed boundary state/output against this stored true-state baseline on
these exact frozen inputs; a residual above the predeclared tolerance rejects
its claimed equivalence. An acceptable application tolerance must be declared
separately, not inferred from numerical epsilon. This test does not bound a
large zero reset, state error from changed training weights, longer horizons,
other directions, training gain or a global burn-in length. An overlap method
could still be useful despite imperfect equivalence; this run cannot approve it.

Reproduce from repo root:
`bash research/campaign_2026-10-03/continuation/round4/state_forgetting/run.sh`.
Run prepare.py through the same CPU wrapper only in a new output location;
it refuses to overwrite selection.json. summarize.py reduces saved results.
All computes use run_cpu.sh round2-critic, CPU1core/2GiB/900s. Measurement body
3.41s, CPU4.69s including imports, peak RSS716860KiB (~700MiB). Cost is measured
inside the process, so wrapper admission/startup time is excluded. STATE/COST
record commands and costs; frozen source/checkpoint/input hashes and all eight
target-free replay captures are delivered. No full-sequence forward, GPU, fit,
live source/queue mutation, scoring or future washout reads occurred.
