# E487: Muon orthogonalization precision diagnostic

## Question and separation from E485

E485's H128 CUDA probe passed output, state and gradient parity but failed the
registered native-versus-custom post-update check. For `gru.weight_hh_l0`,
the reference update RMS was `1.362852e-4`, the discrepancy RMS was
`2.3377e-6` (1.7153%), and the maximum absolute delta error was `6.85006e-5`.
That exceeded both the probe's own `6.82426e-5` per-tensor cap and the stricter
root admission cap of `2e-5`. The E485 full trajectories remain **not
admitted**; this proposal does not change that result or any tolerance.

E487 asks whether the CUDA BF16 Newton–Schulz orthogonalizer in the existing
Muon update contributes to the native/custom update discrepancy, compared
with an otherwise identical FP32 Newton–Schulz calculation. This is a
single-batch numerical diagnostic, not a training or model-quality study.
The only treatment is the intermediate dtype used inside orthogonalization.
Keep the native and custom recurrences, H128 two-layer topology, common
parameters, input batch, labels, mask, optimizer buffers, group membership,
global clipping, LR, weight decay, betas, Newton–Schulz coefficients/steps,
and all other arithmetic fixed.

## Fixed observations and crossed replays

Use the E485 fit-only loader and immutable cohort. Fix initialization seed
271836, take the first four canonical fit-role indices from
`default_rng(271836).shuffle(np.arange(24))`, and use the first 256 rows of
that B4 batch. Load only fit-role features, labels and mask. Use identical
H128 native-GRU and custom alpha-zero parameter bytes, reset state, eval-free
train-mode forward, and the E470 weighted-MSE loss. Capture each model's
finite unclipped gradient tensors once from this one batch, clone them, and
then do no further forward/backward passes during the update comparisons.

Replay each captured gradient source (native-generated and custom-generated)
through both optimizer variants:

| Fixed gradient source | Existing CUDA BF16 orthogonalization | Diagnostic CUDA FP32 orthogonalization |
|---|---:|---:|
| Native-GRU gradient snapshot | 1 update replay | 1 update replay |
| Custom-scan gradient snapshot | 1 update replay | 1 update replay |

Every replay starts from the same byte-identical parameter snapshot and fresh
zero optimizer state. Clip the exact stored gradient tensors with the same
global norm rule independently for each replay; require the resulting clipped
gradient bytes to match across precision arms for a given gradient source.
Keep all parameter, gradient and optimizer-state tensors FP32. The BF16/FP32
choice is local to the five-step Newton–Schulz iteration: same coefficients,
normalization, epsilon and final cast to FP32. Do not edit or monkeypatch
`src/wnn/optim.py`; implement the diagnostic FP32 path in a new, separately
reviewed helper. No AdamW operation, clipping rule or parameter-group
assignment may differ between the two variants.

Add a fixed-shared-gradient control. Assign one deterministic synthetic
gradient snapshot, with global norm above the clipping threshold, to identical
native/custom parameter clones. For each precision path, both clones must
produce byte-identical parameter and optimizer-state results. The BF16 result
must additionally match the frozen E474 update helper byte-for-byte. The FP32
diagnostic helper must match the existing CPU FP32 orthogonalization path
byte-for-byte on the same CPU synthetic matrices. These controls separate optimizer implementation
differences from gradients produced by the native and custom recurrences.

## Gates and outcomes

Keep the existing E485 forward/state/input-gradient/parameter-gradient
tolerances unchanged. For each precision variant, compare the native-gradient
and custom-gradient replay deltas from identical pre-step snapshots. Require,
for each of the four GRU matrices and each emitted AdamW parameter family:

- discrepancy RMS divided by reference update RMS no greater than 0.02;
- maximum absolute delta discrepancy no greater than `2e-5`.

These are the existing strict root admission criteria, not new post-result
tolerances. Report raw and clipped gradient norms, pre-step hashes, per-matrix
and per-family native/custom delta RMS, relative RMS error, maximum error,
BF16-to-FP32 delta changes for each fixed gradient, and active precision flags
captured inside the full-IEEE context. Reject missing/nonfinite tensors,
changed input/initialization bytes, any shared-gradient control mismatch, and
any measurement that crosses a gate. Preserve the failing result; do not retry
with changed tolerances or a different seed.

If FP32 passes while BF16 fails on the same fixed gradients and state, that is
evidence that the orthogonalizer's intermediate precision contributes to this
one-step parity failure. It would not show improved convergence or predictive
performance, and it would not retroactively admit E485. Any full experiment
using FP32 orthogonalization requires a new preregistered study with its own
source review and probe. If both fail, the gradient-path difference remains
unresolved; if both pass, investigate reproducibility of the earlier failure
without discarding it.

## Source and resource gates before execution

First obtain an independent blind review of this design. Then create a new
diagnostic-only source packet and synthetic CPU tests for the exact optimizer
formula, same-gradient exactness, FP32-vs-existing-CPU path equality, and
strict delta accounting. Bind it to the frozen E470/E471/E474/E485 source
hashes, immutable dataset/cohort hashes and actual-shape fit-only CPU
preflight. Independently review the new source before any CUDA use.

One GPU diagnostic worker only; cap at 2 CPUs, 6 GiB host RAM, 2 GiB allocator
memory and stop at 3,000,000,000 bytes observed device use. Use a short bounded
timeout (maximum 600 seconds), systemd resource accounting and no swap. Parent
limits remain 12 CPUs and 30 decimal GB RAM. No full fit, test-role load,
score, public forecast, optimizer benchmark sweep, provider mutation or ledger
edit is part of E487.
