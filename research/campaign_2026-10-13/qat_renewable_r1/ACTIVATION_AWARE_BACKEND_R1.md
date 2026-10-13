---
kind: method
status: admitted
created_utc: 2026-10-13T17:30:00Z
updated_utc: 2026-10-13T17:40:25Z
hypothesis_ids: [H-QAT-A8-20261013]
experiment_ids: [E-QAT-DEPLOYMENT-20261013-R1]
evidence_paths:
  - QUANTIZATION_SPEC_R1.md
  - ../../campaign_2026-10-04/E419_RECOVERED_QAT_REPORT.md
---

# Activation-aware recurrent QAT backend

## Question or claim

Can training simulate the native dynamic-A8 rounding at both GRU matvec
boundaries while preserving the float32 recurrent state, without building a
Python graph with one node per timestep?

## Method

`src/wnn/activation_qat.py` implements per-vector A8 codes using the native
formula `round_ties_even(clamp(x * (127 / maxabs), -127, 127))`, returns the
matching `maxabs / 127` scale, and applies an identity straight-through
gradient to the dequantized vector. Its scan-based GRU precomputes each
layer's input-to-hidden terms over the sequence and scans only the recurrent
matvec. This keeps causal histories across calls/chunks and carries the final
FP32 state. Existing fixed or learned weight-QAT presets use the same
represented-weight cache and gradient helpers as `training_qat.py`.

The opt-in `train_rnn.py --activation-backend scan_fp32|scan_a8` route keeps
the existing model loading, actual data/cache readers, training objective,
sampler, AdamW updates and QAT cache invalidation. It is disabled by default.
The scan implementation is imported only when this opt-in path is selected, so
the established fused-GRU CLI does not import the private scan operator.
The profiler derives the 10% recurrent-W4 fit argv from the frozen recipe and
compares the compiled FP32-activation scan with A8 on the same first actual
B16 batch, reusing it for one compile/warmup and one measured optimizer update
per mode. Only the first 1000 rows from sequence origin are forwarded; this is
a backend smoke, not a fit-fidelity or quality experiment. The receipt records
checkpoint, normalization, cache and source hashes; loss/gradient/parameter
finiteness; QAT cache invalidation and represented-weight fingerprints; compile
times; and CUDA/RSS peaks. Propensity-cache sidecar details are collected only
for this bounded activation profile, preserving compatibility with existing
fit runs and caches that predate those sidecars. For same-batch host reuse the
profile removes `--fast` and `--pin-memory` only; explicit `--fused-adam`,
`--sync-free` and `--prop-cache` flags remain. The wrapper requires matching
checkpoint, split/subset membership, normalization, distillation/propensity
cache, source and sample-order identities across both modes, as well as finite
parameters, valid pre-update cache identity, post-update invalidation and
changed represented-weight fingerprints. The `synthetic_smoke` mode remains
explicit and has no fit admission value. Backend errors are separate from quality
(`quality_status: not_evaluated`).

## Recipe audit and limits

The reference equations match PyTorch's GRU gate order `[reset, update,
new]`: reset and update combine input/recurrent affine terms, while reset
multiplies the candidate recurrent affine term including its recurrent bias.
The custom scan parity test compares outputs, final states and gradients for
two layers, a nonzero initial state and split chunks against fused `nn.GRU`.
The pre-correction four-test suite passed in 4.76 seconds under the root-granted
1-CPU/1.5-GB/180-second window. The current source adds explicit nonfinite A8
input rejection tests; they have not run under the present code-only scope. Its controlled native fixture sets the input
projection weights and bias to zero, so the I16 path produces an exact zero
projection; nonzero recurrent candidate bias drives hidden-state variation.
Using the supplied `q8vnni_portable` library and exact imported Q8 codes,
maximum final hidden-state error against the A8 scan was `5.96046448e-08`.
Disabling A8 changed the final hidden state by `6.89923763e-04`; native error
against that no-A8 control was `6.89938664e-04`. This demonstrates sensitivity
on this fixture and close native agreement, not bitwise gate-level equivalence
or a quality result. The test also checks A8 code/scale rounding against an
independent scalar formula. One initial post-edit run exposed a `torch.fmax`
bound type error; that issue was corrected before the full focused file passed.

This implementation applies A8 before every GRU `weight_ih` input matvec,
including the first layer's post-encoder activation, and before each
`weight_hh` recurrent matvec. It does not quantize raw model inputs, input
projection weights/native I16 projection math, or the head's hidden/projection
activation blocks. Native `q8vnni*` still uses integer accumulation and its own
floating epilogue, so the state comparison is close numeric agreement rather
than bitwise gate-level equivalence. Fast nonlinearities are disabled in that
diagnostic. The previous B16/C1000 random-input/square-loss invocation is
retained only as an explicitly synthetic compile smoke; it is not a profile
admission input or training objective. A source-qualified actual-data
profile using the CP5 checkpoint, real fit labels/cache and AdamW updates is
prepared in [the actual-data profile plan](ACTIVATION_AWARE_ACTUAL_PROFILE_PLAN_R1.json)
and remains pending root-owned GPU admission. No Rust build, GPU profile, long
scan, fit or quality evaluation has been performed by this code-only actor.
The public eager forward validates finite inputs, initial state and model
parameters before recurrence. The pure scan body does no host-side finiteness
synchronization; the trainer validates each actual input chunk and carried
state before entering the compiled scan, and checks finite losses, gradients
and parameters after the bounded updates. The actual recipe command and its
two-update-per-mode cap are in the [profile plan](ACTIVATION_AWARE_ACTUAL_PROFILE_PLAN_R1.json).

## Source records

- Initial backend revision: `7939e64c0a14cba9e0c1eaaf5822787c89dd54bf`; the opt-in trainer hook and profile wrapper are the current code-only successor and have not been compiled or run.
- [Versioned quantization specification](QUANTIZATION_SPEC_R1.md) — fixed
  weight and native A8 semantics.
- [Historical E419 QAT report](../../campaign_2026-10-04/E419_RECOVERED_QAT_REPORT.md)
  — prior weight-only Q8 evidence and rationale for activation-aware work.
- [Plan](PLAN.md), [frontier](FRONTIER.md) and [campaign contract](CONTRACT.json)
  — active experiment scope and execution ownership.
- Tests: `tests/test_activation_qat.py`.
- [Focused CPU/native receipt R2](ACTIVATION_AWARE_CPU_NATIVE_RECEIPT_R2.json) records four passing tests against successor `5f6812b0` and the supplied pinned native library; compiled trainer profiling remains untested.
- Focused CPU/native command previously passed on the pre-wrapper backend: `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1
  WNN_Q8_TEST_LIBRARY=/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/native_qat_r1
  WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=180s scripts/limited
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/pytest -q
  tests/test_activation_qat.py`. The root granted 1 CPU, 1.5 GB and 180 seconds;
  the documented four test functions passed. The current wrapper-state edge test and finite-input parameter cases have not run. The supplied library SHA256 was
  `a4390ecf7bcd54bb6826b43f5dcf6d7a40064c64ed1450bdc1780c34c2a6e108`.
- The exact actual-data profile command, pinned files, maximum four AdamW updates,
  payload sizing and root-owned execution boundary are in the [actual-data
  profile plan](ACTIVATION_AWARE_ACTUAL_PROFILE_PLAN_R1.json). Run only after
  root grants the measured profile window and single-GPU admission. The
  synthetic square-loss smoke requires the explicit `--profile-kind
  synthetic_smoke` and is not an admission substitute.
