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

`scripts/profile_activation_qat.py` is a bounded compiled forward/backward
profile, defaulting to batch 16, chunk 1,000, hidden width 256 and two layers.
It writes a receipt on success or exception. Compiler/device failures use
`backend_status: backend_failure`; every receipt keeps
`quality_status: not_evaluated`. It records compile/first-update and warmed
forward/backward peak CUDA memory. The profile is not a fit or score.

## Recipe audit and limits

The reference equations match PyTorch's GRU gate order `[reset, update,
new]`: reset and update combine input/recurrent affine terms, while reset
multiplies the candidate recurrent affine term including its recurrent bias.
The custom scan parity test compares outputs, final states and gradients for
two layers, a nonzero initial state and split chunks against fused `nn.GRU`.
The focused four-test file passed in 4.76 seconds under the root-granted
1-CPU/1.5-GB/180-second window. Its controlled native fixture sets the input
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

This implementation is recurrent-only. It does not fake-quantize the
projection-to-first-layer activation or the head hidden/projection activation
blocks. Native `q8vnni*` still uses integer accumulation and its own floating
epilogue, so the state comparison is close numeric agreement rather than
bitwise gate-level equivalence. Fast nonlinearities are disabled in that
diagnostic. No Rust build, GPU profile, long scan, training run or quality
evaluation was performed. The B16/C1000 compiled profile and trainer
integration remain pending root-owned admission.

## Source records

- [Versioned quantization specification](QUANTIZATION_SPEC_R1.md) — fixed
  weight and native A8 semantics.
- [Historical E419 QAT report](../../campaign_2026-10-04/E419_RECOVERED_QAT_REPORT.md)
  — prior weight-only Q8 evidence and rationale for activation-aware work.
- [Plan](PLAN.md), [frontier](FRONTIER.md) and [campaign contract](CONTRACT.json)
  — active experiment scope and execution ownership.
- Tests: `tests/test_activation_qat.py`.
- Focused CPU/native command: `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1
  WNN_Q8_TEST_LIBRARY=/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/native_qat_r1
  WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=180s scripts/limited
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/pytest -q
  tests/test_activation_qat.py`. The root granted 1 CPU, 1.5 GB and 180 seconds;
  all four tests passed. The supplied library SHA256 was
  `a4390ecf7bcd54bb6826b43f5dcf6d7a40064c64ed1450bdc1780c34c2a6e108`.
- Deferred command: `python scripts/profile_activation_qat.py --device cuda
  --batch-size 16 --chunk 1000 --features 112 --hidden 256 --layers 2
  --receipt research/campaign_2026-10-13/qat_renewable_r1/ACTIVATION_AWARE_PROFILE_R1.json`.
  Run only after root grants the measured test/profile window and GPU admission.
