# E471 recurrent candidate activation pilot

Status: implementation/preflight only. No full fit or test scoring has been
run. This is a bounded mechanism screen, not a public-score claim.

## Question and fixed arms

Does a small, bounded change to the GRU candidate's pointwise map change
one-epoch optimization on the frozen E470 recipe? Hold the reset and update
gates at sigmoid and compare `alpha ∈ {-0.1, 0, +0.1}` in

```text
u = tanh(x_n + r * h_n)
n = u + alpha * sin(pi * u)
h' = z * h + (1-z) * n
```

Cross the three fixed values with `AdamW` and `MuonAdamW` for six arms. There
is no learned alpha, dropout, schedule change, architecture sweep, or
post-hoc arm selection. Alpha zero is the custom-scan/backend control.

The candidate derivative is `1 + alpha*pi*cos(pi*u)`, hence is at least
`1 - 0.1*pi > 0` over the tested range. The map fixes `u=-1,+1`, is monotone,
and stays within `[-1,1]`. This supports a bounded curvature/gradient-flow
perturbation; it does not imply periodicity in the data or prove long-horizon
gradient stability.

## Matched recipe and data boundary

Use E470's frozen causal 42-feature construction, fit-only normalization,
absolute-label-weighted MSE, AdamW/MuonAdamW settings and Muon gate-matrix
partition. The cohort is 24 fit plus 16 exposed diagnostic components,
20,000 rows each. Fit only the 24 designated fit sequences, in their
seed-271828 shuffled order, with B=4, 256-row chunks, carried/detached state,
one epoch (474 optimizer updates), width 32, and a zero-initialized residual
head. Score the 16 exposed test components with E470's fixed
`global_wp_clip2_v1` metric only after a separately reviewed full run.

All arms use seed 271828 and identical initialization layout. Their trainable
parameter count is 8,235. The custom scan retains `gru.weight_ih_l0` and
`gru.weight_hh_l0`, enabling the exact existing Muon selector. It implements
PyTorch's reset-after-hidden-affine candidate term, including both biases:
`r * (W_hn h + b_hn)`, while the input candidate affine is `W_in x + b_in`.
Only after `tanh(input_candidate + r*hidden_candidate)` is the candidate
activation modified.

The E470 recipe implementation is
`activation_optimizer/fit_activation_optimizer.py`; cohort and data hashes
are written by the trainer into any run receipt. The E471 trainer and tests
are separately hashed there. No existing E470 source, scorer, or grader is
modified.

## Checks and run gate

Before considering a full run, require review of the frozen code/spec and
these checks: alpha-zero forward/state/backward/AdamW parity to `nn.GRU`;
alpha-zero complete-model initialization/output/gradient/update parity to
E470 tanh; chunk/full-sequence parity, state reset, future-perturbation
causality, candidate bounds, finite gradients/parameters, and one fit-only
real B4×256 CPU update for every arm. The production-shape probe compares
alpha-zero complete-model output, state, and parameter/input gradients to
E470 on that same real fit-only B4×256 batch before making one update in
every arm. It checks values for finiteness before losses or optimizer updates.
Recorded parity tolerances are output/state `atol=2e-5, rtol=2e-4` and
gradient `atol=5e-5, rtol=5e-4`; a CUDA mismatch outside these bounds fails
the probe. It loads no test row groups and scores nothing.
The required new-only `--output` directory receives `result.json` with source,
cohort, and data hashes; per-arm update durations; and a rough 474-update
projection that excludes cohort loading and parity work. The CPU dry-check
receipt is in `recurrent_activation/probe_drycheck_v3/result.json`; its timing
is a correctness dry check, not a CUDA throughput estimate.

Suggested bounded commands (set any campaign resource environment variables
required by the local runner):

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited \
  .venv/bin/python research/campaign_2026-10-05/recurrent_activation/tests.py

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited \
  .venv/bin/python research/campaign_2026-10-05/recurrent_activation/fit_recurrent_candidate.py \
  --preflight-real --device cpu \
  --output research/campaign_2026-10-05/recurrent_activation/cpu_preflight
```

After spec/source review only, a parent-controlled CPU dry check and CUDA
shape probe can be run from `gpu_probe.py` under the campaign resource guard.
The CPU dry check is:

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited .venv/bin/python \
  research/campaign_2026-10-05/recurrent_activation/gpu_probe.py --device cpu \
  --output research/campaign_2026-10-05/recurrent_activation/probe_drycheck_v3
```

The CUDA probe is:

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2G WNN_RUNTIME_MAX=180 WNN_GPU_MEMORY_BYTES=5000000000 \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-05/recurrent_activation/gpu_probe.py \
  --output research/campaign_2026-10-05/recurrent_activation/probe_cuda
```

The CUDA form has not been launched by this actor. A full six-arm
run requires a separate parent decision, immutable content seal, single
tracked campaign launch, and independent results review.

## Interpretation limit

This pilot's network is a one-layer width-32 GRU with tanh embedding and
readout activations to isolate the candidate equation while preserving the
E470 parameter layout. The actual incumbent is a two-layer GRU with GELU
embedding/readout. Therefore even a clean E471 result would not directly test
the incumbent architecture and cannot justify submission replacement or a
public-performance forecast. A follow-up would need an exact incumbent
architecture/control and matching inference implementation.
