# PR32 Blackwell execution amendment, 2026-10-24

The user authorized the full four-teacher PR32 run on eight B300 GPUs, preceded
by measured CPU/GPU/memory/time calculations. The previous four-hour campaign
limit does not apply to this full run. On October 24 the user explicitly asked
to compare precision modes and other performance parameters, including Blackwell
alternatives. This amendment preserves the original R3 strict-FP32 recipe and
its evidence; alternative execution recipes receive distinct source/config/runtime
identities. It supersedes strict-FP32-only policy for these explicitly requested
alternatives, not for unrelated research.

The scientific allocation is four teachers: 630M and 1.54B, each on F0 and F2,
30 epochs, fixed final endpoint. It does not activate the conditional 40-fit plan.
Data membership, fitting-only normalization, causal 1024-row local attention,
512-row detached TBPTT, label masks, official evaluation and optimizer settings
remain frozen unless an explicit deviation is recorded. Baseline global batch
is 16 independent sequences, not 128; eight accumulated time chunks are not
independent sequence examples.

## Measured baseline

Pinned PR head: `9d901dc3bfb6652884c8f82ed61997eb23e24754`. Receiving-host
test-fixture repair: `faa478aba4f0e8154754a4c07ef85e21a2cbaeb5`; training code
and configurations unchanged. Acceptance: 50 passed, zero skipped; two-rank
objective and Muon checks passed. The legacy optimizer test needed a deep copy
of its state dict to avoid shared momentum storage between reference and resume.

| Model | GPUs | Updates | Training-loop seconds | Peak reserved GiB/GPU |
|---|---:|---:|---:|---:|
| 630M | 1 | 12 | 45.9501 | 10.1152 |
| 630M | 8 | 12 | 47.5750 | receipt retained on node |
| 1.54B | 1 | 12 | 64.8891 | 22.2773 |
| 1.54B | 8 | 12 | 65.7728 | 28.3867 |

These are engineering profiles, not quality fits. Timings exclude cold process
startup, model initialization, normalization, complete-epoch evaluation and
checkpoint export. Update-window telemetry also excludes batch loading. The
profile receipt includes batch loading and final finite-state checks.

Host: eight B300, capability 10.3, 275040 MiB/device, NVSwitch; 256 logical CPUs,
approximately 2 TiB host RAM. Python 3.12.14, PyTorch 2.14.0+cu130, CUDA 13.0.
F0 has 11940 fitting sequences and 468 evaluation sequences; F2 has 11965 and
443. At global batch 16 the original four teachers require 448500 updates total.
Fitting-only normalizers were completed and their source/membership identities
retained, so unchanged attempts reuse the verified normalization cache.

## Bounded engineering selection

Compare native FP32, TF32 model compute, BF16 model compute with FP32 master
weights/gradients/loss/optimizer, and requested cuBLAS FP32 emulation via BF16x9.
BF16x9 keeps FP32 external tensors but is not a bitwise-equivalent IEEE mode.
The installed PyTorch rejects the `bfx9` precision setter; any environment-based
emulation request must be recorded, including its possible effect on Muon GEMMs.
An environment request is not evidence of actual kernel dispatch.

The initial sweep uses the real 1.54B/F0 recipe with one GPU per profile, batches
2/8/16/32, native and emulation requested: eight simultaneous profiles, six
updates each, 900 seconds maximum per worker, 512 GiB aggregate host-memory cap.
It preserves raw logs and exits, writes no scientific checkpoint, and cannot
admit a full fit. Follow-up finalists use warmed representative profiles and
checks of finite gradients/state, causality, checkpoint recomputation and resume
identity. Attention alternatives must preserve the exact rolling causal mask;
unsupported kernels fail explicitly. Compilation time counts as execution cost.

Prefer configurations by measured time to deliver all four evaluated teachers,
including startup, communication, validation and checkpoint export, with resource
headroom. GPU utilization and allocated memory alone are not the objective.
Layouts W8/B2, W4/B4 and W2/B8 preserve 16 sequence slots but need explicit
layout/source admissions. Larger global batches change the optimization recipe
and update/scheduler counts and must be identified as such. Short profiles cannot
establish equal convergence or a model-quality improvement.

Results and actual full-fit admission will be appended after measurements.
The [PR32 engineering update](https://github.com/VladKochetov007/WunderFunder/pull/32#issuecomment-6376581098)
records the first receiving-host results. No full scientific fit had started at
that update.

Primary references: [NVIDIA cuBLAS emulation](https://docs.nvidia.com/cuda/cublas/index.html),
[PyTorch CUDA precision semantics](https://docs.pytorch.org/docs/stable/notes/cuda),
[PyTorch FlexAttention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html).
