# Activation × optimizer pilot

This directory holds a draft 16-arm trainer for the E467 sequence setup. It
varies one pointwise activation at both the input projection and residual MLP
hidden layer, and compares AdamW with the repository's MuonAdamW. The GRU's
internal sigmoid/tanh gates, zero-initialized residual output, anchor, loss,
normalization, clipping, data split, update order, and number of updates stay
fixed. Every arm has the same parameter count; activation functions add no
parameters.

The activation set is `tanh`, `GELU`, `ReLU`, `LeakyReLU(0.01)`,
`GELU(x)+0.1 sin(x)`, `GELU(x)+0.1 sin(x)+0.1(cos(x)-1)`,
`GELU(x)(1+0.5 torch.sinc(x))`, and channel-split GELU/LeakyReLU. The
channel-split variant divides the existing channels in half (floor split for
odd widths), applies one activation to each half, then concatenates them; it
does not add a projection or change width. `torch.sinc` uses PyTorch's
normalized definition `sin(pi*x)/(pi*x)`. These are pointwise choices only;
the screen makes no claim that they are KANs or a complete activation search.

Both optimizers use LR `0.001`, weight decay `0.01`, and betas `(0.9, 0.999)`.
MuonAdamW reuses `wnn.optim.muon_param_groups(scope="rnn")` and applies Muon
only to the two stacked GRU gate matrices, in three gate blocks. All biases,
input and output projections, anchor, and MLP weights use AdamW. The trainer
checks exact, non-overlapping parameter coverage before constructing Muon.

The full recipe uses the frozen exposed diagnostic cohort (24 fit and 16 test
components), 20,000 rows per sequence, batch size four, chunk size 256, one
epoch, and exactly 474 optimizer updates per arm. Evaluation retains the
existing global clipped weighted-Pearson metric. The test components are
previously exposed and this is a mechanism screen, not a fresh validation or a
public-score estimate. Seed `271828` and the shuffled fit order are fixed for
all arms. No checkpoint or hyperparameter selection is performed.

Run the correctness checks under the shared CPU/RAM guard:

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited \
  .venv/bin/python research/campaign_2026-10-05/activation_optimizer/tests.py
```

Run the fit-only real-data preflight (one optimizer update per arm on the same
B4 × 256-row chunk; it does not read test row groups or launch full fits):

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited \
  .venv/bin/python research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py \
  --preflight-real --output /tmp/activation-optimizer-preflight
```

The trainer imports `wnn.resources.configure_torch()` for the established
PyTorch thread/allocator controls. A full GPU fit must be launched through the
parent's reviewed, content-bound campaign runner and its stricter observed
VRAM/RAM/time guards; invoking `--device cuda` directly is not the launch
procedure. No full fits are part of this implementation task.

Dropout, residual scaling, and learning-rate schedules are intentionally held
out of this matrix. A separate, preregistered stage could test fixed paired
arms with dropout off/on, residual scale `1.0`/`0.5`, and constant LR versus a
fixed warmup-cosine schedule. Those are additional factors and must not be
combined with or retroactively used to select a winner from these 16 arms.

Controller result: all16fullGPUfits and the realGPUprobe completed successfully. See REPORT.md, RESULT.json, the independent review and eda/sixteen_v2/. The command examples above describe bounded implementation preflights; full fits used the content-bound runner. V1 seals were never GPU-launched; current v2 finite guards passed before admission.
