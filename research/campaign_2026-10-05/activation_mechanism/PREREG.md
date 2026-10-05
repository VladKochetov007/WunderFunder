# E473: matched activation-form mechanism screen

Status: implementation and fit-only CPU preflight only; no full fit or held-out
scoring has been run. This is a single-seed mechanism screen with exposed
diagnostic components, not a public-score estimate.

## Fixed contrasts

Use one activation choice in both E470 nonlinear locations: the input embedding
and the correction-head hidden layer. Keep its zero-initialized correction
output, recurrent GELU-free internals, output anchor, data, loss, optimizers,
and training recipe fixed. Compare these five pointwise formulas:

| Name | Formula | Intended contrast |
|---|---|---|
| `gelu` | `GELU(x)` | Exact PyTorch GELU control |
| `gelu_plus_sine_01` | `GELU(x) + 0.1 sin(x)` | Bounded oscillatory curvature term |
| `gelu_plus_linear_01` | `GELU(x) + 0.1 x` | Same local derivative increase at zero, with a linear term instead of periodic curvature |
| `scaled_gelu_11` | `1.1 GELU(x)` | Output and gradient scale control |
| `jax_tanh_gelu_plus_sine_01` | `GELU_tanh(x) + 0.1 sin(x)` | Approximate-versus-exact GELU control under the same sine term |

`GELU_tanh(x)` means the standard tanh approximation,
`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`, equivalently
`torch.nn.functional.gelu(x, approximate="tanh")`. Cross all five activations
with AdamW and MuonAdamW for ten arms. Seed is fixed at 271828; no dropout,
learning-rate, schedule, initialization, or architecture factor is added.

At zero, the first derivative is 0.5 for exact GELU and 0.55 for scaled GELU;
the three additive variants have derivative 0.6. Comparing the sine and
linear additions distinguishes their curvature away from zero while matching
their local slope. The `1.1*GELU` arm separates output rescaling from those
additions; the tanh-GELU-plus-sine arm checks the framework approximation
choice. This design does not isolate every possible higher-order derivative
effect and does not imply an input periodicity mechanism.

## Matched recipe and limits

Reuse the immutable E470 loader, normalization, weighted MSE, metric, and
optimizer partition from
`activation_optimizer/fit_activation_optimizer.py` through a thin subclass
that swaps only parameter-free activation modules. Parameters and names are
identical across all arms, with 8,235 trainable parameters. Each arm uses
seed 271828, the frozen 24-fit/16 exposed-test cohort, 20,000 rows per
sequence, B=4, 256-row chunks, carried/detached recurrent state, 474 AdamW or
MuonAdamW updates (one epoch), and `global_wp_clip2_v1`. The ten-arm screen
uses the test components as exposed diagnostics only. It cannot support a
public performance forecast or model-selection claim.

## Checks and commands

Before any full run, require tests of formulas and zero derivatives, finite
outputs/derivatives at large positive and negative inputs, exact seeded
parameter initialization and model output/state parity for the GELU control,
preserved zero-initialized residual output, equal parameter counts, and one
fit-only real B4×256 CPU optimizer update per arm. No test rows are loaded by
the preflight.

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited \
  .venv/bin/python research/campaign_2026-10-05/activation_mechanism/tests.py

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited \
  .venv/bin/python research/campaign_2026-10-05/activation_mechanism/fit_activation_mechanism.py \
  --preflight-real --device cpu \
  --output research/campaign_2026-10-05/activation_mechanism/cpu_preflight
```

The full-run entry point exists for parent review but must not be launched by
this actor. It requires a separate content seal, a tracked campaign run, and
independent review of all ten results.
