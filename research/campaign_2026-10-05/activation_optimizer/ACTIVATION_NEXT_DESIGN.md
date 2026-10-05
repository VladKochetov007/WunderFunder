# Next activation study: local placement on the incumbent GRU topology

Status: design only; no model code, fits, or scores were produced for this
proposal. It follows the one-seed, one-epoch E470 screen with a balanced
two-seed placement study. It does not claim or estimate the 0.7000 expected
public score.

## Decision this experiment can answer

E470 changed the embedding and correction-head activation together. Its
sixteen results therefore cannot say which site produced the optimizer-
dependent pattern. E471's custom recurrent candidate did not reach its first
update: alpha-zero CUDA parity failed before fitting. E471 is a recurrent
backend/parity incident, not an activation score and not evidence against
embedding or readout function shaping.

Use the actual incumbent *topology*: linear input projection, GELU embedding,
two standard `nn.GRU` layers, concatenation of the recurrent output with the
embedding skip, and a `Linear(2H,H) -> GELU -> Linear(H,2)` head. For a
time-bounded screen use `H=64` and the same 42-feature, frozen 24-fit/16-test
cohort as E470. This is topology-faithful but not width-, feature-, data-, or
training-budget-faithful to the full RNNModel default (`scripts/train_rnn.py`
uses H=256, batch 32 and two epochs). The H64 model has 61,826 trainable
parameters, versus 932,354 at H256
with 42 inputs. All proposed activation arms have exactly the same H64 shape
and 61,826 parameters, so within this screen activation computation does not
change capacity. No result should be described as an H256 result.

## Fixed twelve-fit matrix

Use two paired seeds, `271828` and `271829`. At each seed, create six models
from one cloned GELU/GELU initialization:

| Arm | Embedding nonlinearity | Head hidden nonlinearity |
|---|---|---|
| `base` | GELU | GELU |
| `sine_embed` | GELU + 0.1 sin(x) | GELU |
| `slope_embed` | GELU + 0.1 x | GELU |
| `sine_head` | GELU | GELU + 0.1 sin(x) |
| `slope_head` | GELU | GELU + 0.1 x |
| `sine_both` | GELU + 0.1 sin(x) | GELU + 0.1 sin(x) |

The shared base is fit once per seed, not separately for each site. The five
nonbaseline rows plus the base give six fits per seed and twelve total. The
site-local comparisons are `sine_embed - base` versus `slope_embed - base`,
and `sine_head - base` versus `slope_head - base`. Sine and linear additions
both have derivative 0.6 at zero (GELU contributes 0.5; each addition adds
0.1), while their derivatives away from zero differ: `0.1 cos(x)` versus
`0.1`. These comparisons ask whether a sine-shaped local derivative profile
helps beyond its local slope. `sine_both` tests whether applying that fixed
shape at both sites adds or interferes; report its interaction as
`sine_both - sine_embed - sine_head + base` within seed.

Use only the activation at the specified site; do not add trainable activation
coefficients. Keep one fixed optimizer across every arm: AdamW with
`lr=0.001`, `weight_decay=0.01`, `betas=(0.9,0.999)`, global gradient clip
1.0. Keep the E470 weighted MSE, frozen feature normalization, fit order,
B=4, 256-row chunks, carried state, one epoch and 474 updates. For direct
continuity, freeze constant LR in this 12-fit design. Do not add dropout,
warmup/cosine scheduling, Muon, or width changes here. The full pipeline's
default scheduler is OneCycle; the E470 mechanism harness instead used
constant LR. This screen is explicitly the latter fixed recipe, not an exact
replica of the incumbent's full training schedule. A separate schedule
contrast is specified below.

For each seed, shuffle the exact 24 fit row groups once using a seeded local
generator and reuse the order across all six arms. Reset Torch's RNG before
building each model; clone the base state dict and assert exact equality of
every trainable parameter before replacing parameter-free activation
modules. The 16 test components have been exposed by earlier campaign work;
use them only for the predeclared within-seed paired diagnostic. Report every
component/target delta and both seed tables; do not pool rows into a confidence
interval or choose later settings from these same scores.

## Cost and backend

Each fit executes 474 optimizer updates over 24×20,000 fit rows, or 480,000
row positions per fit. Twelve fits cost 5,688 updates and 5.76 million fit-row
positions. The exposed diagnostic pass adds 12×16×20,000 = 3.84 million test
row positions. It is one optimizer, one epoch, one small cohort, and two
seeds; this is not a public evaluation. The H64 42-feature cohort array uses
about 134 MB in float32 for X across all forty components, plus small targets
and masks. Model weights total about 0.24 MiB per arm before gradients and
Adam states. cuDNN workspace and saved activations still require measurement.

The recurrent core remains ordinary PyTorch `nn.GRU` and uses its normal
cuDNN implementation on CUDA; only pointwise activation functions change.
The sine treatment adds one `sin` and one addition at each modified site and
adds no weights. The linear-slope control adds a multiply and addition and no
weights. GELU remains at untouched sites. A parent-run B4×256 two-layer H64
parity/throughput probe must first establish allocation and full-precision
policy under the current GPU cap; require <=2 GiB PyTorch allocator and <=3
decimal GB observed device use, leaving headroom under the campaign 5 GB
limit. Use one active GPU worker, at most 2 CPUs and 6 GiB host memory for
that probe, then set fit timeout from its measured update time plus evaluation
and a generous margin. Do not infer twelve-fit wall time from E471's failed
custom-loop probe.

Training/evaluation still use the same exposed E470 feature cohort; no test
row group enters loss, normalization, scheduling, or calibration. Save
complete checkpoints, predictions, per-arm LR trace, preclip gradient RMS,
per-family parameter-update RMS, elapsed fit/evaluation times, peak allocated
and observed VRAM, plus exact source/cohort/data hashes. Gradient/update
telemetry helps detect an activation effect that is actually just a change in
effective step size; it is diagnostic and must not trigger post-hoc retuning.

## What it falsifies, and what it cannot

The narrow curvature hypothesis is that the sine derivative profile adds
something beyond the matched local slope. It is not supported under this
recipe if, for both seeds and at both localized sites, sine-minus-linear has
nonpositive pooled GlobalWP and no consistent positive component pattern.
That outcome rejects this sine amplitude, these placements, and this
one-epoch recipe as evidence for a sine-specific benefit. It does not reject
all smooth activation shaping. If sine and linear move together relative to
baseline, the result points to local slope/amplitude change rather than
sine-specific curvature. If only one site moves, it localizes the effect; if
the combined arm differs from the sum of the two local deltas, report the
observed interaction without selecting the combined arm as a winner.

The two seeds and already-exposed component panel remain weak evidence. Any
positive pattern needs a fresh, predeclared component evaluation and more
paired seeds before performance claims. The 12 fits do not test dropout,
OneCycle versus constant LR, LR matching, H256 capacity, or the full
211-feature input. Width is constant to prevent capacity from changing with
the activation, but an H64-to-H256 transfer is untested.

## Separate follow-ups for LR, schedule and dropout

Do not bundle these variables into the 12 activation fits. If time remains,
register each as a new balanced block before spending compute:

1. **Schedule block (8 fits):** cross exact GELU versus embedding-only
   GELU+sine with constant versus OneCycle LR, for two paired seeds. Use
   OneCycle `max_lr=0.001`, `pct_start=0.05`, `div_factor=25`,
   `final_div_factor=100`, `cycle_momentum=False`, total steps 474; keep
   AdamW betas and weight decay fixed. Match the constant rate to the
   arithmetic mean of that precomputed OneCycle LR trace. Record the trace and
   integrated LR. This estimates schedule and activation-by-schedule effects
   without conflating LR area or beta1 cycling. The exact incumbent OneCycle
   configuration cycles momentum by default; reproducing it answers a
   different question and should be separately preregistered.
2. **Effective-LR block (8 fits):** with exact GELU and embedding-only
   GELU+sine, cross the two activations with nominal versus per-activation
   scalar LR multipliers calibrated on a preregistered fit-only B4×256 batch
   so each activation's first-update global trainable-parameter RMS matches
   the exact-GELU nominal reference. Repeat for two seeds, using the same
   OneCycle trace shape and weight decay with `cycle_momentum=False`. Freeze
   the multipliers before any test read and report per-family update RMS,
   since matching one initial global update does not ensure later or
   family-wise equality. If schedule
   questions are still open, do that block separately with the common
   calibrated LR fixed.
3. **Dropout block (12 fits):** cross exact GELU versus embedding-only
   GELU+sine with `p=0`, sequence-locked channel dropout `p=0.1`, and
   independent rowwise dropout `p=0.1`, for two paired seeds. Apply masks
   only to the GRU input (`z_rnn`); keep the original unmasked `z` in the
   `[recurrent, z]` head skip. Use inverted dropout in training and no masks
   at evaluation. Locked masks have shape `B×1×H`; rowwise masks have shape
   `B×T×H`. Keep optimizer and the frozen LR schedule identical. This
   separates regularization from temporal noise without changing the head
   skip path.

These blocks are additional compute: 28 more fits (13,272 optimizer updates)
if all are run. The present 12-fit proposal is the priority and does not
silently spend that additional budget. A fit-only dry update and a guarded
production-shape CUDA probe are admission requirements before any block.

## Precision and E471 containment

The E471 incident failed CUDA alpha-zero comparison before any optimizer
update: maximum prediction absolute difference was 1.2124e-4 against
`atol=2e-5, rtol=2e-4`. This concerns its custom recurrent scan versus fused
cuDNN GRU. E473 keeps the fused `nn.GRU` path and cannot be taken as a
successful or failed E471 candidate result.

Local PyTorch is `2.14.0+cu130`. Read-only inspection showed float32 matmul
precision `highest`, CUDA matmul TF32 disabled, but
`torch.backends.cudnn.rnn.fp32_precision == "tf32"`; the local
`torch.backends.cuda.matmul.fp32_precision` inherited value is `"none"`.
This backend asymmetry is a plausible cause, not confirmed from the failed
result alone. PyTorch documents the newer per-backend controls and lists GRU
and LSTM among operations affected by them in its [CUDA TF32 precision
notes](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices). A separate temporary diagnostic wrapper can
save/restore the current settings, disable autocast, require FP32 tensors and
set `torch.backends.cuda.matmul.fp32_precision="ieee"`,
`torch.backends.cudnn.fp32_precision="ieee"`, and
`torch.backends.cudnn.rnn.fp32_precision="ieee"`. The local setters accepted
and restored these values without launching CUDA. Do not mix the deprecated
`allow_tf32` properties with the new `fp32_precision` settings; the docs warn
against mixing them. IEEE precision can make both sides use FP32 internally
but does not promise bitwise equality of fused cuDNN and a scripted loop; the
original tolerances must pass on the GPU before any E471 continuation.

If E471 still fails equation/state/gradient parity under the same explicit
IEEE policy, pause only its custom recurrent-cell activation family and
investigate exact gate/bias/order parity or use a deployment-supported cell.
Do not convert that backend mismatch into evidence against pointwise
embedding/head activations, additive splines, or other KAN-inspired local
function combinations that leave the standard GRU intact. Reopen the
recurrent family only after a separately identified diagnostic passes
full-sequence and chunk/state parity, finite gradient/update checks, and
measured inference-cost limits. Conversely, if parity passes but the bounded
alpha variants underperform in repeated fresh-component fits, close those
specific recurrent formulas; that still does not close the separate local
activation/spline families.

## Matching inference is mandatory

All trained variants must be evaluated using the exact same formulas as
training: exact GELU at baseline sites, `GELU(x)+0.1 sin(x)` for sine sites,
and `GELU(x)+0.1x` for slope controls. Export state dicts with explicit
activation metadata. Check complete-sequence outputs and every 256-row
chunk/state boundary against the deployment implementation on identical FP32
inputs, including values close to activation saturation. The changed variants
cannot be promoted if the native/inference implementation silently falls
back to GELU, evaluates a different approximate GELU, or changes sequence
state. This head/embedding study does not require a custom recurrent cell;
standard GRU gate and hidden-state formats stay intact.
