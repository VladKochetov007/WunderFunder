# Complete selected parameters — Muon / 30 epochs

**Engineering starting recipe, not a measured optimum.** User amendment on
2026-10-23 requests 30 epochs and Muon. No model, test, fit or profile was run
locally. R3 replaces the four unlaunched R2 cells; R2 remains historical at
`54b8fe6e`. No extra hyperparameter-search fits are implicitly allocated.

## Architecture

| Setting | Reference | Main candidate |
|---|---:|---:|
| Parameter count, source formula | 629,827,842 | 1,542,548,226 |
| Hidden width | 1280 | 1792 |
| Transformer blocks | 32 | 40 |
| Attention heads | 20 | 28 |
| Dimensions per head | 64 | 64 |
| Feed-forward width | 5120 | 7168 |

Both use the following complete architecture contract:

| Setting | Value |
|---|---|
| Inputs / tokenization | 112 raw features; one ordered row = one token |
| Normalization | Mean/std from all fitting components only; frozen FP32 buffers; std floor .001 |
| Position encoding | RoPE, base 10000, no learned position table |
| Per-layer attention window | 1024 causal rows including current row |
| Attention kernel | PyTorch math SDPA, 256 query rows per internal attention tile |
| Residual structure | Pre-LayerNorm, epsilon 1e-5; final LayerNorm |
| FFN | GELU, expansion 4; two dense projections |
| Dense biases | Enabled, as existing model source |
| Initialization | Cold start; existing PyTorch Linear/LayerNorm initialization, no pretrained ancestor |
| Outputs | Linear two-target head; no target feedback into inputs |
| Dropout | 0, including attention and residual branches |
| Activation checkpointing | Enabled per Transformer block |
| Gradient horizon | 512 rows; detach after each chunk |
| KV state | At most 1023 preceding rows/layer; reset at independent sequence boundary |
| Hidden hints / KD while fitting teacher | Disabled: dimension 0, hint weight 0, KD weight 0 |

Stacked sliding attention is not full 20,000-row attention at every layer.
Accumulation does not extend the 512-row backward graph. Cached states across
optimizer steps have the existing, explicit stale-state TBPTT approximation.

## Optimizer and schedule

| Setting | Selected value |
|---|---|
| Optimizer | Repository MuonAdamW with opt-in strict-FP32 Newton–Schulz |
| Muon parameters | Q, K and V separately; attention output; FFN input/output matrices |
| Auxiliary AdamW parameters | Input projection, output head, norms and biases |
| Peak LR, all groups | 0.0003 |
| Muon LR convention | AdamW-matched RMS: multiply orthogonalized update by `0.2 * sqrt(max(rows, cols))` |
| Muon momentum | .95, Nesterov-style update; constant through training |
| Newton–Schulz | 5 iterations; coefficients (3.4445, -4.7750, 2.0315); epsilon 1e-7 |
| Auxiliary AdamW betas / epsilon | (.9, .95) / 1e-8 |
| Decoupled weight decay | .01 on matrices; 0 on norms and biases |
| Gradient clipping | Global L2 norm 1.0 after complete accumulation and DDP synchronization |
| Warmup | Linear, first 5% of planned updates, approximately 1.5 epochs |
| Decay | Cosine over remaining updates; no plateau/validation-triggered schedule |
| Final LR | .00003 (10% of peak), reached on final actual update |
| Precision | Parameters, gradients, state, attention, NS intermediates FP32; TF32/autocast off |
| Quantization / EMA / SWA | None |

Let `N` be the number of fitting sequences, and `B = ceil(ceil(N/8)/2)`
the equalized rank batches per epoch. There are `ceil(20000/512)=40` chunks,
five accumulation windows per sequence batch, and `T=30*B*5` total updates.
`W=ceil(.05*T)` is computed separately from each cell's actual fitting membership.
At zero-based actual update `s`:

```text
if s < W:
    lr(s) = 3e-4 * (s + 1) / W
else:
    u = clamp((s - W) / (T - 1 - W), 0, 1)
    lr(s) = 3e-4 * [0.1 + 0.9 * (1 + cos(pi*u)) / 2]
```

The scheduler advances once after each successful optimizer update, never after
each chunk or epoch. Its position and all optimizer groups are checkpointed.
There is no additional momentum warmup, restart or automatic LR multiplication
by the number of GPUs. Initial and final LR indexing have prepared remote checks.

## Data, batch and evaluation

| Setting | Value |
|---|---|
| Full-data epochs | 30 per fit; 120 teacher epochs across four fits |
| Cells | F0/101001 and F2/101002 for both model sizes |
| Devices / parallelism | Eight B300s, one process per GPU, replicated DDP |
| Batch per GPU | Two independent complete sequences |
| Chunk rows / accumulated chunks | 512 / 8 |
| Global sequence slots | 16; no claim of 128 independent sequences from accumulation |
| New rows per full update | 65,536, before excluding warmup/padding |
| Final window rows | 3,616 per real sequence; normalize by its actual target-weight mass |
| Training objective | WMSE weighted by absolute clipped target; targets clipped to [-2,2], predictions unclipped in training loss |
| Training eligibility | Labeled rows >=99 from fitting components only |
| Evaluation eligibility | Official is_scored and row>=99; pooled `global_wp_clip2_v1` |
| Normalization/cache ancestry | Exact fitting membership, source and split hashes; no evaluation components |
| Primary endpoint | Fixed last epoch, index 29; no outer-fold best-checkpoint selection |
| Recovery | Whole-epoch checkpoints; partial epoch replay; same exact source/config/data/runtime |
| Quality evidence | Both paired teacher deltas and component bootstrap intervals; folds already exposed |

## Why these choices, and what remains unknown

The author recommends hidden-matrix Muon plus auxiliary AdamW and separate Q/K/V
orthogonalization ([Muon README](https://github.com/KellerJordan/Muon),
[author discussion](https://kellerjordan.github.io/posts/muon/)). Five NS rounds
and momentum .95 are published defaults, not results from this dataset.
The LR scale above follows the explicit
[AdamW-matched convention](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html),
not the original `lr=.02` convention. Using .02 with the chosen RMS convention
would be a different, much larger step. The base LR 3e-4 and beta2 .95 are
intentional new engineering starting values, not transferred tuned R2 values.

Five-percent warmup gives the random deep model roughly 1.5 passes before peak
LR. The .1 floor keeps all late updates active during the requested 30 passes;
it has not been shown better than a zero floor. Clipping and zero norm/bias decay
are explicit safeguards; gradient norms, clipping frequency and named group LRs
must be read from the actual run, not assumed healthy from these constants.

Accumulation8 amortizes expensive strict-FP32 Muon steps. Source-shape arithmetic
for the 1.54B model is approximately 69.1 TFLOP of NS GEMMs per local update,
versus roughly 101 TFLOP of dense model work with eight chunks and activation
recomputation, excluding attention and many overheads. These are arithmetic
estimates, not timings. Compared with accumulation2, updates per pass fall4x;
gradient averaging and noise also change, and adjacent rows are correlated.
Both model sizes use the same explicit R3 recipe. No optimizer sharding,
distributed NS acceleration, linear speedup or quality advantage is claimed.

Before any fit, the receiver runs synthetic correctness/recovery checks and
12-update exact-shape one/eight-GPU profiles, plus a separate eight-GPU peak-LR
cold-start stress profile. The latter has warmup=0 only for engineering work;
its weights are discarded and cannot initialize a scientific endpoint. Log
optimizer CUDA-event time separately to see whether accumulation is sufficient.
Neither these short profiles nor source arithmetic select the best quality LR.

If tuning is needed after actual learning curves arrive, freeze a separate
allocation with fitting-only inner component splits and a matched LR/batch or
cooldown comparison. Report the cost and every tried setting; do not silently
expand the four-fit panel or select the final checkpoint on outer F0/F2.
