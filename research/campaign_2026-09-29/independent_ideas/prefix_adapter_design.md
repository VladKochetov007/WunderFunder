# Prefix-conditioned rank-8 input adapter: frozen design

Design only; no GPU run, dataset scan, or edit to the active trainer. The screen belongs on the root Challenge 2 clean fold manifests, beginning with `research/campaign_2026-09-29/artifacts/splits/fold0.json`. Parent models and all three adapter arms must inherit its component exclusions, fold-specific propensity, fixed checkpoint policy, and batch 32. The public incumbent's 0.6622 association remains user-reported; local score gates cannot establish a 0.6660 expectation.

## Exact causal computation

Use the existing `build_inputs` order `[112 raw, 99 causal event]` (211 features), then the frozen fit-only normalization buffers `mean, std`. Let `x_t=(input_t-mean)/std`. Keep an input-prefix sum `s_t=Σ_{j=0}^{t-1}x_j` and a row count `n_t=t`; reset both at each sequence boundary. Define `c_{t-1}=s_t/(256+n_t)`. This is a shrinkage running mean of **earlier** normalized inputs; the current row does not enter its own gate. Warm-up rows update the state. The 256 pseudo-count is fixed before scoring and stores no fitted per-sequence parameter. Sequence IDs are reset keys only, never inputs.

Conditional rank-8 arm:

```text
g_t = tanh(C c_{t-1} + b_c)                    # 8
v_t = V x_t                                   # 8
z_t = GELU(W x_t + b + U (g_t elementwise v_t)) # 128
```

Initialize `U=0`, `b_c=0`, draw `C,V` with the same seeded fan-in rule as controls. The initial network then reproduces the frozen parent exactly. `U` receives a gradient on the first update; `C,V` receive gradients after `U` moves. Apply no labels, future events, masks, or current-row values to `c_{t-1}`. In code use a model-owned prefix state, not a global cache. Retain the ordinary GRU/skip head.

For each training chunk of length 2,000, compute the normalized input once. Under `no_grad`, form `previous = cat([zeros_like(x[:, :1]), x[:, :-1]])`, then `cprev=(sum0 + cumsum(previous,dim=1))/(256+count0+arange(T))`; carry `sum1=sum0+sum(x)` and `count1=count0+T`. Shifting before the sum gives an exact computational exclusion of the current row, without `cumsum(x)-x` cancellation. The context derives from input only and needs no autograd tape. It must carry across TBPTT chunks and be detached with the GRU state. This avoids a Python loop over 20,000 rows and avoids storing a full-sequence `(32,20000,211)` context tensor. Directly compare one 20,000-row call, ten 2,000-row calls, and a one-row streaming update on a synthetic sequence before any scoring. Use float32 in all paths and freeze a parity tolerance before evaluation.

## Matched controls and budget

| Arm | Residual path | Trainable extra parameters at d=211, h=128 | Approximate extra multiply-adds/row |
| --- | --- | ---: | ---: |
| Conditional rank 8 | `U[(C c) ⊙ (V x)]` | `8*211 + 8*211 + 128*8 + 8 = 4,408` | ~4.4k |
| Unconditional rank 13 | `U13(V13 x)` | `13*211 + 128*13 = 4,407` | ~4.4k |
| Conditional diagonal rank 10 | `W[x ⊙ (1+b + D tanh(C10 c))]` | `10*211 + 211*10 + 211 + 10 = 4,441` | ~4.5k |
| Frozen parent | no residual | 0 | 0 |

The unconditional arm receives the same prefix computation during training and streaming as a **detached unused control** so data path and allocated context memory are matched; an optimized packaged version may omit it. The diagonal arm changes only per-input scaling; the conditional low-rank arm can rotate cross-feature interactions through `V` and `U`. Set `U13=0` and `D=b=0` for identity at initialization. Keep identical parent weights, fit rows, mask weights, batch size, optimizer steps, schedule, AMP policy, seed, and loss. In the stage-1 GCLP head fit, freeze `inp,gru,adapter` in all arms; stage 2 unfreezes all, matching the existing two-stage recipe's treatment of the encoder. A frozen-parent one-epoch adapter-only fit is a cheap **mechanism screen**; its failure does not prove a jointly retrained adapter has no value.

## Cheapest complete falsifier

1. Wait for `c20260929_since_s4_f0` and the corresponding clean fold-0 pair to finish. Add adapter model code in a separate working area or after the active trainer is idle. Freeze its implementation hash, parameter counts, and command line before seeing fold-0 scores.
2. CPU/no-data unit checks: zero-residual parent identity, warm-up/reset, chunked versus one-row context/prediction, no current-row gate effect when a row is perturbed, and finite outputs. This catches leakage and state bugs before GPU use.
3. Monitored **historical batch 32** GPU smoke on exactly one synthetic 2,000-row chunk for all three arms, without training data. The active clean parent job has observed 4.56 GB total device memory; each float32 `(32,2000,211)` tensor is 54 MB, and shifted input, prefix scan, context, and saved activations can add roughly 0.2-0.3 GB at once. The margin to 5 GB is only about 0.44 GB. If total observed VRAM exceeds 5 GB, keep batch 32 semantics with a matched gradient-accumulation control or stop; do not silently reduce the historical batch.
4. Fold-0 screen from the same clean fixed-epoch seed-3 parent: freeze the parent base parameters and fit each zero-initialized residual arm for one predetermined epoch on the identical component-excluded fit chains. Freeze a fit-only inner checkpoint epoch, using no outer-fold evaluation loader during fitting. The package-facing output is the fixed mean of the two clean since/since members, replacing only this seed-3 arm. Compare on the same exact scored rows and masks. If the frozen-parent screen is underpowered, run one *preregistered* full matched 4-epoch triplet only if the parent-only screen shows a consistent direction or diagnostic evidence of unlocked conditioning; record the choice before looking at fold 1.
5. Screen threshold: conditional minus unchanged pair >=+0.0005 Global WP **and** conditional minus the better of unconditional rank 13 and diagonal rank 10 >=+0.0005. Require no class delta below -0.0005, no parity failure, and nonnegative exact-mask and softened-mask (λ=0.02) direction. Equal mask/row/fold provenance and one chain-component bootstrap SD are mandatory. A sign-AUC improvement alone fails. Freeze candidate and epoch before folds 1-3; confirmation uses the campaign's positive-each-fold, bootstrap interval, class floor, and independent expectation audit gates. The historical block-average metric may be logged diagnostically; Global WP is authoritative.

## Implementation map and traps

- `src/wnn/nn.py`: add explicit `enc="prefix_lowrank"`, `"lowrank"`, `"prefix_diag"` paths or a separate adapter enum. Keep the default `linear` checkpoint schema untouched. Pass a state tuple `(gru_h, prefix_sum, count)` only for adapter encoders; `forward_chunk`, `predict`, and `detach_state` in `scripts/train_rnn.py` must preserve it. Confirm `fit_normalization` occurs before using prefix context.
- `scripts/train_rnn.py`: extend `--enc`, `make_model`, checkpoint argument inheritance, adapter initialization from a clean linear parent, and named `--freeze` prefixes. The current `--inherit-init-preprocessing` forces the parent's `enc="linear"`; a new narrowly scoped adapter-init mode should allow this architecture change while requiring identical feature width, split-manifest SHA, fixed epoch, and no-eval provenance. Do not use `strict=False` on arbitrary checkpoint keys without printing/validating the exact expected adapter-only missing keys.
- `native/src/python.rs` and `native/src/rnn_fast.rs`: the current `weights_from_dict` takes known base keys and silently discards leftovers, which would make an adapter checkpoint appear deployable while predicting the old model. Add explicit adapter schema/shape validation and reject all unknown keys. `FastRnn::step` must compute normalized `x`, the prefix gate, projection residual, GELU/GRU/head, then update prefix sum/count. Reset zeroes both prefix and GRU state. The input mean/std are currently folded into the base projection; store enough normalization coefficients for the adapter and prefix, or fold the adapter's `V,C` with mathematically equivalent biases while keeping prefix semantics.
- Package metadata must identify the adapter architecture and fixed rank. `scripts/package_submission.py` exports all adapter arrays, but the native loader and parity checker must consume every one. `solution_multi.py` may keep fixed membership/weights if the native constructor handles the schema. Test batch/Python streaming/native v2/AVX2/v3 with identical raw feature order and repeated deterministic runs. The baseline two-member public runtime was 36m27s; ~4.4k extra arithmetic per row is roughly 1% of two h128 two-layer GRUs, but only a one-vCPU scorer-image timing establishes the <=52 minute gate.

This is an architecture hypothesis. The current data contain no matched adapter scores or expected-public forecast.
