# H230 independent optimizer-fidelity source V5 (H273)

V5 is an isolated successor to the independently maintained H230 V4 source.
V4 and its seals remain unchanged. V4 supplies explicit Muon geometry,
learning-rate and orthogonalization-dtype controls; V5 adds a focused
raw-feature treatment correction for the H273 full-versus-raw family contrast.
It does not include data, weights, caches, or run outputs. The earlier V4
synthetic CUDA matrix-step profile is not a H273 actual-data profile. No H273
V5 profile or fit has been executed.

## Reference evidence and limits

The inspected reference endpoint is pinned to commit
`833aaf541155c591e930bbc6c640b8817e01a536`. Its final wave-6 trainer is
`research/iterations/2026-10-05-wave6/scripts/train_rnn.py`, SHA256
`26b2b96fe2d132d068f7afcc906d11aa28fbb1ab955d006125841cbd6e1cc9aa`. Its
default `--muon-gatewise none` orthogonalizes each packed GRU matrix as one
matrix. `hh` and `all` are separate gatewise geometries. The inspected trainer
uses Muon momentum 0.95, a whole-matrix update scale of
`0.2 * sqrt(max(rows, columns))`, and OneCycleLR with 5% warmup and a
100-fold final divisor. The source also applies Muon to the hidden output
projection while leaving the input/output projections and vectors in AdamW.

The reference provides no complete final `fbc1k_s5` training ancestry or
checkpoint hash. The wave-15 `composite256_lane.sh` recipe is plausible
predecessor evidence, not proof of the final teacher's command. The inspected
wave-18 teacher lane exports existing weights; it does not train them. Exact
full-feature teacher ancestry and the organizer-baseline source identity are
also absent. Accordingly, a V4 run can be described as an independently
implemented, method-compatible own-weight experiment, not an exact endpoint
replication.

Our previous all-data H230 parent used AdamW without EMA or bootstrapped
distillation. It remains an immutable comparison. The available own
`c20261003_combo_s3_f0` and `_s4_f0` checkpoints are clean-fold, independently
trained h256x2 initializers, not all-data teacher weights. The complete-row
H230 teacher cache provides predictions for all supplied target-index groups,
but its teachers were trained on the clean fold only. Full prediction coverage
does not make those teachers all-data trained. Any recipe using them must state
that scope and the seed-specific initializer ancestry explicitly.

## V4 implementation controls

- `--muon-gatewise none|hh|all` selects whole packed matrices, hidden-to-hidden
  gate blocks, or all recurrent gate blocks. The reference-matching geometry is
  `none`.
- `--muon-lr-mult` multiplies the peak learning rate of every Muon group. The
  OneCycle scheduler receives per-group peak rates, preserving the ratio during
  warmup and cooldown. Decoupled weight-decay coefficients stay equal; their
  per-step shrinkage therefore scales with each group's scheduled learning
  rate.
- `--muon-momentum` records the optimizer's starting Muon momentum. With
  OneCycleLR, beta1/momentum is then cycled on every optimizer group using the
  reference schedule's default 0.85-to-0.95 range.
- `--muon-ortho-dtype-policy
  native_bf16_if_supported_else_fp32_v1` chooses BF16 matrix orthogonalization
  only when CUDA reports BF16 support and compute capability is at least 8.0;
  other devices use FP32. This policy is independent of model autocast and is
  recorded with device capability, support result and selected dtype. Forced
  BF16 fails closed on unsupported hardware.

`MuonAdamW` is our own optimizer implementation. Its Newton-Schulz update uses
the same inspected polynomial coefficients, Nesterov momentum form and RMS
scale, while group construction and trainer integration remain in our source
style. V4 does not import or copy the reference trainer implementation.

## Checks and execution boundary

Focused CPU tests cover whole-matrix/gatewise parameter grouping, LR ratios over
OneCycle steps, weight-decay scaling, dtype-policy selection, parser spelling,
the default orthogonalization helper, and a finite optimizer update. The frozen
device-profile runner uses only synthetic input at the strong-parent's
`B16 x C1000 x 211` model-input shape, performs one optimizer update, and caps
PyTorch's CUDA allocator at 4.3 decimal GB. It reads no data and writes only a
small result JSON. These source checks do not replace an actual-device
matrix-step profile. Root review and a fresh resource admission are required
before that profile or any fit. A successful profile must record device
capability, BF16 support, selected orthogonalization dtype, finite output and
memory use. A full-shape fit requires a separate reviewed contract and
admission.

## V5 H273 row-weight correction

The inspected reference and V4 both zero the 99 derived event channels for
`--raw-only`, while computing the `--since-weight` multiplier from the
post-intervention model-input tensor. That makes the multiplier identically
one for raw-only fits, even though the same flag changes weights in full-view
fits. V5 preserves zeroed model inputs but derives `since_i0px` for row weights
from the original causal raw batch. The full and raw arms therefore share the
same row weights for the same rows. A missing raw batch fails closed when both
raw-only and since-weighting are enabled. This is an explicit correction to a
no-op in the inspected reference path, not a claim of literal replication.
