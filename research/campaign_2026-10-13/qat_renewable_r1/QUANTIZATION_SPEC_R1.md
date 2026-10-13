# Versioned QAT and archive specification

This note freezes the Python representation contract for the first QAT code
milestone. It describes encoding and transport only; it makes no model-quality,
callback-speed, or submission claim.

## Policies

The legacy `recurrent` and `recurrent_head` presets retain their version-1
behavior: all `gru.weight_ih_*` and `gru.weight_hh_*` matrices use per-row Q8;
`recurrent_head` also quantizes `head.0.weight` as two independent Q8 blocks.

The new version-2 `--qat recurrent_w4` policy uses per-row Q4 for every GRU
input and recurrent matrix. `--qat input_w4_recurrent_w8` uses per-row Q4 for
`gru.weight_ih_*` and Q8 for `gru.weight_hh_*`. Both new policies use Q8 for
`head.0.weight`, split by input columns at the hidden width: hidden features
first, projection features second. `inp.*`, all biases, `head.2.*`, and other
parameters stay float32 in the training checkpoint and fake-quantized model.

Q8 uses signed codes `[-127, 127]`; `-128` is invalid. For a nonzero row its
float32 scale is `max(abs(w)) / 127`, bounded below by the smallest positive
float32 subnormal. The all-zero row has zero scale and zero codes. Q4 follows
the same rule with divisor `7`, signed codes `[-7, 7]`, and invalid code `-8`.
Codes use NumPy float32 arithmetic and nearest-even rounding (`rint`) before
clipping to the signed range. A zero scale is valid only with all-zero codes.
The represented float32 row is `code * scale`.

QAT applies represented weights in the forward pass and an identity straight-
through gradient to latent float32 parameters. Scales, rounding, and clipping
have no gradient. The represented-weight cache is keyed by parameter identity,
version, and data pointer; optimizer updates and averaging paths explicitly
invalidate it. Checkpoints store latent parameters and optimizer state, never
the derived cache, so continuation configures the selected policy again from
the frozen CLI recipe.

## Archive and native import

Packed archive version 2 stores Q8 codes as one unsigned byte per code, in
row-major order, preserving the signed byte's bits. Q4 codes are flattened in
row-major order, encoded as signed two's-complement nibbles, and packed with
the first code in the low nibble and the second in the high nibble. If the code
count is odd, the final high nibble is zero padding. Row scales are separate
float32 arrays and keep their exact bits. The archive metadata records the
preset, matrix and block shapes, bit width, and code/scale fields.

The reader expands packed codes once and returns the existing native mapping:
`<matrix>.codes` as a row-major `int8` array plus `<matrix>.scales` as `float32`.
For Q4, the signed `[-7, 7]` values and Q4 scales are preserved; the native
`DynQ8::from_rows` import path must not derive new scales from decoded weights.
This is exact transport through a byte kernel, not a native INT4 kernel or
evidence of an INT4 speedup. Archive bytes, resident storage, and callback
runtime are separate measurements.

The native `q8vnni*` inference path keeps its existing I16 input-projection
representation and dynamically quantizes recurrent/head activations to A8.
Training leaves the input projection float32 and recurrent state float32;
these activation and projection differences remain outside this weight-only
QAT milestone and require separate controlled work.

## Implemented entry points

- `src/wnn/quantization.py`: row encoders, policy selection, and exact represented
  rows.
- `src/wnn/training_qat.py`: preset configuration, forward substitution, cache
  invalidation, and effective checkpoint export.
- `src/wnn/quantization_export.py`: version-2 packed Q8/Q4 archive read/write
  helpers; version-1 Q8 export functions remain available for old archives.
- `scripts/train_rnn.py`: the two new presets are accepted by `--qat`, and
  checkpoint provenance records their versioned policy.

Focused tests cover Q4 boundaries and invalid values, policy matrix selection,
packed nibble order/padding, exact code/scale archive roundtrip, represented
forward parity, cache refresh, and trainer checkpoint continuation. Passing
these checks establishes implementation behavior only; actual-data profiles,
QAT/control fits, native parity, and quality grading remain owned by the
campaign controller.
