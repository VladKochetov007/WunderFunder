# enc="conv" in the native engine

`wnn.nn.RNNModel(enc="conv")` now runs in `wnn_native.Predictor` / `GruStepper`:
z = GELU(inp(x_norm)); c = causal depthwise conv (k taps) over the last k rows of z;
z += GELU(conv_pw(c)); then the GRU stack and head(cat[o, z]). Streaming state is the GRU
hidden states plus a ring of the last k pre-residual z rows (zeros before row 0). The ring is
never reset inside a sequence, so the engine computes the true causal conv, not the chunk-reset
version used in training and in `train_rnn.predict`.

## What changed

- `src/rnn.rs`: `ConvEncoder` (k, conv_dw, conv_pw) as an optional part of `RnnWeights`, with shape
  checks. The exact reference model uses a shifted window.
- `src/rnn_fast.rs`: the ring buffer and lag-major taps sit in the aligned arenas. The depthwise
  conv runs as 16-lane register blocks. The pointwise projection is stored column-major and
  accumulated in 32-lane blocks, with no per-row horizontal sums. `reset` clears the ring. Nothing
  allocates per row. The linear path is untouched: the new build gives bit-identical outputs to
  the verified `ab18_wmse_struct` v3 build on linear weights (20k real rows, v2 and v3).
- `src/python.rs`: the conv encoder is selected when `conv_dw.weight` is present, and k is read
  from its shape. Weights the engine does not consume now raise `ValueError` (for example mlp or
  token encoders, or a dual-clock model); before, they were silently ignored.
  `wnn_native.ENCODERS == ["linear", "conv"]` and `Predictor.encoder` expose what the engine runs.
- `scripts/package_submission.py`: members.json records `"enc"` for each member. `--native DIR`
  picks the v2/v3 builds and `--out DIR` sets where the package is written. RUN may also be a path
  to a model.pt. Packaging refuses when a build lacks a member's encoder. The default builds
  (ab18, linear only) still work for linear packages.
- `submission/solution_multi.py`: raises at load if the shipped build cannot run a member's `enc`,
  or if the weights select a different encoder than members.json says.
- `scripts/build_native.sh`: the Docker v2/v3 build moved out of `build_submission.sh` (same image
  and flags; `WNN_BUILD_CPUS` / `WNN_BUILD_MEM` are configurable, defaults 8 / 8g).
- Tests: `native` has cargo tests (FastRnn vs reference for k = 1, 5, 16 and odd hidden sizes; ring
  reset; shape errors) and `tests/test_conv_predictor.py` (synthetic vs torch for k = 1, 4, 16; the
  encoder is selected by weights; unknown weights are rejected; the ab61 checkpoint on 3 valid
  sequences must stay within 1e-4). `scripts/check_conv_native.py` reproduces every number below.

## Parity (ab61_conv_gru_f0, valid row groups 0-2, rows >= 99)

The reference is one full-sequence torch forward in float32 on CPU, which is a true causal conv
with no chunk resets. The native side is the final Docker x86-64-v3 build.

| seq | fast activations (shipped) max / mean abs | exact activations max / mean abs |
|---|---|---|
| 0 | 1.37e-6 / 2.7e-7 | 1.37e-6 / 2.7e-7 |
| 1 | 1.73e-6 / 2.7e-7 | 1.73e-6 / 2.7e-7 |
| 2 | 1.67e-6 / 3.5e-7 | 1.61e-6 / 3.5e-7 |

End to end, a conv + linear blend package (0.5 each) was run through `verify_submission.py` on
these 3 sequences. It differs from the torch blend by at most 1.3e-6, and v2 and v3 outputs are
bitwise equal (block WP 0.733755 for both).

## Chunk reset versus the true stream (torch, same 3 sequences)

Mean |prediction| is 0.444. The table shows |diff| by rows since the last conv restart, with the
GRU state carried across restarts:

| restart every | rows 0-14 | rows 15-99 | rows 100-999 | later | rows with diff > 1e-2 |
|---|---|---|---|---|---|
| 4000 (eval) | max 0.147, mean 3.2e-2 | max 0.030, mean 2.8e-3 | max 0.004, mean 1.8e-4 | mean 1.7e-5 | 0.34% |
| 2000 (TBPTT) | max 0.175, mean 3.2e-2 | max 0.064, mean 2.8e-3 | max 0.008, mean 3.2e-4 | mean 1.6e-4 | 0.79% |

The conv output differs only in the first 15 rows after a restart. The GRU state carries that
perturbation, though, and it decays over about 100 to 1000 rows, so the gap is not confined to
15 rows. Its effect on the metric is negligible. On all 468 sequences of the held-out fold 0:

| regime | global WP | per-sequence WP |
|---|---|---|
| true stream (native engine) | 0.67423 | 0.64856 |
| reset every 4000 | 0.67422 | 0.64856 |
| reset every 2000 | 0.67421 | 0.64854 |

The run's history.json records 0.67417 for epoch 1 (GPU, fp16 autocast, reset every 4000).

## Latency (x86-64-v3 build, i9-11900H, one pinned core, machine shared with GPU training)

Both runs have hidden 128 and 2 layers. The linear run is ab60_final_ema4_p1_lpft.

| measurement | linear | conv | conv overhead |
|---|---|---|---|
| Rust engine only (`bench_fast 128 2 16`, paired 250-row blocks, 4 runs) | 37-41 us/row | 39-43 us/row | median +4.5% to +5.6% (IQR +2.5% .. +7.9%) |
| `Predictor.step` from Python, real valid rows, paired 500-row blocks | 42.0 us/row median | 44.6 us/row median | median +7.0% (IQR +3.1% .. +10.9%) |
| `verify_submission.py`, single-member packages, 60k rows, 7 alternating runs | 53.6 median / 50.0 min | 56.2 median / 55.2 min | +4.9% median, +10.4% min |

The MAC count predicts +7% (18.4k on top of about 256k per row). The whole-run numbers in the
last row are noisy because other jobs share the machine. The paired measurements are the
reliable ones, and they put the overhead at about 5% for the engine and 7% for a Predictor call,
inside the +10% target.

Two optimizations were measured. Storing the pointwise projection column-major reduced the
overhead from +8% to +13% (with the row-major `gemv`) down to about 5%. A 4-row blocked `gemv`
for all layers was also tried and reverted: it gave no measurable gain, and keeping the linear
path unchanged keeps linear outputs bit-identical to the verified builds.

## Packaging a conv member

Run these from the repository root after merging.

```bash
# 1. v2/v3 builds of current native/ in the scorer image (2 cores, 6 GB)
WNN_BUILD_CPUS=2 WNN_BUILD_MEM=6g scripts/build_native.sh submissions/_native_conv
#    (already built from this branch: <worktree>/submissions/_native_conv/{v2,v3}/wnn_native.abi3.so)

# 2. package; members.json gets "enc": "conv" / "linear"; refuses builds without conv support
.venv/bin/python scripts/package_submission.py conv_blend \
    ab61_conv_gru_f0:0.5 ab60_final_ema4_p1_lpft:0.5 --native submissions/_native_conv

# 3. scorer-style check at both levels
for l in v3 v2; do WNN_NATIVE_LEVEL=$l taskset -c 3 scripts/limited .venv/bin/python \
    scripts/verify_submission.py submissions/conv_blend/submission.zip datasets/valid.parquet /tmp/cb_$l.npz; done

# 4. parity, chunk-reset gap, latency (add --score-seqs 468 for the fold-0 WP table, ~12 min)
PYTHONPATH=submissions/_native_conv/v3 taskset -c 3 scripts/limited .venv/bin/python \
    scripts/check_conv_native.py --reps 10
PYTHONPATH=submissions/_native_conv/v3 .venv/bin/python -m pytest -q tests/test_conv_predictor.py
cd native && cargo test -j 2 --release
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo run -j 2 --release --example bench_fast -- 128 2 16
```

The `.venv` editable `wnn_native` is still the old build. Until it is rebuilt
(`.venv/bin/maturin develop --release -m native/Cargo.toml`), `tests/test_conv_predictor.py`
skips unless PYTHONPATH points at a new build, as in step 4.
