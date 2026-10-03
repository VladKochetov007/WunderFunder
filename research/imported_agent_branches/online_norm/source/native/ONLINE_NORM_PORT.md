# Online-norm inputs in the native engine (2026-09-27)

Runs trained with `--online-norm` feed the GRU `[raw 112 | event 99 | online z-scores of raw 112]`
(`n_in` 323, `build_inputs` in `scripts/train_rnn.py`). The streaming `wnn_native.Predictor` now
builds the same 323 inputs row by row, so such a run can be packaged as a blend member.

## What changed

- `native/src/stream.rs`: `OnlineNorm`, per-sequence running sums of x and x^2 (112 columns),
  emitting `z = (x - mean) / sqrt(max(sumsq/n - mean^2, 0) + 1e-4)` after adding the current row.
  It is reset with the other per-sequence state.
- `native/src/python.rs`: `Predictor(weights, fast_activations=True, online_norm=False)`. With
  `online_norm=True` the z-scores are appended after the 99 event features. A width mismatch fails
  at construction (`model expects 323 inputs, predictor provides 211 (online_norm=false)`). The
  module exports `CAPABILITIES = ("online_norm",)`.
- `scripts/package_submission.py`: `export()` reads the checkpoint args. It refuses models the
  engine cannot run: non-GRU cells, non-linear encoders, `raw_only`, readout heads, or an `n_in`
  other than 211/323. Online-norm members get `"online_norm": true, "n_in": 323` in
  `members.json`. The new `--native PKGDIR` option picks which v2/v3 builds to ship. Packaging an
  online-norm member fails if the chosen build lacks the capability (the default ab18 build does).
- `submission/solution_multi.py`: passes `online_norm=True` only for flagged members, so older
  builds still load linear-only packages.
- `scripts/build_native.sh OUT`: the v2/v3 docker build from `build_submission.sh`, now a separate
  script. Default limits are 2 CPUs, 6 GB and `CARGO_BUILD_JOBS=2` (`WNN_BUILD_CPUS/MEM/JOBS`).
- `src/wnn/features.py`: `online_zscore(x)`, a NumPy twin of the training-time z-scores (below).
- `scripts/parity_native.py`: streaming-vs-PyTorch parity and per-row latency for any runs.
- `native/examples/bench_predictor.rs`: engine-only latency, linear against online-norm pipeline.
- Tests: `stream::tests` (bit-exact float32 prefix computation, two-pass sanity, reset),
  `tests/test_predictor.py` (Predictor with `online_norm` against torch, reset, width error),
  `tests/test_features.py` (NumPy twin is the float32 sequential computation; a CUDA bit-exact
  test that skips without a GPU).

## Accumulator precision: float32 in row order, not f64

The brief asked for f64 accumulators. The engine uses float32 sums in row order instead, because
that is what training computed:

- `online_zscore_torch` runs on the GPU over (B, 20000, 112). CUDA `cumsum` along a non-innermost
  dim runs `tensor_kernel_scan_outer_dim`, a sequential `scalar_t acc` loop per column
  (`torch/include/ATen/native/cuda/ScanUtils.cuh`). The accumulation is float32, in row order.
- CPU torch accumulates a float32 cumsum in double. A CPU "PyTorch reference" is therefore close
  to exact statistics and hides the gap. The first f64 build passed against it (max 2.3e-5) but
  missed what the model saw in training.
- Late in a sequence, float32 accumulation moves some z-scores by up to 1.9 z units against exact
  statistics (row group 1).
- The cached GPU predictions (`cache/pred_ab25_onlinenorm_valid.npy`, from `predict_run.py`)
  confirm this on row group 1. On the 2214 rows where the two choices differ by more than 3e-3,
  the GPU output is 2.1e-3 (mean) from a CPU forward fed float32-sequential z-scores and 4.6e-3
  (max 3.0e-2) from one fed exact z-scores.
- `OnlineNorm` repeats torch's float32 ops in the same order, with no reciprocal or FMA
  shortcuts. `n` is an exact float32 integer below 2^24 rows. `wnn.features.online_zscore`
  (float32 `np.cumsum` is the same sequential loop) is the CPU reference.
- `tests/test_features.py::test_numpy_online_zscore_matches_cuda_bit_for_bit` asserts that the
  NumPy twin equals the CUDA z-scores bit for bit. It has not run yet, because the GPU was reserved
  for training (see the checklist).

## Parity (ab25_onlinenorm; ab66_onorm_f0 did not exist yet)

Setup: shipped v3 build (docker, `x86-64-v3`), fast activations, `valid.parquet` row groups 0, 1
and 2, rows >= 99. The reference is the float32 CPU forward of `make_model` on `build_inputs`,
using training-time z-scores.

| check | rg 0 | rg 1 | rg 2 |
|---|---:|---:|---:|
| max abs native - torch32 (the requirement: <= 1e-3) | 1.43e-6 | 1.76e-6 | 1.67e-6 |
| same, exact activations vs float64 torch | 1.26e-6 | 1.12e-6 | 1.05e-6 |
| linear member ab60_final_ema4_p1_lpft, native - torch32 | 1.63e-6 | 1.43e-6 | 1.91e-6 |
| f64 statistics instead, torch32 output shift (rejected) | 1.59e-3 | 2.67e-2 | 2.56e-3 |
| native vs cached GPU preds (fp16 autocast), ab25 | 5.0e-3 | 1.0e-2 | 8.8e-3 |
| native vs cached GPU preds (fp16 autocast), ab60 linear | 1.0e-2 | 2.0e-2 | 1.5e-2 |

The last two rows are the fp16-autocast noise of `predict_run.py`. It is the same for linear
members, and the online member adds none.

End to end, the package `onorm_2gru_test` (ab25:0.5 + ab60:0.5) was run through
`verify_submission.py` on `cache/valid_subset20.parquet` inside `python:3.11-slim-bookworm
--cpus 1`:

- The packaged blend matches the torch32 blend to at most 1.13e-6 (first 3 sequences).
- Container and host predictions are identical.
- The new build gives the linear 2-GRU (`final_2gru_gclp` members) predictions bit-identical to the
  ab18 build on all 400k rows.
- The repackaged linear weights and `members.json` are byte-identical to
  `submissions/final_2gru_gclp`.

## Latency (single core, loaded host: training was running)

| measure | linear (211) | online (323) | delta |
|---|---:|---:|---:|
| engine only, bench_predictor, min over 21 passes (3 runs) | 34.3-35.2 us | 36.4-37.4 us | +1.5-3.1 us |
| engine only, median | 37.9-39.8 us | 40.9-41.4 us | +1.2-3.5 us |
| OnlineNorm update alone | | 0.11 us | |
| Predictor.step from Python, 1 member, min / median | 42.1 / 47.2 | 43.1 / 46.7 | about +1 |
| 2 members (lin+lin vs onorm+lin), min / median | 90.4 / 95.1 | 93.8 / 96.3 | +1 to +3.4 |
| scorer image, verify_submission, 2 members, 20 seqs | 99.6 | 98.5 | noise |

- The extra cost is 1-3 us/row per online member (2-4 %). Nearly all of it comes from the 112
  dense input columns (128 x 112 more multiply-adds and 57 KB more weights); the running-stat
  update is 0.11 us.
- The linear 2-GRU package ran 36.3 min on the scorer, so the online pair should need at most
  about 1 min more of the 60 min limit.

## Packaging the 2-member submission (run in the main repo after merging this branch)

```bash
# 1. scorer builds with online_norm support (docker, python:3.11-slim-bookworm base)
scripts/build_native.sh submissions/_native_onorm

# 2. parity + latency for the online member and its linear partner (require max <= 1e-3; expect ~2e-6)
taskset -c 3 scripts/limited .venv/bin/python scripts/parity_native.py \
    ab66_onorm_f0 ab64_final_ema4_gclp --native submissions/_native_onorm/v3 --reps 5

# 3. package (coefficients from scripts/pair_eval.py; 0.5/0.5 shown)
.venv/bin/python scripts/package_submission.py final_2gru_onorm \
    ab66_onorm_f0:0.5 ab64_final_ema4_gclp:0.5 --native submissions/_native_onorm
cat submissions/final_2gru_onorm/pkg/members.json   # member 0 has "online_norm": true, "n_in": 323

# 4. scorer-image check, 1 CPU (20 labelled sequences)
docker run --rm --cpus 1 --memory 4g -v "$PWD/scripts:/w/scripts:ro" -v "$PWD/utils.py:/w/utils.py:ro" \
  -v "$PWD/submissions/final_2gru_onorm/submission.zip:/w/submission.zip:ro" \
  -v "$PWD/cache/valid_subset20.parquet:/w/data.parquet:ro" -v /tmp:/out python:3.11-slim-bookworm \
  bash -c "pip install -q numpy==2.2.6 pyarrow==19.0.1 && cd /w && python scripts/verify_submission.py submission.zip data.parquet /out/final_2gru_onorm.npz"
```

When the GPU is free, confirm the NumPy twin is bit-exact to CUDA (it runs one small cumsum):
`.venv/bin/python -m pytest tests/test_features.py -k cuda` and
`scripts/parity_native.py ab66_onorm_f0 --native submissions/_native_onorm/v3 --zdevice cuda`.

The dev extension installed in `.venv` predates this change, so
`test_online_norm_predictor_matches_batch_on_synthetic` skips until it is rebuilt with
`cd native && ../.venv/bin/maturin develop --release`. Linear packages need no rebuild: the new
`solution_multi.py` also runs on the old ab18 builds.
