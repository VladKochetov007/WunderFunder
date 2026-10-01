# native_cpp: streaming C++ inference engine

C++17 twin of the research model `wnn.nn.RNNModel` (enc `linear`, GRU cell) plus the causal event
features `wnn.features.StreamingEventFeatures`, evaluated one row at a time for the submission callback.
It replaces the Rust engine in `native/` for code that must stay in C++; Python stays the ML side.

| file | role |
|---|---|
| `include/wnn/event_features.hpp` | 99 causal event features from consecutive rows (exact twin of the Python class) |
| `include/wnn/kernels.hpp` | aligned storage, blocked row-major gemv (16 lanes x 4 rows), sparse transposed saxpy, fp16-weight gemv (F16C) |
| `include/wnn/online_norm.hpp` | causal float32 z-scores for `--online-norm` members (bit-exact CUDA cumsum order) |
| `include/wnn/activations.hpp` | exact and fast (vectorised, error < 5e-6) exp / sigmoid / tanh / erf / GELU |
| `include/wnn/model.hpp` | `Weights` (state_dict names, one 64-byte aligned arena), `Predictor`, `Ensemble` |
| `src/bindings.cpp` | pybind11 module `wnn_cpp` |
| `src/glibc_compat.c` | keeps the module loadable on glibc 2.36 (scorer image) |
| `build.sh` | builds `build/{v2,avx2,v3}/wnn_cpp<EXT_SUFFIX>`, static libstdc++, reports max glibc version |
| `package_cpp.py` | submission ZIP: cp311 builds + `submission/solution_cpp.py` + exported weights |
| `tests/test_parity.py` | streaming C++ vs full-sequence PyTorch float32 on validation sequences |
| `bench/bench_kernels.cpp` | per-kernel timings at production shapes |
| `tools/scorer_env.sh` | runnable scorer image (python:3.11-slim-bookworm, glibc 2.36) without docker |

```bash
native_cpp/build.sh ~/venv311/bin/python v2 avx2 v3           # scorer builds (Python 3.11)
native_cpp/build.sh .venv/bin/python v3                        # local research interpreter
PYTHONPATH=src:native_cpp/build/v3 .venv/bin/python native_cpp/tests/test_parity.py RUN
.venv/bin/python native_cpp/package_cpp.py NAME RUN_A:0.5 RUN_B:0.5
~/venv311/bin/python scripts/verify_submission.py submissions/NAME/NAME.zip DATA.parquet OUT.npz
```

Python API: `wnn_cpp.Predictor(weights_dict, fast_activations=True)` and
`wnn_cpp.Ensemble([weights_dict, ...], [coef, ...], fast_activations=True)`, both with `reset()`,
`step(row_float32_112) -> (2,)` and `run(x_T_by_112) -> (T, 2)`.

## Measured (beleriand, Xeon Gold 6230, one core, shared host, 2026-09-29)

| check | result |
|---|---|
| parity vs PyTorch float32 (ref_f0, 3 sequences, fast activations) | max abs 2.5e-6, corr 1.00000000 |
| parity vs the Rust engine (`native/`, same weights) | max abs 1.4e-6 |
| parity vs PyTorch for an `--online-norm` member (z-scores from CUDA) | max abs 9.5e-7 (fp16 weights 1.1e-4) |
| glibc symbols needed by the cp311 builds | <= GLIBC_2.36 (bookworm) |
| one member, Python step loop, v3, hidden 128 | C++ fp16 17 us, C++ fp32 32 us, Rust 27 us (same core, min of 3-5) |
| same, hidden 192 / 256 (weights exceed L2) | fp16 44 / 85 us, fp32 90 / 157 us, Rust 91 / 156 us |
| fp16 weights vs fp32, 60 validation sequences | WP delta -0.000001, max abs prediction diff 1.2e-3 |
| hidden 256 (h256_f0), 60 sequences: fp32 / fp16 / int8 per-row | 155 / 85 / 72 us/row; WP delta 0 / +1e-6 / -1.2e-5; max abs diff 0 / 5.5e-4 / 2.1e-2 |
| scorer simulation (`verify_submission.py`, same 2 members, 3 sequences, one core) | C++ fp16 60 us/row, Rust 101 us/row; global WP 0.754262 for both |
| projected time for a validation-sized test on this host (39.4M rows) | C++ 39 min, Rust 66 min (the scorer CPU is faster than this host) |

`int8_weights=True` (per-row symmetric int8, v3 only) is experimental: 16% faster than fp16 at hidden 256
with 40x larger prediction deviations, so it is not used by default.

fp16 weight storage (`half_weights=True`, used by `solution_cpp.py`) is active on the v3 build only; v2 and
avx2 builds run fp32, so CPU levels agree to ~1e-3 rather than 1e-6 (no measurable WP change).
Verified in the scorer image userland (`tools/scorer_env.sh`: bookworm glibc 2.36, Python 3.11, pinned
numpy/pyarrow): all three levels load and give global WP 0.754262 on the 3-sequence smoke set, like the
Rust engine; v3 fp16 57.8 us/row, avx2 100.2, v2 131.8, Rust v3 99.9 (2 members, same core).
