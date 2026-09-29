# native_cpp: streaming C++ inference engine

C++17 twin of the research model `wnn.nn.RNNModel` (enc `linear`, GRU cell) plus the causal event
features `wnn.features.StreamingEventFeatures`, evaluated one row at a time for the submission callback.
It replaces the Rust engine in `native/` for code that must stay in C++; Python stays the ML side.

| file | role |
|---|---|
| `include/wnn/event_features.hpp` | 99 causal event features from consecutive rows (exact twin of the Python class) |
| `include/wnn/kernels.hpp` | aligned storage, blocked row-major gemv (16 lanes x 4 rows), sparse transposed saxpy |
| `include/wnn/activations.hpp` | exact and fast (vectorised, error < 5e-6) exp / sigmoid / tanh / erf / GELU |
| `include/wnn/model.hpp` | `Weights` (state_dict names, one 64-byte aligned arena), `Predictor`, `Ensemble` |
| `src/bindings.cpp` | pybind11 module `wnn_cpp` |
| `src/glibc_compat.c` | keeps the module loadable on glibc 2.36 (scorer image) |
| `build.sh` | builds `build/{v2,avx2,v3}/wnn_cpp<EXT_SUFFIX>`, static libstdc++, reports max glibc version |
| `package_cpp.py` | submission ZIP: cp311 builds + `submission/solution_cpp.py` + exported weights |
| `tests/test_parity.py` | streaming C++ vs full-sequence PyTorch float32 on validation sequences |
| `bench/bench_kernels.cpp` | per-kernel timings at production shapes |

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
| glibc symbols needed by the cp311 builds | <= GLIBC_2.36 (bookworm) |
| one member, Python step loop, v3 | C++ 30-32 us/row, Rust 25-27 us/row (same core, interleaved, min of 5) |
| scorer simulation (`verify_submission.py`, 2 members, 3 sequences) | runs, level v3 picked, governor active |

Not yet verified inside the real scorer image (no docker access on beleriand). Next speed levers:
explicit AVX2/AVX-512 intrinsics for the 384x128 GRU gemv and fp16 weight storage (halves the ~1 MB
per-row weight stream, which exceeds L2 on this host).
